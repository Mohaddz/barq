"""Non-mutating structural checks and exact-overlap keys for text training data.

These checks do not grade Arabic, factual correctness, or tool-call semantics.
Tool schemas support nested JSON types, required fields, enums, and additional
properties. Other JSON Schema constraints are left to a full schema validator.
"""

import hashlib
import json
import math
import unicodedata


_ROLES = {"system", "developer", "user", "assistant", "tool"}
_JSON_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprints(messages: list[dict], tools: list[dict]) -> tuple[str, str]:
    """Return (example hash, input-group hash) for JSON-serializable records.

    The input key omits only the final assistant message, retaining previous
    assistant/tool turns and tool definitions. Call validate_example first;
    invalid final roles are retained rather than silently discarded.
    """
    context = messages
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        context = messages[:-1]
    return (
        _json_hash({"messages": messages, "tools": tools}),
        _json_hash({"messages": context, "tools": tools}),
    )


def benchmark_key(text: str) -> str:
    """Hash NFC text with collapsed whitespace, preserving Arabic distinctions."""
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def benchmark_texts(messages: list[dict]) -> list[str]:
    """Return nonempty user texts, unchanged, for exact benchmark overlap checks."""
    return [
        message["content"]
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]


def _valid_schema(schema: object) -> bool:
    if isinstance(schema, bool):
        return True
    if not isinstance(schema, dict):
        return False
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else types
        if not isinstance(types, list) or not types:
            return False
        if any(not isinstance(kind, str) or kind not in _JSON_TYPES for kind in types):
            return False
    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
            return False
        if len(required) != len(set(required)):
            return False
    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict) or any(
            not isinstance(key, str) or not _valid_schema(value) for key, value in properties.items()
        ):
            return False
    for keyword in ("items", "additionalProperties"):
        if keyword in schema and not _valid_schema(schema[keyword]):
            return False
    return "enum" not in schema or isinstance(schema["enum"], list) and bool(schema["enum"])


def _matches_type(value: object, kind: str) -> bool:
    if kind == "null":
        return value is None
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return not isinstance(value, bool) and (
            isinstance(value, int)
            or isinstance(value, float) and math.isfinite(value) and value.is_integer()
        )
    if kind == "number":
        return not isinstance(value, bool) and (
            isinstance(value, int) or isinstance(value, float) and math.isfinite(value)
        )
    return isinstance(value, {"object": dict, "array": list, "string": str}[kind])


def _argument_reasons(value: object, schema: dict | bool) -> list[str]:
    if isinstance(schema, bool):
        return [] if schema else ["tool_argument_schema"]
    reasons = []
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    if types and not any(_matches_type(value, kind) for kind in types):
        return ["tool_argument_type"]
    # JSON booleans must not compare equal to numeric enum values (True == 1).
    if "enum" in schema and not any(
        value == option and isinstance(value, bool) == isinstance(option, bool)
        for option in schema["enum"]
    ):
        reasons.append("tool_argument_enum")
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            reasons.append("missing_required_tool_argument")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                reasons.extend(_argument_reasons(item, properties[key]))
            elif additional is False:
                reasons.append("unexpected_tool_argument")
            elif isinstance(additional, dict):
                reasons.extend(_argument_reasons(item, additional))
    elif isinstance(value, list) and "items" in schema:
        for item in value:
            reasons.extend(_argument_reasons(item, schema["items"]))
    return reasons


def _tool_schemas(tools: list[dict], reasons: list[str]) -> dict[str, dict]:
    schemas = {}
    if not isinstance(tools, list):
        reasons.append("invalid_tools")
        return schemas
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            reasons.append("invalid_tool_definition")
            continue
        function = tool.get("function", tool)
        if not isinstance(function, dict):
            reasons.append("invalid_tool_definition")
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            reasons.append("invalid_tool_definition")
            continue
        if name in schemas:
            reasons.append("duplicate_tool_name")
        schema = function.get("parameters", {"type": "object"})
        if not isinstance(schema, dict) or not _valid_schema(schema):
            reasons.append("invalid_tool_schema")
            continue
        if schema.get("type", "object") != "object":
            reasons.append("invalid_tool_schema")
            continue
        schemas[name] = schema
    return schemas


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON number: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON argument key")
        result[key] = value
    return result


def _valid_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_valid_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _valid_json_value(item) for key, item in value.items())
    return False


def _call_reasons(calls: object, schemas: dict[str, dict]) -> list[str]:
    if not isinstance(calls, list):
        return ["invalid_tool_calls"]
    reasons = []
    for call in calls:
        if not isinstance(call, dict) or call.get("type", "function") != "function":
            reasons.append("invalid_tool_call")
            continue
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            reasons.append("invalid_tool_call")
            continue
        name = function["name"]
        if not name.strip():
            reasons.append("invalid_tool_call")
        elif name not in schemas:
            reasons.append("unknown_tool")
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(
                    arguments, parse_constant=_reject_constant, object_pairs_hook=_unique_object
                )
            except (ValueError, RecursionError):
                reasons.append("invalid_tool_arguments")
                continue
        if not isinstance(arguments, dict) or not _valid_json_value(arguments):
            reasons.append("invalid_tool_arguments")
        elif name in schemas:
            reasons.extend(_argument_reasons(arguments, schemas[name]))
    return reasons


def validate_example(
    messages: list[dict], tools: list[dict], *, allow_missing_target: bool = False
) -> list[str]:
    """Return unique reason codes for structural defects; never rewrite inputs.

    Accept text conversations and OpenAI-style assistant function calls. An
    assistant may have empty/null content when its nonempty tool-call list is
    valid. Tool schemas may use OpenAI wrappers or flat name/parameters objects.
    For blind holdouts, allow_missing_target also accepts an input-only history
    or an empty final assistant placeholder. It does not relax input validation.
    This deliberately does not validate conversation/tool execution semantics.
    """
    reasons: list[str] = []
    schemas = _tool_schemas(tools, reasons)
    if not isinstance(messages, list) or not messages:
        return list(dict.fromkeys([*reasons, "invalid_messages"]))
    roles = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            reasons.append("invalid_message")
            continue
        role = message.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            reasons.append("invalid_role")
            continue
        roles.append(role)
        content = message.get("content")
        calls = message.get("tool_calls", [])
        if role == "assistant":
            placeholder = (
                allow_missing_target
                and index == len(messages) - 1
                and (content is None or isinstance(content, str) and not content.strip())
                and (calls is None or calls == [])
            )
            call_reasons = _call_reasons([] if placeholder else calls, schemas)
            reasons.extend(call_reasons)
            if content is not None and not isinstance(content, str):
                reasons.append("invalid_message_content")
            if not placeholder and (not isinstance(content, str) or not content.strip()):
                if not isinstance(calls, list) or not calls or call_reasons:
                    reasons.append("empty_assistant")
        else:
            if not isinstance(content, str) or role == "user" and not content.strip():
                reasons.append("invalid_message_content")
            if "tool_calls" in message:
                reasons.append("invalid_tool_calls")
    if "user" not in roles:
        reasons.append("missing_user")
    if not allow_missing_target and "assistant" not in roles:
        reasons.append("missing_assistant")
    if not allow_missing_target and (
        not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant"
    ):
        reasons.append("final_message_not_assistant")
    return list(dict.fromkeys(reasons))
