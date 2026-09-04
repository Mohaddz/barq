"""Non-mutating structural checks and exact-overlap keys for text training data.

These checks do not grade Arabic, factual correctness, or tool-call semantics.
Tool schemas support nested JSON types, required fields, enums, and additional
properties. Other JSON Schema constraints are left to a full schema validator.
"""

import ast
import hashlib
import json
import math
import re
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


_VOWELS = re.compile("[\u064b-\u0652\u0670]")
_DIACRITIZATION_PREFIX = "أضف التشكيل إلى النص التالي:\n\n"
_SENTIMENT_PREFIX = "ما هو شعور النص التالي؟\n\n"
_REQUEST_START = r"(?:^|\n)\s*(?:(?:من فضلك|لو سمحت|please|can you|could you)\s*[,،]?\s*)?"
_PYTHON_REQUEST = re.compile(
    _REQUEST_START + r"(?:[اأ]كتب|أنشئ|انشئ|برمج)\s+(?:لي\s+)?"
    r"(?:دالة|وظيفة|برنامج|كود|شيفرة|شفرة)\b|"
    + _REQUEST_START + r"(?:write|create|implement|build)\s+(?:(?:me|a|an|the)\s+){0,2}"
    r"(?:python(?:\s*3)?\s+)?(?:function|program|script|code)\b", re.IGNORECASE,
)
_CREATIVE_REQUEST = re.compile(
    _REQUEST_START + r"[اأ]كتب\s+(?:لي\s+)?(?:قصة|قصّة|قصيدة|رواية|حكاية|مشهد)\b|"
    + _REQUEST_START + r"[اأ]كتب\s+(?:لي\s+)?(?:فقرة|نصا|نصًا)\b[^\n]{0,60}"
    r"\b(?:ال)?(?:قصة|رواية|قصيدة)\b|"
    + _REQUEST_START + r"write\s+(?:(?:me|a|an|the)\s+){0,2}(?:story|poem|novel)\b",
    re.IGNORECASE,
)


def _base_text(text: str) -> str:
    # Keep hamza/maddah and every other mark; stripping all combining marks is unsafe.
    return " ".join(_VOWELS.sub("", unicodedata.normalize("NFC", text)).split())


def review_checks(messages: list[dict], tools: list[dict], source: str, task: str) -> dict:
    """Return human-review signals, never filtering, repairing or executing content.

    Checks name the rules applied, for conditional denominators. A *_skipped_* check
    records an unsupported input; it is not an evaluated example. Python checks only
    syntax/structure, and sentiment checks only the declared label vocabulary.
    """
    flags, checks = [], []
    messages = messages if isinstance(messages, list) else []
    users = [m["content"] for m in messages if isinstance(m, dict)
             and m.get("role") == "user" and isinstance(m.get("content"), str)]
    prompt = users[-1] if users else ""
    final = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    answer = final.get("content") if final.get("role") == "assistant" else None
    hint = "creative_writing" if _CREATIVE_REQUEST.search(prompt) else None
    if not isinstance(answer, str):
        return {"flags": flags, "checks": checks, "task_hint": hint}

    if task == "diacritization":
        if not prompt.startswith(_DIACRITIZATION_PREFIX):
            checks.append("diacritization_skipped_unknown_wrapper")
        else:
            checks.append("diacritization")
            text = prompt[len(_DIACRITIZATION_PREFIX):]
            if _base_text(text) != _base_text(answer):
                flags.append("underlying_text_changed")
            has_arabic = any(unicodedata.category(c).startswith("L")
                             and "ARABIC" in unicodedata.name(c, "") for c in text)
            if has_arabic and not _VOWELS.search(answer):
                flags.append("no_diacritics_added")

    if source == "twitter_sentiment" and task == "sentiment_analysis":
        if not prompt.startswith(_SENTIMENT_PREFIX):
            checks.append("sentiment_label_skipped_unknown_wrapper")
        else:
            checks.append("sentiment_label")
            if unicodedata.normalize("NFC", answer).strip() not in {"سلبي", "إيجابي", "ايجابي", "محايد"}:
                flags.append("invalid_sentiment_label")

    if task in {"summarization", "translation"}:
        checks.append("source_boilerplate")
        if "مواضيع قد تهمك نهاية" in prompt:
            flags.append("source_boilerplate")

    if (task != "tool_use" and _PYTHON_REQUEST.search(prompt)
            and re.search(r"\bpython(?:3)?\b|بايثون|بيثون", prompt, re.I)
            and not final.get("tool_calls")):
        blocks = re.findall(r"```[ \t]*(?:python3?|py)[ \t]*\r?\n(.*?)```", answer, re.I | re.S)
        code = "\n\n".join(blocks) if blocks else answer
        if len(code) > 100_000:
            flags.append("python_check_skipped_too_long")
            checks.append("python_syntax_skipped_too_long")
        else:
            try:
                tree = ast.parse(code)
            except (MemoryError, RecursionError):
                flags.append("python_check_skipped_parser_limit")
                checks.append("python_syntax_skipped_parser_limit")
            except (SyntaxError, ValueError):
                checks.append("python_syntax")
                flags.append("python_syntax_invalid")
            else:
                checks.append("python_syntax")
                meaningful = [node for node in tree.body if not isinstance(node, ast.Pass)
                              and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
                if not meaningful:
                    flags.append("python_code_empty")
                elif re.search(r"\bfunction\b|دالة|وظيفة", prompt, re.I) and not any(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.walk(tree)
                ):
                    flags.append("python_function_missing")
    return {"flags": flags, "checks": checks, "task_hint": hint}
