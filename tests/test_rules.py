import copy
import unittest

from barq.rules import benchmark_key, benchmark_texts, fingerprints, validate_example


def example(answer="الجواب"):
    return [{"role": "user", "content": "السؤال"}, {"role": "assistant", "content": answer}]


def tool_definition():
    return {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "unit": {"enum": ["C", "F"]},
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }


def tool_example(arguments='{"city":"الرياض"}', name="weather"):
    return [
        {"role": "user", "content": "كيف الجو بالرياض؟"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}
            ],
        },
    ]


class ValidationTests(unittest.TestCase):
    def test_text_and_roles_are_unchanged(self):
        messages = [
            {"role": "system", "content": ""},
            {"role": "developer", "content": "حافظ على التشكيل"},
            {"role": "user", "content": "صحّح: الطالبان ذهب إلى المدرسه"},
            {"role": "assistant", "content": "ذَهَبَ الطَّالِبَانِ إِلَى الْمَدْرَسَةِ."},
            {"role": "user", "content": "وش قلت؟"},
            {"role": "assistant", "content": "قلت راحوا المدرسة"},
        ]
        original = copy.deepcopy(messages)
        self.assertEqual(validate_example(messages, []), [])
        self.assertEqual(messages, original)

    def test_factual_accuracy_is_not_a_structural_rule(self):
        self.assertEqual(validate_example(example("2 + 2 = 5"), []), [])

    def test_valid_empty_or_null_tool_assistant(self):
        for content in ("", None):
            messages = tool_example()
            messages[-1]["content"] = content
            tools = [tool_definition()]
            original = copy.deepcopy((messages, tools))
            self.assertEqual(validate_example(messages, tools), [])
            self.assertEqual((messages, tools), original)

    def test_flat_tool_and_object_arguments(self):
        self.assertEqual(
            validate_example(tool_example({"city": "الرياض"}), [tool_definition()["function"]]), []
        )

    def test_missing_roles_and_target(self):
        self.assertIn("missing_user", validate_example([{"role": "assistant", "content": "x"}], []))
        self.assertIn("missing_assistant", validate_example([{"role": "user", "content": "x"}], []))
        messages = example() + [{"role": "user", "content": "متابعة"}]
        self.assertIn("final_message_not_assistant", validate_example(messages, []))

    def test_empty_target_requires_valid_calls(self):
        self.assertIn("empty_assistant", validate_example(example("  "), []))
        self.assertIn("empty_assistant", validate_example(tool_example("bad JSON"), [tool_definition()]))

    def test_blind_holdout_allows_missing_target_or_empty_placeholder(self):
        prompt = [{"role": "developer", "content": "استخدم الأدوات"}, {"role": "user", "content": "الجو؟"}]
        self.assertEqual(validate_example(prompt, [tool_definition()], allow_missing_target=True), [])
        messages = prompt + [{"role": "assistant", "content": "", "tool_calls": None}]
        self.assertEqual(validate_example(messages, [tool_definition()], allow_missing_target=True), [])
        self.assertIn("empty_assistant", validate_example(messages, [tool_definition()]))

    def test_blind_holdout_does_not_relax_input_or_malformed_call_validation(self):
        self.assertIn("missing_user", validate_example([{"role": "system", "content": "x"}], [], allow_missing_target=True))
        self.assertIn("invalid_tool_arguments", validate_example(tool_example("bad JSON"), [tool_definition()], allow_missing_target=True))
        self.assertIn("invalid_tool_definition", validate_example(example(""), [None], allow_missing_target=True))

    def test_malformed_structures_do_not_raise(self):
        cases = [None, {}, [], [None], [{"role": []}], [{"role": "stranger", "content": "x"}]]
        for messages in cases:
            with self.subTest(messages=messages):
                self.assertTrue(validate_example(messages, []))
        self.assertIn("invalid_tools", validate_example(example(), None))
        self.assertIn("invalid_tool_definition", validate_example(example(), [None]))

    def test_unknown_tool_is_distinct_from_bad_json(self):
        reasons = validate_example(tool_example(name="unknown"), [tool_definition()])
        self.assertIn("unknown_tool", reasons)
        self.assertNotIn("invalid_tool_arguments", reasons)

    def test_invalid_call_shapes(self):
        for calls in (None, {}, [None], [{}], [{"function": {"name": []}}]):
            messages = tool_example()
            messages[-1]["tool_calls"] = calls
            self.assertTrue(validate_example(messages, [tool_definition()]))

    def test_arguments_must_be_unambiguous_json_objects(self):
        for arguments in ('{"city":', "[]", "null", '{"days":NaN}', '{"city":"x","city":"y"}', {"days": float("inf")}):
            with self.subTest(arguments=arguments):
                self.assertIn(
                    "invalid_tool_arguments", validate_example(tool_example(arguments), [tool_definition()])
                )

    def test_required_types_enums_and_extra_fields(self):
        cases = [
            ({}, "missing_required_tool_argument"),
            ({"city": 10}, "tool_argument_type"),
            ({"city": "الرياض", "days": True}, "tool_argument_type"),
            ({"city": "الرياض", "unit": "K"}, "tool_argument_enum"),
            ({"city": "الرياض", "extra": 1}, "unexpected_tool_argument"),
        ]
        for arguments, reason in cases:
            with self.subTest(arguments=arguments):
                self.assertIn(reason, validate_example(tool_example(arguments), [tool_definition()]))

    def test_nested_schemas_and_nullable_types(self):
        tool = tool_definition()
        tool["function"]["parameters"]["properties"]["city"] = {
            "type": "array", "items": {"type": ["string", "null"]}
        }
        self.assertEqual(validate_example(tool_example({"city": ["الرياض", None]}), [tool]), [])
        self.assertIn("tool_argument_type", validate_example(tool_example({"city": [10]}), [tool]))

    def test_bad_schema_and_duplicate_tools(self):
        tool = tool_definition()
        tool["function"]["parameters"]["required"] = "city"
        self.assertIn("invalid_tool_schema", validate_example(example(), [tool]))
        self.assertIn("duplicate_tool_name", validate_example(example(), [tool_definition(), tool_definition()]))


class FingerprintTests(unittest.TestCase):
    def test_same_input_different_targets_share_group(self):
        first, first_group = fingerprints(example("نعم"), [])
        second, second_group = fingerprints(example("لا"), [])
        self.assertNotEqual(first, second)
        self.assertEqual(first_group, second_group)

    def test_different_tool_targets_share_group(self):
        first = fingerprints(tool_example({"city": "الرياض"}), [tool_definition()])
        second = fingerprints(tool_example({"city": "جدة"}), [tool_definition()])
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])

    def test_tool_definitions_and_prior_context_affect_group(self):
        original = fingerprints(example(), [tool_definition()])[1]
        changed_tool = tool_definition()
        changed_tool["function"]["description"] = "Weather service"
        self.assertNotEqual(original, fingerprints(example(), [changed_tool])[1])
        messages = example() + example()
        first = fingerprints(messages, [tool_definition()])[1]
        messages[1]["content"] = "رد سابق مختلف"
        self.assertNotEqual(first, fingerprints(messages, [tool_definition()])[1])

    def test_object_key_order_is_canonical(self):
        reordered = [{"content": item["content"], "role": item["role"]} for item in example()]
        self.assertEqual(fingerprints(example(), []), fingerprints(reordered, []))

    def test_benchmark_normalization_is_limited(self):
        self.assertEqual(benchmark_key("  أكل\n الطعام  "), benchmark_key("ا\u0654كل الطعام"))
        self.assertNotEqual(benchmark_key("عَلَم"), benchmark_key("علم"))
        self.assertNotEqual(benchmark_key("أكل"), benchmark_key("اكل"))
        self.assertNotEqual(benchmark_key("فتى"), benchmark_key("فتي"))
        self.assertNotEqual(benchmark_key("مدرسة"), benchmark_key("مدرسه"))

    def test_benchmark_texts_only_user_messages_unchanged(self):
        messages = [
            {"role": "system", "content": "تعليمات"},
            {"role": "user", "content": " عَرَبِيّ\n"},
            {"role": "assistant", "content": "الرد"},
            {"role": "tool", "content": "نتيجة"},
            {"role": "user", "content": "وش أخبارك؟"},
        ]
        self.assertEqual(benchmark_texts(messages), [" عَرَبِيّ\n", "وش أخبارك؟"])


if __name__ == "__main__":
    unittest.main()
