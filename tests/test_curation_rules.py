"""Offline lexical signals stay review-only and never rewrite source examples."""

from copy import deepcopy
import json
import unittest

from barq.rules import curation_checks, validate_example


def conversation(prompt, answer):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]


def call(arguments, *, name="lookup"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}},
    ]}


def tool(properties=None):
    return {"function": {"name": "lookup", "parameters": {
        "type": "object", "properties": properties or {}, "required": [],
    }}}


def curate(messages, tools=None, *, source="fixture", task="instruction_following"):
    return curation_checks(messages, tools or [], source, task)


class CurationSignalsTests(unittest.TestCase):
    def test_reuses_existing_checks_and_preserves_arabic_and_reasoning_fields(self):
        messages = conversation("أضف التشكيل إلى النص التالي:\n\nأكل الطالب.", "أكل الطالب.")
        messages[-1]["think"] = "أُبْقِي التَّشْكِيلَ واللهجة وشلون كما هي"
        tools = [tool()]
        original = deepcopy((messages, tools))
        result = curate(messages, tools, task="diacritization")
        self.assertIn("no_diacritics_added", result["flags"])
        self.assertIn("assistant_reasoning_fields_present", result["flags"])
        self.assertIn("diacritization", result["checks"])
        self.assertEqual((messages, tools), original)

    def test_replacement_character_in_nested_message_data_is_review_signal(self):
        messages = [{"role": "user", "content": "ابحث عن الاسم"}, call({"query": "اسم�"})]
        self.assertIn("replacement_character_present", curate(messages)["flags"])
        self.assertNotIn("replacement_character_present", curate(conversation("رَمْز؟", "نعم"))["flags"])

    def test_exact_aisa_no_call_target_is_a_format_signal_not_a_wrong_call_label(self):
        generic = "هذا السؤال لا يتطلب استدعاء أي أداة."
        messages = conversation("صباح الخير", generic)
        result = curate(messages, source="TuwaiqAcademy/AISA-ArabicFC", task="tool_use")
        self.assertIn("aisa_generic_no_call_target", result["flags"])
        self.assertNotIn("tool_argument_tokens_unverified", result["flags"])
        self.assertNotIn("aisa_generic_no_call_target", curate(messages, task="tool_use")["flags"])
        messages[-1]["content"] = "يقول المصدر: " + generic
        self.assertNotIn("aisa_generic_no_call_target", curate(
            messages, source="TuwaiqAcademy/AISA-ArabicFC", task="tool_use")["flags"])

    def test_aisa_normalized_think_fields_are_detected_without_removal(self):
        messages = [{"role": "user", "content": "رقم الإقامة ١٢٣"}, call({"iqama_number": "123"})]
        messages[-1]["_think_for_train"] = "الرقم مذكور"
        original = deepcopy(messages)
        result = curate(messages, [tool()], source="TuwaiqAcademy/AISA-ArabicFC", task="tool_use")
        self.assertEqual(result["flags"], ["assistant_reasoning_fields_present"])
        self.assertEqual(messages, original)

    def test_unicode_digits_numeric_forms_and_leading_zero_identifiers(self):
        messages = [{"role": "user", "content": "رقم ۰۰۱۲۳ والمبلغ ١٬٥٠٠٫٥ لثلاثة وأعني ٣ أشخاص"},
                    call({"id_number": "00123", "amount": 1500.5, "guests": 3.0})]
        self.assertNotIn("tool_argument_tokens_unverified", curate(messages, [tool()])["flags"])
        messages[-1] = call({"id_number": "123"})
        self.assertIn("tool_argument_tokens_unverified", curate(messages, [tool()])["flags"])

    def test_assistant_claims_and_future_messages_do_not_ground_earlier_call(self):
        messages = conversation("تحقق من الإقامة", "رقمها 789012") + [call({"iqama_number": "789012"}),
                    {"role": "user", "content": "789012"}]
        self.assertIn("tool_argument_tokens_unverified", curate(messages, [tool()])["flags"])
        grounded = [{"role": "user", "content": "تحقق"},
                    {"role": "tool", "content": "الرقم 789012"}, call({"iqama_number": "789012"})]
        self.assertNotIn("tool_argument_tokens_unverified", curate(grounded, [tool()])["flags"])

    def test_dates_arithmetic_and_written_quantities_remain_unverified_not_errors(self):
        for prompt, arguments in [
            ("التاريخ 2025-01-01، احجز غدًا", {"date": "2025-01-02"}),
            ("اجمع 20 و30", {"amount": 50}),
            ("ثلاثة أشخاص", {"guests": 3}),
        ]:
            with self.subTest(arguments=arguments):
                result = curate([{"role": "user", "content": prompt}, call(arguments)], [tool()])
                self.assertEqual(result["flags"], ["tool_argument_tokens_unverified"])
        same_date = [{"role": "user", "content": "يوم ٢٠٢٥-٠١-٠٢"}, call({"date": "2025-01-02"})]
        self.assertNotIn("tool_argument_tokens_unverified", curate(same_date, [tool()])["flags"])

    def test_direct_schema_defaults_and_nested_json_arguments(self):
        definition = tool({"days": {"type": "integer", "default": 1},
                           "entry": {"type": "object", "properties": {"id_number": {"type": "string"}}}})
        arguments = {"days": 1, "entry": {"id_number": "0012"}, "enabled": False, "extra": None}
        messages = [{"role": "user", "content": "رقم ٠٠١٢"}, call(json.dumps(arguments))]
        self.assertEqual(validate_example(messages, [definition]), [])
        self.assertNotIn("tool_argument_tokens_unverified", curate(messages, [definition])["flags"])
        definition["function"]["parameters"]["properties"]["days"]["default"] = True
        self.assertIn("tool_argument_tokens_unverified", curate(messages, [definition])["flags"])

    def test_argument_failure_is_skipped_and_left_for_structural_validator(self):
        messages = [{"role": "user", "content": "ابحث"}, call('{"amount":NaN}')]
        result = curate(messages, [tool()])
        self.assertIn("tool_argument_token_grounding_skipped_invalid_arguments", result["checks"])
        self.assertNotIn("tool_argument_token_grounding", result["checks"])
        self.assertIn("invalid_tool_arguments", validate_example(messages, [tool()]))

    def test_extreme_numeric_text_is_explicitly_skipped_not_silently_accepted(self):
        for text in ["1e999999999999999999999999", "9" * 300, "1e-999999"]:
            with self.subTest(text=text[:30]):
                result = curate(conversation("تلخيص المقال: " + text, "ملخص"), task="summarization")
                self.assertIn("numeric_token_check_skipped_limit", result["flags"])
                self.assertIn("transformation_numeric_grounding_skipped_numeric_limit", result["checks"])
                self.assertNotIn("transformation_numeric_grounding", result["checks"])

    def test_huge_integer_arguments_cannot_overflow_or_silently_pass(self):
        messages = [{"role": "user", "content": "احسب"}, call({"amount": 10 ** 400})]
        self.assertEqual(validate_example(messages, [tool()]), [])
        result = curate(messages, [tool()])
        self.assertEqual(result["flags"], ["numeric_token_check_skipped_limit"])
        self.assertIn("tool_argument_token_grounding_skipped_numeric_limit", result["checks"])
        self.assertNotIn("tool_argument_token_grounding", result["checks"])

    def test_transformation_numbers_use_supported_source_boundaries_only(self):
        cases = [
            ("translation", "ترجم النص التالي من الإنجليزية إلى العربية:\n\n", "There are 12 items.", "هناك ١٣ عنصرًا."),
            ("summarization", "لخّص النص التالي بشكل موجز:\n\n", "حضر 12 طالبًا.", "حضر 13 طالبًا."),
            ("dialect_translation", "ترجم النص التالي إلى اللغة العربية الفصحى:\n\n", "معي 12 كتاب", "لدي 13 كتابًا."),
            ("grammar_correction", "صحّح الأخطاء النحوية في النص التالي:\n\n", "كان 12 طالب", "كان هناك 13 طالبًا."),
        ]
        for task, prefix, source, answer in cases:
            with self.subTest(task=task):
                result = curate(conversation(prefix + source, answer), task=task)
                self.assertIn("transformation_numbers_unverified", result["flags"])
                self.assertIn("transformation_numeric_grounding", result["checks"])
        equivalent = curate(conversation("تلخيص المقال: حضر ١٢ طالبًا.", "حضر 12 طالبًا."), task="summarization")
        self.assertNotIn("transformation_numbers_unverified", equivalent["flags"])

    def test_unknown_transformation_wrapper_and_numeric_word_conversion_do_not_prove_error(self):
        unknown = curate(conversation("اشرح ثم ترجم: 12", "13"), task="translation")
        self.assertIn("transformation_numeric_grounding_skipped_unknown_wrapper", unknown["checks"])
        self.assertNotIn("transformation_numbers_unverified", unknown["flags"])
        written = curate(conversation("ترجم النص التالي إلى اللغة العربية الفصحى:\n\nعندك طاولة لأربع أشخاص؟",
                                     "هل لديك طاولة لـ4 أشخاص؟"), task="dialect_translation")
        self.assertEqual(written["flags"], ["transformation_numbers_unverified"])

    def test_explicit_standalone_start_end_and_comma_constraints(self):
        prompt = ('اكتب قصة قصيرة.\nابدأ القصة حرفيًا بـ «في البداية».\n'
                  'اختم القصة حرفيًا بـ «تمت».\nلا تستخدم الفواصل العربية أو الإنجليزية.')
        good = curate(conversation(prompt, "في البداية ظهر القمر. تمت"))
        self.assertEqual(good["flags"], [])
        self.assertEqual(good["task_hint"], "creative_writing")
        for check in ["literal_start_constraint", "literal_end_constraint", "forbidden_comma_constraint"]:
            self.assertIn(check, good["checks"])
        bad = curate(conversation(prompt, "بدأت، الحكاية ثم انتهت."))
        self.assertEqual(set(bad["flags"]), {"literal_start_mismatch", "literal_end_mismatch", "forbidden_comma_present"})

    def test_unrecognized_or_embedded_creative_constraints_are_not_guessed(self):
        for prompt in ["اكتب قصة تبدأ بعبارة صباح الخير وتنتهي بعبارة وداعًا.",
                       "اكتب قصة قصيرة.\nيقول الراوي: ابدأ القصة حرفيًا بـ «نعم».",
                       "اكتب قصة قصيرة.\nابدأ القصة حرفيًا بـ «نعم».\nإلا إذا لم يناسب السياق"]:
            with self.subTest(prompt=prompt):
                result = curate(conversation(prompt, "حكاية"))
                self.assertEqual(result["flags"], [])
                self.assertIn("instruction_constraints_skipped_semantic_review", result["checks"])
                self.assertNotIn("literal_start_constraint", result["checks"])

    def test_inherited_python_check_never_executes_the_target(self):
        result = curate(conversation("اكتب برنامج بايثون", 'raise AssertionError("must not execute")'))
        self.assertIn("python_syntax", result["checks"])
        self.assertEqual(result["flags"], [])


if __name__ == "__main__":
    unittest.main()
