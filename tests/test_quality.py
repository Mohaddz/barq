"""Review signals are narrow, non-mutating checks rather than quality verdicts."""

from copy import deepcopy
import unittest

from barq.rules import review_checks


def review(prompt, answer, *, source="fixture", task="dialogue"):
    return review_checks([
        {"role": "user", "content": prompt}, {"role": "assistant", "content": answer},
    ], [], source, task)


class QualityChecksTests(unittest.TestCase):
    def test_diacritization_preserves_letters_hamza_punctuation_and_other_marks(self):
        prefix = "أضف التشكيل إلى النص التالي:\n\n"
        good = review(prefix + "ا\u0654كل الطالب.", "أَكَلَ الطَّالِبُ.", task="diacritization")
        self.assertEqual(good["flags"], [])
        self.assertEqual(good["checks"], ["diacritization"])
        for before, after in (("أكل", "اَكَلَ"), ("آكل", "اَكِلٌ"),
                              ("كتب.", "كَتَبَ!"), ("café", "cafe")):
            with self.subTest(before=before):
                result = review(prefix + before, after, task="diacritization")
                self.assertIn("underlying_text_changed", result["flags"])

    def test_diacritization_empty_output_numeric_input_and_unknown_wrapper(self):
        prefix = "أضف التشكيل إلى النص التالي:\n\n"
        unchanged = review(prefix + "كتب الطالب", "كتب الطالب", task="diacritization")
        self.assertEqual(unchanged["flags"], ["no_diacritics_added"])
        numeric = review(prefix + "123 (45)", "123 (45)", task="diacritization")
        self.assertEqual(numeric["flags"], [])
        unknown = review("شكّل هذا النص: كتب الطالب", "نص آخر", task="diacritization")
        self.assertEqual(unknown["flags"], [])
        self.assertEqual(unknown["checks"], ["diacritization_skipped_unknown_wrapper"])

    def test_sentiment_checks_vocabulary_only_for_known_source_and_wrapper(self):
        prompt = "ما هو شعور النص التالي؟\n\nأحب هذا المكان"
        for label in ("سلبي", "إيجابي", "ايجابي", "محايد", " ا\u0655يجابي "):
            result = review(prompt, label, source="twitter_sentiment", task="sentiment_analysis")
            self.assertEqual(result["flags"], [])  # Even a semantically wrong valid label passes.
            self.assertEqual(result["checks"], ["sentiment_label"])
        invalid = review(prompt, "سعيد", source="twitter_sentiment", task="sentiment_analysis")
        self.assertEqual(invalid["flags"], ["invalid_sentiment_label"])
        self.assertEqual(review(prompt, "سعيد", task="sentiment_analysis")["checks"], [])
        unknown = review("صنّف المشاعر", "سعيد", source="twitter_sentiment", task="sentiment_analysis")
        self.assertNotIn("sentiment_label", unknown["checks"])
        self.assertEqual(unknown["flags"], [])

    def test_python_syntax_and_function_structure_without_executing_code(self):
        prompt = "Please write a Python function that doubles a number."
        good = review(prompt, "مثال:\n```python\ndef double(x):\n    return 2 * x\n```\nشرح المثال")
        self.assertEqual(good["flags"], [])
        self.assertEqual(good["checks"], ["python_syntax"])
        self.assertEqual(review(prompt, "def double(:")["flags"], ["python_syntax_invalid"])
        self.assertEqual(review(prompt, "print(2)")["flags"], ["python_function_missing"])
        self.assertEqual(review(prompt, "# TODO")["flags"], ["python_code_empty"])
        program = review("اكتب برنامج بلغة بايثون", 'raise AssertionError("must not execute")')
        self.assertEqual(program["flags"], [])
        self.assertEqual(program["checks"], ["python_syntax"])

    def test_python_does_not_grade_explanation_repair_or_oversized_code(self):
        for prompt in ("Explain how to write a Python function.", "Fix this Python program.",
                       "اشرح هذه الدالة في بايثون", "اكتب قصة عن مبرمج يستخدم Python"):
            self.assertNotIn("python_syntax", review(prompt, "نص توضيحي")["checks"])
        no_call = review("اكتب لي برنامج بايثون لحساب المتوسط",
                         "هذا السؤال لا يتطلب استدعاء أي أداة.",
                         source="TuwaiqAcademy/AISA-ArabicFC", task="tool_use")
        self.assertEqual(no_call["flags"], [])
        self.assertNotIn("python_syntax", no_call["checks"])
        large = review("Write a Python program.", "#" * 100_001)
        self.assertEqual(large["flags"], ["python_check_skipped_too_long"])
        self.assertNotIn("python_syntax", large["checks"])

    def test_creative_hint_and_source_boilerplate_are_distinct_and_non_mutating(self):
        creative = review("اكتب قصة قصيرة عن طفل فضولي", "كان يا مكان")
        self.assertEqual(creative["task_hint"], "creative_writing")
        self.assertEqual(creative["flags"], [])
        paragraph = review("اكتب فقرة قصيرة من الرواية عن الموضوع التالي: شروق الشمس", "بدأ الصباح")
        self.assertEqual(paragraph["task_hint"], "creative_writing")
        self.assertIsNone(review("اكتب فقرة عن أهمية القراءة", "القراءة مفيدة")["task_hint"])
        messages = [{"role": "developer", "content": "احتفظ باللهجة والتشكيل"},
                    {"role": "user", "content": "لخّص: هٰذا خبر. مواضيع قد تهمك نهاية"},
                    {"role": "assistant", "content": "الخَبَرُ مُوجَزٌ."}]
        tools = [{"function": {"name": "sample", "parameters": {"type": "object"}}}]
        original = deepcopy((messages, tools))
        result = review_checks(messages, tools, "news", "summarization")
        self.assertEqual(result["flags"], ["source_boilerplate"])
        self.assertEqual(result["checks"], ["source_boilerplate"])
        self.assertEqual((messages, tools), original)


if __name__ == "__main__":
    unittest.main()
