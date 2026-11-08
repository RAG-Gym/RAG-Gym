import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import locate_answer, normalize_answer_multichoice


class MultipleChoiceScoringTests(unittest.TestCase):
    def score_label(self, prediction):
        return normalize_answer_multichoice(locate_answer(prediction))

    def test_explicit_choice_takes_precedence_over_capitalized_prose(self):
        for prediction in ["Answer choice: D", "Based on the evidence, option D is correct."]:
            with self.subTest(prediction=prediction):
                self.assertEqual(self.score_label(prediction), "D")

    def test_supported_answer_formats(self):
        for choice in "ABCD":
            for template in ["{}", "  {}  ", "({})", "'{}'", "Option {}", "Answer: {}",
                             "The answer is {}", r"\boxed{{{}}}", '{{"answer_choice": "{}"}}']:
                with self.subTest(choice=choice, template=template):
                    self.assertEqual(self.score_label(template.format(choice)), choice)

    def test_words_are_not_choice_labels(self):
        for prediction in ["Cannot determine", "DNA", "Based on insufficient evidence", "Answer unavailable"]:
            with self.subTest(prediction=prediction):
                self.assertEqual(self.score_label(prediction), prediction)

    def test_ambiguous_fallback_is_not_guessed(self):
        prediction = "Considering B versus D"
        self.assertEqual(self.score_label(prediction), prediction)

    def evaluate_saved_predictions(self, data, samples):
        # Exercise the real CLI and its saved-prediction directory convention.
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / data / "direct" / "tiny-model" / "10"
            predictions.mkdir(parents=True)
            for index, (prediction, answer) in enumerate(samples):
                (predictions / f"{index}.json").write_text(
                    json.dumps({"prediction": prediction, "answer": answer}), encoding="utf-8"
                )
            result = subprocess.run(
                [sys.executable, "src/evaluate.py", "--data", data, "--llm_name", "tiny-model",
                 "--save_dir", directory],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            return ast.literal_eval(result.stdout.split("Metrics: ", 1)[1].strip())

    def test_medqa_cli_recognizes_correct_explicit_answers(self):
        samples = [(choice, choice) for choice in "ABCD"] + [
            ("Answer choice: D", "D"), ("Based on the evidence, option D is correct.", "D")
        ]
        self.assertEqual(self.evaluate_saved_predictions("medqa", samples), {"acc": "1.0000"})

    def test_medqa_cli_does_not_reward_incidental_capitals(self):
        samples = [("Answer choice: D", "A"), ("Based on the evidence, option D is correct.", "B"),
                   ("Cannot determine", "C")]
        self.assertEqual(self.evaluate_saved_predictions("medqa", samples), {"acc": "0.0000"})

    def test_open_qa_cli_metrics_are_unchanged(self):
        samples = [("Paris", "Paris"), ("London", "Paris")]
        for data in ["hotpotqa", "2wikimultihopqa", "bamboogle"]:
            with self.subTest(data=data):
                self.assertEqual(self.evaluate_saved_predictions(data, samples),
                                 {"em": "0.5000", "f1": "0.5000", "prec": "0.5000", "recall": "0.5000"})


if __name__ == "__main__":
    unittest.main()
