import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline import Action, Config, PipelineRunner, validate_retrieval


class RecordingRunner(PipelineRunner):
    def __init__(self, config):
        super().__init__(config)
        self.commands = []

    def execute_process(self, command):
        self.commands.append(command)


class PipelineValidatorTests(unittest.TestCase):
    def test_retrieval_validation_accepts_first_completed_subset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "metrics_lt30.json").write_text("{}", encoding="utf-8")
            (root / "predictions_lt30.csv").write_text("placeholder", encoding="utf-8")
            with patch("scripts.pipeline.validate_predictions") as validate:
                validate_retrieval(root)
            validate.assert_called_once_with(root / "predictions_lt30.csv", 243)

    def test_force_overwrite_is_execution_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = Config(root / "data", root / "outputs", force=True)
            runner = RecordingRunner(config)
            action = Action("feature_stores", ["python", "builder"])
            runner.execute_action(action)
            self.assertEqual(runner.commands, [["python", "builder", "--overwrite"]])
            receipt = runner.receipt(action)
            self.assertEqual(receipt["command"], ["python", "builder"])


if __name__ == "__main__":
    unittest.main()
