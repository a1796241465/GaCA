import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.pipeline import (
    Action,
    Config,
    PipelineError,
    PipelineRunner,
    Stage,
    build_pipeline,
)


ROOT = Path(__file__).resolve().parents[1]


def config(root, **overrides):
    values = {
        "data_root": root / "data",
        "output_root": root / "outputs",
        "device": "cuda",
        "workers": 4,
        "seed": 42,
        "resume": False,
        "force": False,
    }
    values.update(overrides)
    return Config(**values)


def keys(stages):
    return [action.key for stage in stages for action in stage.actions]


class PipelineDefinitionTests(unittest.TestCase):
    def test_gaca_stage_order_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            stages = build_pipeline("gaca", config(Path(temp)))
        self.assertEqual(
            [stage.name for stage in stages],
            [
                "Preparing CARE data",
                "Downloading AlphaFold structures",
                "Preprocessing and validating sequences",
                "Extracting ESM-2 embeddings",
                "Extracting ESM-IF1 embeddings",
                "Building feature stores",
                "Exporting the effective dataset",
                "Training and evaluating GaCA",
            ],
        )
        action_keys = keys(stages)
        self.assertEqual(action_keys[-1], "gaca_train")
        for excluded in ("ablations_standard", "classical_baselines", "lightgbm", "blast_database", "figures"):
            self.assertNotIn(excluded, action_keys)

    def test_full_contains_paper_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            action_keys = keys(build_pipeline("full", config(Path(temp))))
        for required in (
            "gaca_train",
            "ablations_standard",
            "ablation_cross_attention",
            "classical_baselines",
            "lightgbm",
            "blast_database",
            "foldseek_search_lt30",
            "collect_metrics",
            "paired_statistics",
            "figures",
        ):
            self.assertIn(required, action_keys)
        self.assertFalse(any("clean" in key for key in action_keys))

    def test_verify_has_no_expensive_actions(self):
        with tempfile.TemporaryDirectory() as temp:
            action_keys = keys(build_pipeline("verify", config(Path(temp))))
        self.assertEqual(action_keys, ["verify_archived", "release_guard_tests"])
        forbidden = ("download", "train", "embedding", "blast", "foldseek")
        self.assertFalse(any(word in key for key in action_keys for word in forbidden))

    def test_subprocess_failure_stops_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = PipelineRunner(config(Path(temp)), root=ROOT)
            stages = [
                Stage("failure", [Action("fail", [sys.executable, "-c", "raise SystemExit(7)"])]),
                Stage("must not run", [Action("later", [sys.executable, "-c", "raise SystemExit(0)"])]),
            ]
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                runner.run(stages)
            self.assertEqual(caught.exception.returncode, 7)

    def test_resume_rejects_invalid_recorded_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = config(root, resume=True)
            output = root / "result.txt"
            output.write_text("valid", encoding="utf-8")
            action = Action("recorded", [sys.executable, "-c", "pass"], outputs=(output,))
            runner = PipelineRunner(cfg, root=ROOT)
            receipt = runner.receipt_path(action)
            receipt.parent.mkdir(parents=True)
            receipt.write_text(
                '{"schema": 1, "command": ["different"], "inputs": {}, "outputs": {}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PipelineError, "Cannot resume recorded"):
                runner.execute_action(action)

    def test_gaca_help(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "reproduce.py"), "gaca", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ("--data-root", "--output-root", "--device", "--workers", "--seed", "--resume", "--force"):
            self.assertIn(option, result.stdout)


class PipelineRecoveryTests(unittest.TestCase):
    def test_fresh_care_download_preserves_empty_clone_destination(self):
        from data_download import download_care
        with tempfile.TemporaryDirectory() as temp:
            cfg = config(Path(temp))
            runner = PipelineRunner(cfg)
            action = build_pipeline("gaca", cfg)[0].actions[0]
            raw = cfg.data_root / "raw" / "care"

            def fake_git(command, cwd=None):
                self.assertEqual(command[:2], ["git", "clone"])
                self.assertFalse(raw.exists() and any(raw.iterdir()))
                for name in download_care.EXPECTED_SPLITS:
                    path = raw / "splits" / "task1" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("Entry,Sequence,EC number\nA,AA,1.1.1.1\n")

            def invoke_downloader(command):
                with patch.object(sys, "argv", ["download_care", *command[3:]]):
                    download_care.main()

            with patch.object(runner, "check_tools"), patch.object(
                runner, "execute_process", side_effect=invoke_downloader
            ), patch.object(download_care, "run", side_effect=fake_git), patch.object(
                download_care.subprocess, "check_output", return_value="fake-revision\n"
            ):
                self.assertTrue(runner.execute_action(action))
            self.assertTrue(runner.receipt_path(action).is_file())

    def test_resume_binds_command_inputs_and_outputs(self):
        for mutation in ("none", "input", "output", "missing_output", "command"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, output = root / "input.txt", root / "output.txt"
                source.write_text("original")
                output.write_text("complete")
                action = Action("bound", [sys.executable, "-c", "pass"],
                                inputs=(source,), outputs=(output,))
                runner = PipelineRunner(config(root, resume=True))
                runner.write_receipt(action)
                if mutation == "input":
                    source.write_text("changed")
                elif mutation == "output":
                    output.write_text("damaged")
                elif mutation == "missing_output":
                    output.unlink()
                elif mutation == "command":
                    action.command.append("changed-parameter")
                with patch.object(runner, "execute_process") as execute:
                    if mutation == "none":
                        self.assertFalse(runner.execute_action(action))
                    else:
                        with self.assertRaises(PipelineError):
                            runner.execute_action(action)
                    execute.assert_not_called()

    def test_failed_or_interrupted_force_invalidates_previous_completion(self):
        for error in (subprocess.CalledProcessError(7, ["test"]), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                output = root / "result.txt"
                output.write_text("old complete output")
                action = Action("rerun", [sys.executable, "-c", "pass"],
                                outputs=(output,), recovery="rerun")
                runner = PipelineRunner(config(root, force=True))
                runner.write_receipt(action)
                with patch.object(runner, "execute_process", side_effect=error):
                    with self.assertRaises(type(error)):
                        runner.execute_action(action)
                self.assertFalse(runner.receipt_path(action).exists())
                resumed = PipelineRunner(config(root, resume=True))
                with patch.object(resumed, "execute_process") as execute:
                    self.assertTrue(resumed.execute_action(action))
                    execute.assert_called_once()

    def test_force_recovers_malformed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = PipelineRunner(config(Path(temp), force=True))
            action = Action("broken", [sys.executable, "-c", "pass"])
            receipt = runner.receipt_path(action)
            receipt.parent.mkdir(parents=True)
            receipt.write_text("{")
            with patch.object(runner, "execute_process") as execute:
                self.assertTrue(runner.execute_action(action))
                execute.assert_called_once()
            self.assertEqual(runner.receipt(action)["schema"], 1)

    def test_validator_failure_does_not_record_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            runner = PipelineRunner(config(Path(temp), force=True))
            action = Action("invalid", [sys.executable, "-c", "pass"],
                            validator=lambda: (_ for _ in ()).throw(ValueError("invalid output")))
            runner.write_receipt(action)
            with patch.object(runner, "execute_process"):
                with self.assertRaisesRegex(ValueError, "invalid output"):
                    runner.execute_action(action)
            self.assertFalse(runner.receipt_path(action).exists())

    def test_failure_preserves_stderr_and_stops_later_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / "later.txt"
            code = (
                "import sys; from pathlib import Path; "
                "from scripts.pipeline import Action, Config, PipelineRunner, Stage; "
                "r=Path(sys.argv[1]); "
                "a=Action('failure',[sys.executable,'-c',"
                "\"import sys; print('child-error-marker',file=sys.stderr); sys.exit(7)\"]); "
                "b=Action('later',[sys.executable,'-c',"
                "\"from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')\",str(r/'later.txt')]); "
                "PipelineRunner(Config(r/'data',r/'outputs')).run([Stage('first',[a]),Stage('second',[b])])"
            )
            result = subprocess.run([sys.executable, "-B", "-c", code, str(root)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("child-error-marker", result.stderr)
            self.assertFalse(marker.exists())
            self.assertFalse((root / "outputs/.reproduce/receipts/failure.json").exists())

    def test_missing_retrieval_executable_is_clear(self):
        for tool in ("makeblastdb", "blastp", "foldseek"):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temp:
                runner = PipelineRunner(config(Path(temp)))
                action = Action("external", [tool], required_tools=(tool,))
                with patch("scripts.pipeline.shutil.which", return_value=None), patch.object(
                    runner, "execute_process"
                ) as execute:
                    with self.assertRaisesRegex(PipelineError, "not found on PATH: " + tool):
                        runner.execute_action(action)
                    execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
