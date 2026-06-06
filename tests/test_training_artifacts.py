import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TrainingArtifactsTest(unittest.TestCase):
    def test_verify_training_artifacts_accepts_complete_result_bundle(self) -> None:
        from vggt_project.training_artifacts import verify_training_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "outputs/model.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint placeholder")
            experiment_report = root / "outputs/report.json"
            experiment_report.write_text(
                json.dumps(
                    {
                        "mode": "manifest-smoke",
                        "train_metrics": {
                            "loss": 1.0,
                            "checkpoint": str(checkpoint),
                        },
                        "eval_metrics": {
                            "loss": 0.5,
                            "depth_mae": 0.25,
                            "pointmap_l1": 0.75,
                        },
                    }
                ),
                encoding="utf-8",
            )
            occupancy_report = root / "outputs/occupancy.json"
            occupancy_report.write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "class_iou": {"0": 1.0, "1": 0.5},
                        "occupancy_miou": 0.75,
                    }
                ),
                encoding="utf-8",
            )

            report = verify_training_artifacts(
                checkpoint_path=checkpoint,
                experiment_report_path=experiment_report,
                occupancy_report_path=occupancy_report,
                required_eval_metrics=("depth_mae", "pointmap_l1"),
                required_occupancy_class_count=2,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.present_artifacts, ("checkpoint", "experiment_report", "occupancy_report"))
        self.assertEqual(report.missing_artifacts, ())
        self.assertEqual(report.errors, ())

    def test_verify_training_artifacts_rejects_incomplete_occupancy_class_iou(self) -> None:
        from vggt_project.training_artifacts import verify_training_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            train_metrics = root / "train_metrics.json"
            train_metrics.write_text(json.dumps({"loss": 1.0}), encoding="utf-8")
            eval_metrics = root / "eval_metrics.json"
            eval_metrics.write_text(json.dumps({"loss": 0.5}), encoding="utf-8")
            occupancy_report = root / "occupancy.json"
            occupancy_report.write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "class_iou": {"0": 1.0, "1": 0.5},
                        "occupancy_miou": 0.75,
                    }
                ),
                encoding="utf-8",
            )

            report = verify_training_artifacts(
                checkpoint_path=checkpoint,
                train_metrics_path=train_metrics,
                eval_metrics_path=eval_metrics,
                occupancy_report_path=occupancy_report,
                required_occupancy_class_count=3,
            )

        self.assertFalse(report.ready)
        self.assertIn("occupancy_report class_iou has 2 classes, expected 3", report.errors)

    def test_verify_training_artifacts_rejects_invalid_occupancy_metric_values(self) -> None:
        from vggt_project.training_artifacts import verify_training_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            train_metrics = root / "train_metrics.json"
            train_metrics.write_text(json.dumps({"loss": 1.0}), encoding="utf-8")
            eval_metrics = root / "eval_metrics.json"
            eval_metrics.write_text(json.dumps({"loss": 0.5}), encoding="utf-8")
            occupancy_report = root / "occupancy.json"
            occupancy_report.write_text(
                json.dumps(
                    {
                        "sample_count": 1,
                        "class_iou": {"0": 1.0, "1": 1.25},
                        "occupancy_miou": "bad",
                    }
                ),
                encoding="utf-8",
            )

            report = verify_training_artifacts(
                checkpoint_path=checkpoint,
                train_metrics_path=train_metrics,
                eval_metrics_path=eval_metrics,
                occupancy_report_path=occupancy_report,
                required_occupancy_class_count=2,
            )

        self.assertFalse(report.ready)
        self.assertIn("occupancy_report occupancy_miou must be numeric in [0, 1]", report.errors)
        self.assertIn("occupancy_report class_iou[1] must be numeric in [0, 1]", report.errors)

    def test_verify_training_artifacts_reports_missing_eval_metric(self) -> None:
        from vggt_project.training_artifacts import verify_training_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            experiment_report = root / "report.json"
            experiment_report.write_text(
                json.dumps(
                    {
                        "train_metrics": {"loss": 1.0},
                        "eval_metrics": {"loss": 0.5},
                    }
                ),
                encoding="utf-8",
            )

            report = verify_training_artifacts(
                checkpoint_path=checkpoint,
                experiment_report_path=experiment_report,
                required_eval_metrics=("depth_mae",),
            )

        self.assertFalse(report.ready)
        self.assertIn("experiment_report.eval_metrics missing required metric: depth_mae", report.errors)

    def test_verify_training_artifacts_accepts_separate_train_and_eval_metric_files(self) -> None:
        from vggt_project.training_artifacts import verify_training_artifacts

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            train_metrics = root / "train_metrics.json"
            train_metrics.write_text(json.dumps({"loss": 1.0, "checkpoint": str(checkpoint)}), encoding="utf-8")
            eval_metrics = root / "eval_metrics.json"
            eval_metrics.write_text(json.dumps({"loss": 0.5, "depth_mae": 0.25}), encoding="utf-8")

            report = verify_training_artifacts(
                checkpoint_path=checkpoint,
                train_metrics_path=train_metrics,
                eval_metrics_path=eval_metrics,
                required_eval_metrics=("depth_mae",),
            )

        self.assertTrue(report.ready)
        self.assertIn("train_metrics", report.present_artifacts)
        self.assertIn("eval_metrics", report.present_artifacts)
        self.assertEqual(report.eval_metrics, ("depth_mae", "loss"))

    def test_verify_training_artifacts_cli_json_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            experiment_report = root / "report.json"
            experiment_report.write_text(
                json.dumps(
                    {
                        "train_metrics": {
                            "loss": 1.0,
                            "checkpoint": str(checkpoint),
                        },
                        "eval_metrics": {
                            "loss": 0.5,
                            "depth_mae": 0.25,
                        },
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_training_artifacts.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--experiment-report",
                    str(experiment_report),
                    "--required-eval-metric",
                    "depth_mae",
                    "--required-occupancy-class-count",
                    "2",
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ready"])
        self.assertIn("checkpoint", payload["present_artifacts"])


if __name__ == "__main__":
    unittest.main()
