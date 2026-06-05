import tempfile
import unittest
from pathlib import Path

from vggt_project.experiments import ExperimentRunConfig
from vggt_project.training_readiness import (
    DependencyStatus,
    check_training_readiness,
)


class TrainingReadinessTest(unittest.TestCase):
    def test_readiness_reports_missing_split_manifests(self) -> None:
        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            train_manifest_path=Path("missing-train.jsonl"),
            eval_manifest_path=Path("missing-val.jsonl"),
            device="cpu",
        )

        report = check_training_readiness(
            config,
            dependency_probe=lambda: (
                DependencyStatus("torch", True, "2.0"),
                DependencyStatus("PIL", True, "10.0"),
            ),
            device_probe=lambda device: True,
        )

        self.assertFalse(report.ready)
        self.assertIn("train_manifest_path", report.missing_paths)
        self.assertIn("eval_manifest_path", report.missing_paths)
        self.assertEqual(report.device, "cpu")

    def test_readiness_passes_when_manifests_dependencies_and_device_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            train_manifest.write_text("", encoding="utf-8")
            eval_manifest.write_text("", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                device="cpu",
            )

            report = check_training_readiness(
                config,
                dependency_probe=lambda: (
                    DependencyStatus("torch", True, "2.0"),
                    DependencyStatus("PIL", True, "10.0"),
                    DependencyStatus("yaml", True, "6.0"),
                ),
                device_probe=lambda device: True,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.missing_paths, {})
        self.assertEqual(report.missing_dependencies, ())


if __name__ == "__main__":
    unittest.main()
