import unittest
from pathlib import Path


class ProjectAuditTest(unittest.TestCase):
    def test_audit_reports_core_training_project_areas(self) -> None:
        from vggt_project.project_audit import audit_project_files

        report = audit_project_files(Path("."))
        items_by_name = {item.name: item for item in report.items}

        for name in (
            "data_processing",
            "model_framework",
            "losses",
            "train_loop",
            "eval_loop",
            "environment",
            "weights",
            "dataset_setup",
            "reference_setup",
            "github_ci",
            "github_publish",
            "benchmarks",
        ):
            self.assertIn(name, items_by_name)
            self.assertTrue(items_by_name[name].ready, name)
        self.assertIn("scripts/prepare_model_weights.sh", items_by_name["weights"].evidence)
        self.assertIn("scripts/configure_model_weights.py", items_by_name["weights"].evidence)
        self.assertIn("scripts/verify_training_artifacts.py", items_by_name["environment"].evidence)
        self.assertIn("scripts/prepare_occ3d.sh", items_by_name["dataset_setup"].evidence)
        self.assertIn("scripts/attach_occ3d_labels.py", items_by_name["dataset_setup"].evidence)
        self.assertIn("scripts/prepare_satellite_rasters.sh", items_by_name["dataset_setup"].evidence)
        self.assertIn("src/vggt_project/data/occ3d_labels.py", items_by_name["benchmarks"].evidence)
        self.assertIn("scripts/export_occupancy_predictions.py", items_by_name["benchmarks"].evidence)
        self.assertIn("scripts/evaluate_occupancy_benchmark.py", items_by_name["benchmarks"].evidence)

    def test_audit_report_is_not_complete_real_training(self) -> None:
        from vggt_project.project_audit import audit_project_files, format_audit_report

        report = audit_project_files(Path("."))
        rendered = format_audit_report(report)

        self.assertFalse(report.real_training_complete)
        self.assertIn("satellite patch extraction", " ".join(report.remaining_gaps))
        self.assertNotIn("multi-camera depth/pointmap", " ".join(report.remaining_gaps))
        self.assertNotIn("camera-specific reconstruction heads are not implemented", " ".join(report.remaining_gaps))
        self.assertNotIn("GitHub upload still requires", " ".join(report.remaining_gaps))
        self.assertIn("scripts/report_real_training_preflight.py", " ".join(report.next_actions))
        self.assertIn("PYTHONPATH=src python3 scripts/report_training_launch.py", " ".join(report.next_actions))
        self.assertIn("scripts/check_external_assets.py", " ".join(report.next_actions))
        self.assertIn("PYTHONPATH=src python3 scripts/plan_training_run.py", " ".join(report.next_actions))
        self.assertIn("scripts/probe_manifest_forward.py", " ".join(report.next_actions))
        self.assertIn("scripts/verify_training_artifacts.py", " ".join(report.next_actions))
        self.assertIn("scripts/check_training_readiness.py", " ".join(report.next_actions))
        self.assertIn("scripts/prepare_model_weights.sh", " ".join(report.next_actions))
        self.assertIn("scripts/configure_model_weights.py", " ".join(report.next_actions))
        self.assertIn("scripts/prepare_occ3d.sh", " ".join(report.next_actions))
        self.assertIn("scripts/attach_occ3d_labels.py", " ".join(report.next_actions))
        self.assertIn("scripts/prepare_satellite_rasters.sh", " ".join(report.next_actions))
        self.assertIn("next_actions:", rendered)


if __name__ == "__main__":
    unittest.main()
