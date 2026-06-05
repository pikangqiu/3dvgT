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

    def test_audit_report_is_not_complete_real_training(self) -> None:
        from vggt_project.project_audit import audit_project_files

        report = audit_project_files(Path("."))

        self.assertFalse(report.real_training_complete)
        self.assertIn("satellite patch extraction", " ".join(report.remaining_gaps))
        self.assertNotIn("multi-camera depth/pointmap", " ".join(report.remaining_gaps))
        self.assertNotIn("camera-specific reconstruction heads are not implemented", " ".join(report.remaining_gaps))
        self.assertNotIn("GitHub upload still requires", " ".join(report.remaining_gaps))


if __name__ == "__main__":
    unittest.main()
