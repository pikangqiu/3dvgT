import unittest
from pathlib import Path


class CIWorkflowTest(unittest.TestCase):
    def test_github_actions_ci_runs_lightweight_project_checks(self) -> None:
        workflow = Path(".github/workflows/ci.yml")

        self.assertTrue(workflow.exists())
        content = workflow.read_text(encoding="utf-8")
        self.assertIn("python3 -m unittest discover -s tests -v", content)
        self.assertIn("scripts/audit_project_status.py", content)
        self.assertIn("scripts/setup_references.py --dry-run", content)
        self.assertIn("scripts/bootstrap_training_run.py --help", content)
        self.assertIn("scripts/inspect_checkpoint.py --help", content)
        self.assertIn("scripts/prepare_model_weights.sh --help", content)
        self.assertIn("scripts/configure_model_weights.py --help", content)
        self.assertIn("scripts/check_external_assets.py --help", content)
        self.assertIn("scripts/report_real_training_preflight.py --help", content)


if __name__ == "__main__":
    unittest.main()
