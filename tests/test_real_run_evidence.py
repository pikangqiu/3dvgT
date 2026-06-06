import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RealRunEvidenceTest(unittest.TestCase):
    def test_real_run_evidence_accepts_ready_preflight_and_artifacts(self) -> None:
        from vggt_project.real_run_evidence import verify_real_run_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight.json"
            artifacts = root / "artifacts.json"
            preflight.write_text(json.dumps({"ready_for_real_training": True}), encoding="utf-8")
            artifacts.write_text(json.dumps({"ready": True}), encoding="utf-8")

            report = verify_real_run_evidence(
                preflight_report_path=preflight,
                artifact_report_path=artifacts,
                environment_report_path=None,
                expected_git_commit="abc123",
                require_clean_worktree=True,
                git_commit_probe=lambda: "abc123",
                git_status_probe=lambda: "",
            )

        self.assertTrue(report.ready)
        self.assertTrue(report.preflight_ready)
        self.assertTrue(report.artifacts_ready)
        self.assertTrue(report.clean_worktree)
        self.assertEqual(report.git_commit, "abc123")
        self.assertEqual(report.errors, ())

    def test_real_run_evidence_rejects_unready_artifacts(self) -> None:
        from vggt_project.real_run_evidence import verify_real_run_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight.json"
            artifacts = root / "artifacts.json"
            environment = root / "environment.json"
            preflight.write_text(json.dumps({"ready_for_real_training": True}), encoding="utf-8")
            artifacts.write_text(json.dumps({"ready": False, "errors": ["missing checkpoint"]}), encoding="utf-8")
            environment.write_text(json.dumps({"ready": True}), encoding="utf-8")

            report = verify_real_run_evidence(
                preflight_report_path=preflight,
                artifact_report_path=artifacts,
                environment_report_path=environment,
                git_commit_probe=lambda: "abc123",
                git_status_probe=lambda: "",
            )

        self.assertFalse(report.ready)
        self.assertFalse(report.artifacts_ready)
        self.assertTrue(report.environment_ready)
        self.assertIn("artifact report is not ready", report.errors)

    def test_real_run_evidence_rejects_unready_environment_report(self) -> None:
        from vggt_project.real_run_evidence import verify_real_run_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight.json"
            artifacts = root / "artifacts.json"
            environment = root / "environment.json"
            preflight.write_text(json.dumps({"ready_for_real_training": True}), encoding="utf-8")
            artifacts.write_text(json.dumps({"ready": True}), encoding="utf-8")
            environment.write_text(json.dumps({"ready": False, "missing_dependencies": ["torch"]}), encoding="utf-8")

            report = verify_real_run_evidence(
                preflight_report_path=preflight,
                artifact_report_path=artifacts,
                environment_report_path=environment,
                git_commit_probe=lambda: "abc123",
                git_status_probe=lambda: "",
            )

        self.assertFalse(report.ready)
        self.assertFalse(report.environment_ready)
        self.assertIn("environment report is not ready", report.errors)

    def test_real_run_evidence_rejects_unready_model_device_report(self) -> None:
        from vggt_project.real_run_evidence import verify_real_run_evidence

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight.json"
            artifacts = root / "artifacts.json"
            model_device = root / "model_device.json"
            preflight.write_text(json.dumps({"ready_for_real_training": True}), encoding="utf-8")
            artifacts.write_text(json.dumps({"ready": True}), encoding="utf-8")
            model_device.write_text(json.dumps({"ready": False, "errors": ["cuda unavailable"]}), encoding="utf-8")

            report = verify_real_run_evidence(
                preflight_report_path=preflight,
                artifact_report_path=artifacts,
                model_device_report_path=model_device,
                git_commit_probe=lambda: "abc123",
                git_status_probe=lambda: "",
            )

        self.assertFalse(report.ready)
        self.assertFalse(report.model_device_ready)
        self.assertIn("model device report is not ready", report.errors)

    def test_real_run_evidence_cli_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight.json"
            artifacts = root / "artifacts.json"
            output = root / "evidence.json"
            preflight.write_text(json.dumps({"ready_for_real_training": True}), encoding="utf-8")
            artifacts.write_text(json.dumps({"ready": True}), encoding="utf-8")
            (root / "environment.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
            (root / "model_device.json").write_text(json.dumps({"ready": True}), encoding="utf-8")
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/verify_real_run_evidence.py",
                    "--preflight-report",
                    str(preflight),
                    "--artifact-report",
                    str(artifacts),
                    "--environment-report",
                    str(root / "environment.json"),
                    "--model-device-report",
                    str(root / "model_device.json"),
                    "--output",
                    str(output),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            stdout_payload = json.loads(result.stdout)
            output_payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(stdout_payload["ready"])
        self.assertTrue(stdout_payload["environment_ready"])
        self.assertTrue(stdout_payload["model_device_ready"])
        self.assertEqual(stdout_payload["model_device_report_path"], str(root / "model_device.json"))
        self.assertEqual(stdout_payload, output_payload)


if __name__ == "__main__":
    unittest.main()
