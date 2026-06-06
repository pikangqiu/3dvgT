import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EnvironmentReportTest(unittest.TestCase):
    def test_environment_report_records_dependency_and_accelerator_status(self) -> None:
        from vggt_project.environment_report import (
            DependencySnapshot,
            TorchRuntimeSnapshot,
            collect_environment_report,
        )

        report = collect_environment_report(
            dependency_probe=lambda names: tuple(
                DependencySnapshot(name=name, available=name != "missing_pkg", version="1.0" if name != "missing_pkg" else None)
                for name in names
            ),
            torch_runtime_probe=lambda: TorchRuntimeSnapshot(
                importable=True,
                version="2.2.0",
                cuda_available=False,
                cuda_device_count=0,
                mps_available=True,
            ),
            dependency_names=("torch", "numpy", "missing_pkg"),
        )

        self.assertFalse(report.ready)
        self.assertIn("missing_pkg", report.missing_dependencies)
        self.assertEqual(report.torch_runtime.version, "2.2.0")
        self.assertTrue(report.torch_runtime.mps_available)

    def test_environment_report_cli_writes_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "environment.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/report_environment.py",
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
        self.assertIn("python_version", stdout_payload)
        self.assertIn("dependencies", stdout_payload)
        self.assertEqual(stdout_payload, output_payload)


if __name__ == "__main__":
    unittest.main()
