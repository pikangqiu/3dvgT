import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OccupancyLabelValidationTest(unittest.TestCase):
    def test_validate_occupancy_labels_reports_class_histogram(self) -> None:
        from vggt_project.occupancy_label_validation import validate_occupancy_label_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "labels").mkdir()
            (root / "labels/sample-1.json").write_text(
                json.dumps([[0, 1, 17], [255, 2, 2]]),
                encoding="utf-8",
            )
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "occupancy_path": "labels/sample-1.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_occupancy_label_manifest(
                manifest,
                num_classes=18,
                ignore_index=255,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.voxel_count, 5)
        self.assertEqual(report.class_histogram["2"], 2)
        self.assertEqual(report.ignored_count, 1)
        self.assertEqual(report.errors, ())

    def test_validate_occupancy_labels_rejects_out_of_range_class_ids(self) -> None:
        from vggt_project.occupancy_label_validation import validate_occupancy_label_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.json").write_text(json.dumps([[0, 18]]), encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps({"token": "sample-1", "occupancy_path": "target.json"}) + "\n",
                encoding="utf-8",
            )

            report = validate_occupancy_label_manifest(
                manifest,
                num_classes=18,
                ignore_index=None,
            )

        self.assertFalse(report.ready)
        self.assertIn("record 0 has out-of-range class id 18", report.errors[0])

    def test_validate_occupancy_labels_cli_json_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "target.json").write_text(json.dumps([[0, 1]]), encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps({"token": "sample-1", "occupancy_path": "target.json"}) + "\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_occupancy_labels.py",
                    "--manifest",
                    str(manifest),
                    "--num-classes",
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
        self.assertEqual(payload["class_histogram"], {"0": 1, "1": 1})


if __name__ == "__main__":
    unittest.main()
