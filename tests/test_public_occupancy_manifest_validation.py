import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PublicOccupancyManifestValidationTest(unittest.TestCase):
    def test_valid_public_occupancy_manifest_reports_split_alignment(self) -> None:
        from vggt_project.public_occupancy_manifest import validate_public_occupancy_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = (
                root
                / "data/occ3d/occ3d-nuscenes/trainval/gts/scene-0001/sample-1/labels.npz"
            )
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"placeholder")
            manifest = root / "manifests/eval.occ3d.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "scene_name": "scene-0001",
                        "occupancy_path": "../data/occ3d/occ3d-nuscenes/trainval/gts/scene-0001/sample-1/labels.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_public_occupancy_manifest(
                manifest,
                expected_split="trainval",
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.scene_count, 1)
        self.assertEqual(report.label_root_matches, 1)
        self.assertEqual(report.errors, ())

    def test_public_occupancy_manifest_rejects_labels_outside_expected_split(self) -> None:
        from vggt_project.public_occupancy_manifest import validate_public_occupancy_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = root / "data/occ3d/occ3d-nuscenes/gts/scene-0001/sample-1/labels.npz"
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"placeholder")
            manifest = root / "eval.occ3d.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "scene_name": "scene-0001",
                        "occupancy_path": "data/occ3d/occ3d-nuscenes/gts/scene-0001/sample-1/labels.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_public_occupancy_manifest(
                manifest,
                expected_split="trainval",
            )

        self.assertFalse(report.ready)
        self.assertIn("record 0 occupancy_path is not under expected public split: trainval", report.errors)

    def test_validate_public_occupancy_manifest_cli_json_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = root / "occ3d-nuscenes/trainval/gts/scene-0002/sample-2/labels.npz"
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"placeholder")
            manifest = root / "eval.occ3d.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-2",
                        "scene_name": "scene-0002",
                        "occupancy_path": "occ3d-nuscenes/trainval/gts/scene-0002/sample-2/labels.npz",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/validate_public_occupancy_manifest.py",
                    "--manifest",
                    str(manifest),
                    "--expected-split",
                    "trainval",
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
        self.assertEqual(payload["label_root_matches"], 1)


if __name__ == "__main__":
    unittest.main()
