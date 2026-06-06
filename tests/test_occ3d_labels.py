import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Occ3DLabelsTest(unittest.TestCase):
    def test_attach_occ3d_labels_writes_manifest_relative_public_label_paths(self) -> None:
        from vggt_project.data.occ3d_labels import attach_occ3d_label_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = (
                root
                / "data/occ3d/occ3d-nuscenes/trainval/gts/scene-0001/sample-1/labels.npz"
            )
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"fake npz placeholder")
            manifest = root / "input/samples.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "scene_name": "scene-0001",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "manifests/samples.occ3d.jsonl"

            report = attach_occ3d_label_manifest(
                manifest,
                occ3d_root=root / "data/occ3d",
                output_manifest_path=output,
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            resolved_label = output.parent / record["occupancy_path"]
            label_exists = resolved_label.exists()

        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.labels_attached, 1)
        self.assertEqual(record["occupancy_path"], "../data/occ3d/occ3d-nuscenes/trainval/gts/scene-0001/sample-1/labels.npz")
        self.assertTrue(label_exists)

    def test_attach_occ3d_labels_can_resolve_scene_name_from_scene_token(self) -> None:
        from vggt_project.data.occ3d_labels import attach_occ3d_label_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = root / "occ3d-nuscenes/gts/scene-0002/sample-2/labels.npz"
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"fake npz placeholder")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-2",
                        "scene_token": "scene-token-2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "samples.occ3d.jsonl"

            report = attach_occ3d_label_manifest(
                manifest,
                occ3d_root=root,
                output_manifest_path=output,
                scene_name_resolver=lambda scene_token: {"scene-token-2": "scene-0002"}[scene_token],
            )

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report.labels_attached, 1)
        self.assertEqual(record["scene_name"], "scene-0002")
        self.assertEqual(record["occupancy_path"], "occ3d-nuscenes/gts/scene-0002/sample-2/labels.npz")

    def test_attach_occ3d_labels_cli_writes_output_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            label_path = root / "occ3d-nuscenes/trainval/gts/scene-0003/sample-3/labels.npz"
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(b"fake npz placeholder")
            manifest = root / "input.jsonl"
            manifest.write_text(
                json.dumps({"token": "sample-3", "scene_name": "scene-0003"}) + "\n",
                encoding="utf-8",
            )
            output = root / "output.jsonl"
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/attach_occ3d_labels.py",
                    "--manifest",
                    str(manifest),
                    "--occ3d-root",
                    str(root),
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("labels_attached: 1", result.stdout)
        self.assertEqual(record["occupancy_path"], "occ3d-nuscenes/trainval/gts/scene-0003/sample-3/labels.npz")


if __name__ == "__main__":
    unittest.main()
