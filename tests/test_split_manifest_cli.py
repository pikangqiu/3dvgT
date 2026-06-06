import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class SplitManifestCliTest(unittest.TestCase):
    def test_empty_split_error_is_reported_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "eval.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-a","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/split_manifest.py",
                    str(manifest),
                    "--train-output",
                    str(train_manifest),
                    "--eval-output",
                    str(eval_manifest),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error: eval split is empty", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(train_manifest.exists())
        self.assertFalse(eval_manifest.exists())


if __name__ == "__main__":
    unittest.main()
