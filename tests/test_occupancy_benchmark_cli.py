import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OccupancyBenchmarkCliTest(unittest.TestCase):
    def test_npz_array_selection_accepts_occ3d_semantics_key(self) -> None:
        from vggt_project.occupancy_benchmark import _select_npz_array

        selected = _select_npz_array(
            {
                "mask_lidar": [[1, 1]],
                "semantics": [[0, 17]],
            },
            Path("labels.npz"),
        )

        self.assertEqual(selected, [[0, 17]])

    def test_semantic_occupancy_benchmark_reports_class_iou_and_miou(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pred.json").write_text(json.dumps([[0, 1], [2, 2]]), encoding="utf-8")
            (root / "target.json").write_text(json.dumps([[0, 1], [1, 2]]), encoding="utf-8")
            manifest = root / "pairs.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "predicted_occupancy_path": "pred.json",
                        "occupancy_path": "target.json",
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
                    "scripts/evaluate_occupancy_benchmark.py",
                    "--manifest",
                    str(manifest),
                    "--num-classes",
                    "3",
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["sample_count"], 1)
        self.assertAlmostEqual(payload["class_iou"]["0"], 1.0)
        self.assertAlmostEqual(payload["class_iou"]["1"], 0.5)
        self.assertAlmostEqual(payload["class_iou"]["2"], 0.5)
        self.assertAlmostEqual(payload["occupancy_miou"], 2.0 / 3.0)
        self.assertIn("sample-1", payload["sample_tokens"])

    def test_occupancy_benchmark_cli_writes_json_report_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pred.json").write_text(json.dumps([[0, 1]]), encoding="utf-8")
            (root / "target.json").write_text(json.dumps([[0, 1]]), encoding="utf-8")
            manifest = root / "pairs.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "predicted_occupancy_path": "pred.json",
                        "occupancy_path": "target.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "reports/occupancy.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_occupancy_benchmark.py",
                    "--manifest",
                    str(manifest),
                    "--json",
                    "--output",
                    str(output),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["sample_count"], 1)
        self.assertAlmostEqual(payload["occupancy_miou"], 1.0)


if __name__ == "__main__":
    unittest.main()
