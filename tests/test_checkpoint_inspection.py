import tempfile
import subprocess
import sys
import unittest
from pathlib import Path


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        self.shape = shape
        self.dtype = dtype


class CheckpointInspectionTest(unittest.TestCase):
    def test_summary_extracts_nested_state_dict_and_prefix_counts(self) -> None:
        from vggt_project.checkpoint_inspection import summarize_checkpoint

        summary = summarize_checkpoint(
            {
                "state_dict": {
                    "reference_model.encoder.weight": _FakeTensor((2, 3)),
                    "reference_model.depth_head.bias": _FakeTensor((2,)),
                    "satellite_adapter.weight": _FakeTensor((4, 4), dtype="float16"),
                }
            }
        )

        self.assertEqual(summary.container_key, "state_dict")
        self.assertEqual(summary.tensor_count, 3)
        self.assertEqual(summary.prefix_counts["reference_model"], 2)
        self.assertEqual(summary.prefix_counts["satellite_adapter"], 1)
        self.assertEqual(summary.tensor_shapes["reference_model.encoder.weight"], (2, 3))
        self.assertEqual(summary.tensor_dtypes["satellite_adapter.weight"], "float16")

    def test_summary_accepts_raw_state_dict(self) -> None:
        from vggt_project.checkpoint_inspection import summarize_checkpoint

        summary = summarize_checkpoint(
            {
                "module.backbone.point_head.weight": _FakeTensor((12, 8)),
                "module.backbone.depth_head.bias": _FakeTensor((32,)),
            }
        )

        self.assertEqual(summary.container_key, "raw")
        self.assertEqual(summary.prefix_counts["module"], 2)
        self.assertEqual(summary.sample_keys, ("module.backbone.depth_head.bias", "module.backbone.point_head.weight"))

    def test_find_checkpoint_candidates_recurses_weight_files(self) -> None:
        from vggt_project.checkpoint_inspection import find_checkpoint_candidates

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model.pt").write_text("pt", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested/model.bin").write_text("bin", encoding="utf-8")
            (root / "nested/adapter.pth").write_text("pth", encoding="utf-8")
            (root / "config.json").write_text("{}", encoding="utf-8")

            candidates = find_checkpoint_candidates(root)

        self.assertEqual(
            tuple(path.name for path in candidates),
            ("model.pt", "adapter.pth", "model.bin"),
        )

    def test_cli_lists_directory_candidates_without_loading_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model.pt").write_text("pt", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "scripts/inspect_checkpoint.py", str(root)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("checkpoint_candidates: 1", result.stdout)
        self.assertIn("model.pt", result.stdout)


if __name__ == "__main__":
    unittest.main()
