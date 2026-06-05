import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class SmokePipelineTest(unittest.TestCase):
    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch"),
        "Pillow and torch are required for smoke pipeline tests",
    )
    def test_manifest_smoke_pipeline_trains_and_evaluates(self) -> None:
        from vggt_project.smoke_pipeline import run_manifest_smoke_pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_manifest_smoke_pipeline(
                output_dir=Path(temp_dir),
                epochs=1,
                image_size=16,
                point_count=4,
            )

            self.assertTrue(report.checkpoint_path.exists())
            self.assertTrue(report.manifest_path.exists())
            self.assertIn("loss", report.train_metrics)
            self.assertIn("loss", report.eval_metrics)
            self.assertIn("depth_mae", report.eval_metrics)


if __name__ == "__main__":
    unittest.main()
