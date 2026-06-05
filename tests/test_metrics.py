import unittest
from importlib.util import find_spec


class ReconstructionMetricsTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_depth_and_pointmap_metrics_are_reported_as_floats(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            "depth": torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]]),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        }
        batch = {
            "target_pointmap": torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]),
            "target_depth": torch.tensor([[[[2.0, 2.0], [6.0, 6.0]]]]),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.tensor([[0.0, 1.0, 2.0, 4.0]]),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["depth_mae"], 1.0)
        self.assertAlmostEqual(metrics["pointmap_l1"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["local_pose_l2"], 0.0)
        self.assertAlmostEqual(metrics["relative_pose_l2"], 1.0)


if __name__ == "__main__":
    unittest.main()
