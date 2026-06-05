import unittest
from importlib.util import find_spec


class ReconstructionLossesTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for loss tensor tests")
    def test_depth_loss_prefers_multi_camera_depth_targets(self) -> None:
        import torch

        from vggt_project.losses import reconstruction_losses

        prediction = {
            "gravity_aligned_pointmap": torch.zeros(1, 2, 3),
            "depth": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 2, 3),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_camera_depths": torch.tensor(
                [[[[[2.0, 2.0], [2.0, 2.0]]], [[[4.0, 4.0], [4.0, 4.0]]]]]
            ),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        losses = reconstruction_losses(prediction, batch)

        self.assertAlmostEqual(float(losses["depth"]), 2.0)


if __name__ == "__main__":
    unittest.main()
