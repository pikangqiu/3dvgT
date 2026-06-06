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
        self.assertIn("scale_aligned_pointmap_accuracy", metrics)
        self.assertIn("scale_aligned_pointmap_completeness", metrics)
        self.assertIn("scale_aligned_pointmap_chamfer", metrics)
        self.assertIn("gravity_error_deg", metrics)
        self.assertIn("sequence_translation_drift", metrics)
        self.assertAlmostEqual(metrics["local_pose_l2"], 0.0)
        self.assertAlmostEqual(metrics["relative_pose_l2"], 1.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_depth_metric_prefers_multi_camera_depth_targets(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.zeros(1, 1, 3),
            "depth": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]]),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 1, 3),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_camera_depths": torch.tensor(
                [[[[[2.0, 2.0], [2.0, 2.0]]], [[[4.0, 4.0], [4.0, 4.0]]]]]
            ),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["depth_mae"], 2.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_occupancy_iou_is_reported_when_prediction_and_target_exist(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.zeros(1, 1, 3),
            "depth": torch.zeros(1, 1, 2, 2),
            "bev_occupancy": torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]]),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 1, 3),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_occupancy": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["bev_occupancy_iou"], 0.5)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_pointmap_metric_prefers_camera_specific_predictions(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.zeros(1, 2, 3),
            "camera_pointmaps": torch.tensor(
                [[[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
            "depth": torch.zeros(1, 1, 2, 2),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 2, 3),
            "target_camera_pointmaps": torch.tensor(
                [[[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["pointmap_l1"], 1.0 / 6.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_pointmap_metric_expands_global_prediction_for_camera_targets(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            "depth": torch.zeros(1, 1, 2, 2),
            "local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 2, 3),
            "target_camera_pointmaps": torch.tensor(
                [[[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]
            ),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["pointmap_l1"], 1.0 / 3.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_scale_aligned_pointmap_metrics_ignore_global_scale(self) -> None:
        import torch

        from vggt_project.metrics import scale_aligned_pointmap_metrics

        predicted = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

        metrics = scale_aligned_pointmap_metrics(predicted, target)

        self.assertAlmostEqual(float(metrics["accuracy"]), 0.0)
        self.assertAlmostEqual(float(metrics["completeness"]), 0.0)
        self.assertAlmostEqual(float(metrics["chamfer"]), 0.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_gravity_error_reports_quaternion_angle_degrees(self) -> None:
        import math
        import torch

        from vggt_project.metrics import quaternion_angular_error_deg

        predicted = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]])
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        self.assertAlmostEqual(float(quaternion_angular_error_deg(predicted, target)), 90.0, places=4)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_gravity_metric_prefers_camera_specific_pose_predictions(self) -> None:
        import torch

        from vggt_project.metrics import reconstruction_metrics

        prediction = {
            "gravity_aligned_pointmap": torch.zeros(1, 1, 3),
            "depth": torch.zeros(1, 1, 2, 2),
            "local_camera_to_gravity_pose": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            "camera_local_camera_to_gravity_poses": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
            ),
            "relative_yaw_translation": torch.zeros(1, 4),
        }
        batch = {
            "target_pointmap": torch.zeros(1, 1, 3),
            "target_depth": torch.zeros(1, 1, 2, 2),
            "target_local_camera_to_gravity_pose": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "target_camera_local_camera_to_gravity_poses": torch.tensor(
                [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
            ),
            "target_relative_yaw_translation": torch.zeros(1, 4),
            "valid_area_mask": torch.ones(1, 1, 2, 2),
        }

        metrics = reconstruction_metrics(prediction, batch)

        self.assertAlmostEqual(metrics["gravity_error_deg"], 0.0)
        self.assertAlmostEqual(metrics["local_pose_l2"], 0.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for metric tensor tests")
    def test_sequence_translation_drift_uses_batch_relative_track(self) -> None:
        import torch

        from vggt_project.metrics import sequence_translation_drift

        predicted = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        self.assertAlmostEqual(float(sequence_translation_drift(predicted, target)), 0.5)


if __name__ == "__main__":
    unittest.main()
