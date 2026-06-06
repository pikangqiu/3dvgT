import unittest
from importlib.util import find_spec


class G3TVGGTAdapterMappingTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter mapping tests")
    def test_reference_outputs_are_mapped_to_reconstruction_contract(self) -> None:
        import torch

        from adapters.g3t_vggt_adapter import map_reference_prediction_to_contract

        batch = {
            "bev_features": torch.zeros(2, 8, 4, 4),
            "camera_images": torch.zeros(2, 3, 3, 4, 4),
        }
        reference_prediction = {
            "depth": torch.ones(2, 3, 4, 4, 1),
            "world_points": torch.arange(2 * 3 * 4 * 4 * 3, dtype=torch.float32).reshape(2, 3, 4, 4, 3),
            "local_pose_enc": torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                    [[0.0, 1.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                ]
            ),
            "global_pose_enc": torch.tensor(
                [
                    [[0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                    [[0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                ]
            ),
        }

        prediction = map_reference_prediction_to_contract(reference_prediction, batch, point_count=5)

        self.assertEqual(tuple(prediction["depth"].shape), (2, 1, 4, 4))
        self.assertEqual(tuple(prediction["camera_depths"].shape), (2, 3, 1, 4, 4))
        self.assertEqual(tuple(prediction["gravity_aligned_pointmap"].shape), (2, 5, 3))
        self.assertEqual(tuple(prediction["camera_pointmaps"].shape), (2, 3, 5, 3))
        self.assertEqual(tuple(prediction["local_camera_to_gravity_pose"].shape), (2, 4))
        self.assertEqual(tuple(prediction["camera_local_camera_to_gravity_poses"].shape), (2, 3, 4))
        self.assertEqual(tuple(prediction["relative_yaw_translation"].shape), (2, 4))
        self.assertAlmostEqual(float(prediction["local_camera_to_gravity_pose"][0, 0]), 1.0)
        self.assertAlmostEqual(float(prediction["relative_yaw_translation"][1, 3]), 6.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter mapping tests")
    def test_reference_mapping_falls_back_to_scene_outputs_when_camera_outputs_are_missing(self) -> None:
        import torch

        from adapters.g3t_vggt_adapter import map_reference_prediction_to_contract

        batch = {
            "bev_features": torch.zeros(1, 8, 4, 4),
            "camera_images": torch.zeros(1, 2, 3, 4, 4),
        }
        reference_prediction = {}

        prediction = map_reference_prediction_to_contract(reference_prediction, batch, point_count=4)

        self.assertEqual(tuple(prediction["depth"].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(prediction["camera_depths"].shape), (1, 2, 1, 4, 4))
        self.assertEqual(tuple(prediction["gravity_aligned_pointmap"].shape), (1, 4, 3))
        self.assertEqual(tuple(prediction["camera_pointmaps"].shape), (1, 2, 4, 3))
        self.assertAlmostEqual(float(prediction["local_camera_to_gravity_pose"][0, 0]), 1.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter mapping tests")
    def test_reference_adapter_wraps_g3t_forward_outputs(self) -> None:
        import torch
        from torch import nn

        from adapters.g3t_vggt_adapter import G3TVGGTReferenceAdapter

        class FakeReferenceModel(nn.Module):
            def forward(self, images):
                batch_size, camera_count, _, height, width = images.shape
                return {
                    "depth": torch.ones(batch_size, camera_count, height, width, 1),
                    "world_points": torch.ones(batch_size, camera_count, height, width, 3),
                    "local_pose_enc": torch.ones(batch_size, camera_count, 9),
                    "global_pose_enc": torch.zeros(batch_size, camera_count, 9),
                }

        adapter = G3TVGGTReferenceAdapter(FakeReferenceModel(), point_count=6)
        prediction = adapter(
            {
                "bev_features": torch.zeros(2, 8, 4, 4),
                "satellite_patch": torch.zeros(2, 3, 4, 4),
                "camera_images": torch.zeros(2, 3, 3, 4, 4),
            }
        )

        self.assertEqual(tuple(prediction["camera_depths"].shape), (2, 3, 1, 4, 4))
        self.assertEqual(tuple(prediction["camera_pointmaps"].shape), (2, 3, 6, 3))
