import unittest
from importlib.util import find_spec


class ModelScaffoldTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for model scaffold tests")
    def test_model_predicts_camera_specific_depths_when_camera_images_exist(self) -> None:
        import torch

        from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

        model = SatelliteBEVG3TScaffold.build(point_count=4)
        batch = {
            "bev_features": torch.zeros(2, 8, 16, 16),
            "satellite_patch": torch.zeros(2, 3, 16, 16),
            "camera_images": torch.zeros(2, 3, 3, 16, 16),
        }

        prediction = model(batch)

        self.assertEqual(tuple(prediction["depth"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(prediction["camera_depths"].shape), (2, 3, 1, 16, 16))
        self.assertEqual(tuple(prediction["camera_pointmaps"].shape), (2, 3, 4, 3))
        self.assertEqual(tuple(prediction["camera_local_camera_to_gravity_poses"].shape), (2, 3, 4))

    @unittest.skipUnless(find_spec("torch"), "torch is required for model scaffold tests")
    def test_model_keeps_camera_depths_absent_for_synthetic_batches(self) -> None:
        import torch

        from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

        model = SatelliteBEVG3TScaffold.build(point_count=4)
        batch = {
            "bev_features": torch.zeros(2, 8, 16, 16),
            "satellite_patch": torch.zeros(2, 3, 16, 16),
        }

        prediction = model(batch)

        self.assertIsNone(prediction["camera_depths"])
        self.assertIsNone(prediction["camera_pointmaps"])
        self.assertIsNone(prediction["camera_local_camera_to_gravity_poses"])


if __name__ == "__main__":
    unittest.main()
