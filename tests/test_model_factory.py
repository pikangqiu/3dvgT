import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class ModelFactoryTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_external_adapter_module_builds_reconstruction_model(self) -> None:
        import torch

        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "fake_adapter.py"
            adapter_path.write_text(
                "import torch\n"
                "from torch import nn\n"
                "\n"
                "class FakeAdapter(nn.Module):\n"
                "    def __init__(self, point_count):\n"
                "        super().__init__()\n"
                "        self.point_count = point_count\n"
                "        self.scale = nn.Parameter(torch.tensor(1.0))\n"
                "\n"
                "    def forward(self, batch):\n"
                "        batch_size = batch['bev_features'].shape[0]\n"
                "        height, width = batch['bev_features'].shape[-2:]\n"
                "        camera_count = batch.get('camera_images', torch.zeros(batch_size, 0)).shape[1]\n"
                "        return {\n"
                "            'gravity_aligned_pointmap': self.scale * torch.zeros(batch_size, self.point_count, 3),\n"
                "            'depth': torch.zeros(batch_size, 1, height, width),\n"
                "            'camera_depths': torch.zeros(batch_size, camera_count, 1, height, width),\n"
                "            'camera_pointmaps': torch.zeros(batch_size, camera_count, self.point_count, 3),\n"
                "            'local_camera_to_gravity_pose': torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1),\n"
                "            'relative_yaw_translation': torch.zeros(batch_size, 4),\n"
                "        }\n"
                "\n"
                "def build_model(point_count, **kwargs):\n"
                "    return FakeAdapter(point_count)\n",
                encoding="utf-8",
            )

            model = build_reconstruction_model(
                ModelBuildConfig(
                    family="external",
                    adapter_module_path=adapter_path,
                    point_count=4,
                )
            )

        prediction = model(
            {
                "bev_features": torch.zeros(2, 8, 16, 16),
                "satellite_patch": torch.zeros(2, 3, 16, 16),
                "camera_images": torch.zeros(2, 3, 3, 16, 16),
            }
        )

        self.assertEqual(tuple(prediction["gravity_aligned_pointmap"].shape), (2, 4, 3))
        self.assertEqual(tuple(prediction["camera_pointmaps"].shape), (2, 3, 4, 3))

    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_external_adapter_requires_module_path(self) -> None:
        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        with self.assertRaisesRegex(ValueError, "adapter_module_path"):
            build_reconstruction_model(ModelBuildConfig(family="external"))

    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_external_adapter_uses_custom_freeze_backbone_hook(self) -> None:
        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "fake_adapter.py"
            adapter_path.write_text(
                "from torch import nn\n"
                "\n"
                "class FakeAdapter(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.backbone = nn.Linear(1, 1)\n"
                "        self.head = nn.Linear(1, 1)\n"
                "\n"
                "    def freeze_backbone(self):\n"
                "        for parameter in self.backbone.parameters():\n"
                "            parameter.requires_grad = False\n"
                "\n"
                "def build_model(**kwargs):\n"
                "    return FakeAdapter()\n",
                encoding="utf-8",
            )

            model = build_reconstruction_model(
                ModelBuildConfig(
                    family="external",
                    adapter_module_path=adapter_path,
                    freeze_backbone=True,
                )
            )

        self.assertFalse(model.backbone.weight.requires_grad)
        self.assertTrue(model.head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
