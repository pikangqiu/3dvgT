import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeNamedParameterModel:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.parameters_by_name = {name: _FakeParameter() for name in names}

    def named_parameters(self):
        return tuple(self.parameters_by_name.items())

    def parameters(self):
        return tuple(self.parameters_by_name.values())


class ModelFactoryTest(unittest.TestCase):
    def test_fine_tuning_policy_keeps_satellite_fusion_and_heads_trainable(self) -> None:
        from vggt_project.models.factory import apply_fine_tuning_policy

        model = _FakeNamedParameterModel(
            (
                "bev_encoder.0.weight",
                "satellite_encoder.0.weight",
                "fusion.0.weight",
                "point_head.weight",
                "camera_depth_head.weight",
                "reference_model.encoder.weight",
            )
        )

        apply_fine_tuning_policy(model, "satellite_fusion_heads")

        self.assertFalse(model.parameters_by_name["bev_encoder.0.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["satellite_encoder.0.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["fusion.0.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["point_head.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["camera_depth_head.weight"].requires_grad)
        self.assertFalse(model.parameters_by_name["reference_model.encoder.weight"].requires_grad)

    def test_fine_tuning_policy_can_freeze_reference_backbone_only(self) -> None:
        from vggt_project.models.factory import apply_fine_tuning_policy

        model = _FakeNamedParameterModel(
            (
                "reference_model.encoder.weight",
                "reference_model.depth_head.weight",
                "satellite_adapter.weight",
                "pose_head.weight",
            )
        )

        apply_fine_tuning_policy(model, "reference_frozen_heads")

        self.assertFalse(model.parameters_by_name["reference_model.encoder.weight"].requires_grad)
        self.assertFalse(model.parameters_by_name["reference_model.depth_head.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["satellite_adapter.weight"].requires_grad)
        self.assertTrue(model.parameters_by_name["pose_head.weight"].requires_grad)

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

    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_external_adapter_receives_reference_adapter_options(self) -> None:
        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "fake_adapter.py"
            adapter_path.write_text(
                "from torch import nn\n"
                "class FakeAdapter(nn.Module):\n"
                "    def __init__(self, kwargs):\n"
                "        super().__init__()\n"
                "        self.kwargs = kwargs\n"
                "def build_model(**kwargs):\n"
                "    return FakeAdapter(kwargs)\n",
                encoding="utf-8",
            )

            model = build_reconstruction_model(
                ModelBuildConfig(
                    family="external",
                    adapter_module_path=adapter_path,
                    use_reference_adapter=True,
                    reference_root=Path("refs/g3t"),
                    reference_model="g3t",
                    reference_model_kwargs={"img_size": 518},
                    point_count=4,
                )
            )

        self.assertTrue(model.kwargs["use_reference_adapter"])
        self.assertEqual(model.kwargs["reference_root"], Path("refs/g3t"))
        self.assertEqual(model.kwargs["reference_model"], "g3t")
        self.assertEqual(model.kwargs["reference_model_kwargs"], {"img_size": 518})

    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_external_adapter_can_handle_project_weight_loading(self) -> None:
        import torch

        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_path = root / "fake_adapter.py"
            weights_path = root / "weights.pt"
            adapter_path.write_text(
                "import torch\n"
                "from torch import nn\n"
                "class FakeAdapter(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.loaded = None\n"
                "        self.probe = nn.Parameter(torch.tensor(0.0))\n"
                "    def load_project_weights(self, weights_path, strict=True):\n"
                "        self.loaded = (str(weights_path), strict)\n"
                "        self.probe.data.fill_(torch.load(weights_path)['probe'])\n"
                "def build_model(**kwargs):\n"
                "    return FakeAdapter()\n",
                encoding="utf-8",
            )
            torch.save({"probe": 3.0}, weights_path)

            model = build_reconstruction_model(
                ModelBuildConfig(
                    family="external",
                    adapter_module_path=adapter_path,
                    weights_path=weights_path,
                    strict_weights=False,
                )
            )

        self.assertEqual(model.loaded, (str(weights_path), False))
        self.assertAlmostEqual(float(model.probe.detach()), 3.0)

    @unittest.skipUnless(find_spec("torch"), "torch is required for model factory tests")
    def test_repo_g3t_vggt_adapter_template_builds_with_factory(self) -> None:
        import torch

        from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

        model = build_reconstruction_model(
            ModelBuildConfig(
                family="g3t-vggt",
                adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
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

    def test_prediction_contract_reports_missing_keys(self) -> None:
        from vggt_project.models.factory import validate_reconstruction_prediction

        with self.assertRaisesRegex(ValueError, "depth"):
            validate_reconstruction_prediction(
                {
                    "gravity_aligned_pointmap": object(),
                    "local_camera_to_gravity_pose": object(),
                    "relative_yaw_translation": object(),
                }
            )


if __name__ == "__main__":
    unittest.main()
