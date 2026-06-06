import tempfile
import unittest
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


class ModelAdapterContractTest(unittest.TestCase):
    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter contract tests")
    def test_scaffold_model_satisfies_camera_aware_prediction_contract(self) -> None:
        from vggt_project.models.adapter_contract import probe_model_adapter_contract
        from vggt_project.models.factory import ModelBuildConfig

        report = probe_model_adapter_contract(ModelBuildConfig(family="scaffold", point_count=4))

        self.assertTrue(report.contract_ready)
        self.assertFalse(report.template_adapter)
        self.assertEqual(report.errors, ())
        self.assertIn("camera_local_camera_to_gravity_poses", report.prediction_keys)

    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter contract tests")
    def test_repo_g3t_vggt_adapter_reports_template_status(self) -> None:
        from vggt_project.models.adapter_contract import probe_model_adapter_contract
        from vggt_project.models.factory import ModelBuildConfig

        report = probe_model_adapter_contract(
            ModelBuildConfig(
                family="g3t-vggt",
                adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
                point_count=4,
            )
        )

        self.assertTrue(report.contract_ready)
        self.assertTrue(report.template_adapter)
        self.assertEqual(report.adapter_status["status"], "template")

    @unittest.skipUnless(find_spec("torch"), "torch is required for adapter contract tests")
    def test_contract_probe_reports_missing_camera_pose_output(self) -> None:
        from vggt_project.models.adapter_contract import probe_model_adapter_contract
        from vggt_project.models.factory import ModelBuildConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_path = Path(temp_dir) / "bad_adapter.py"
            adapter_path.write_text(
                "import torch\n"
                "from torch import nn\n"
                "class BadAdapter(nn.Module):\n"
                "    def forward(self, batch):\n"
                "        b = batch['bev_features'].shape[0]\n"
                "        h, w = batch['bev_features'].shape[-2:]\n"
                "        return {\n"
                "            'gravity_aligned_pointmap': torch.zeros(b, 4, 3),\n"
                "            'depth': torch.zeros(b, 1, h, w),\n"
                "            'local_camera_to_gravity_pose': torch.zeros(b, 4),\n"
                "            'relative_yaw_translation': torch.zeros(b, 4),\n"
                "        }\n"
                "def build_model(**kwargs):\n"
                "    return BadAdapter()\n",
                encoding="utf-8",
            )

            report = probe_model_adapter_contract(
                ModelBuildConfig(family="external", adapter_module_path=adapter_path, point_count=4)
            )

        self.assertFalse(report.contract_ready)
        self.assertIn("camera_local_camera_to_gravity_poses", " ".join(report.errors))

    @unittest.skipUnless(
        find_spec("torch") and find_spec("yaml"),
        "torch and PyYAML are required for adapter CLI tests",
    )
    def test_check_model_adapter_cli_passes_reference_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter_path = root / "fake_adapter.py"
            adapter_path.write_text(
                "import torch\n"
                "from torch import nn\n"
                "class FakeAdapter(nn.Module):\n"
                "    def __init__(self, status):\n"
                "        super().__init__()\n"
                "        self._status = status\n"
                "    @property\n"
                "    def adapter_status(self):\n"
                "        return {'status': self._status}\n"
                "    def forward(self, batch):\n"
                "        b = batch['bev_features'].shape[0]\n"
                "        c = batch['camera_images'].shape[1]\n"
                "        h, w = batch['bev_features'].shape[-2:]\n"
                "        return {\n"
                "            'gravity_aligned_pointmap': torch.zeros(b, 4, 3),\n"
                "            'depth': torch.zeros(b, 1, h, w),\n"
                "            'camera_depths': torch.zeros(b, c, 1, h, w),\n"
                "            'camera_pointmaps': torch.zeros(b, c, 4, 3),\n"
                "            'local_camera_to_gravity_pose': torch.zeros(b, 4),\n"
                "            'camera_local_camera_to_gravity_poses': torch.zeros(b, c, 4),\n"
                "            'relative_yaw_translation': torch.zeros(b, 4),\n"
                "        }\n"
                "def build_model(**kwargs):\n"
                "    status = 'reference' if kwargs.get('use_reference_adapter') else 'template'\n"
                "    return FakeAdapter(status)\n",
                encoding="utf-8",
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "runtime:\n"
                "  data:\n"
                "    point_count: 4\n"
                "  model:\n"
                "    family: g3t-vggt\n"
                f"    adapter_module_path: {adapter_path}\n"
                "    use_reference_adapter: true\n"
                f"    reference_root: {root}\n"
                "    reference_model: g3t\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "scripts/check_model_adapter.py", "--config", str(config_path)],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("status: reference", result.stdout)


if __name__ == "__main__":
    unittest.main()
