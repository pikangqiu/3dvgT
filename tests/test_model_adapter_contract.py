import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
