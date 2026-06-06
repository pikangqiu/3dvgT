import tempfile
import unittest
from pathlib import Path

from vggt_project.experiments import ExperimentRunConfig
from vggt_project.training_readiness import (
    DependencyStatus,
    check_training_readiness,
)


class TrainingReadinessTest(unittest.TestCase):
    def test_readiness_reports_missing_split_manifests(self) -> None:
        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            train_manifest_path=Path("missing-train.jsonl"),
            eval_manifest_path=Path("missing-val.jsonl"),
            satellite_raster_config_path=Path("missing-satellite-config.json"),
            device="cpu",
        )

        report = check_training_readiness(
            config,
            dependency_probe=lambda: (
                DependencyStatus("torch", True, "2.0"),
                DependencyStatus("PIL", True, "10.0"),
            ),
            device_probe=lambda device: True,
        )

        self.assertFalse(report.ready)
        self.assertIn("train_manifest_path", report.missing_paths)
        self.assertIn("eval_manifest_path", report.missing_paths)
        self.assertIn("satellite_raster_config_path", report.missing_paths)
        self.assertEqual(report.device, "cpu")

    def test_readiness_passes_when_manifests_dependencies_and_device_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            train_manifest.write_text("", encoding="utf-8")
            eval_manifest.write_text("", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                satellite_raster_config_path=None,
                device="cpu",
            )

            report = check_training_readiness(
                config,
                dependency_probe=lambda: (
                    DependencyStatus("torch", True, "2.0"),
                    DependencyStatus("PIL", True, "10.0"),
                    DependencyStatus("yaml", True, "6.0"),
                ),
                device_probe=lambda device: True,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.missing_paths, {})
        self.assertEqual(report.config_errors, ())
        self.assertEqual(report.missing_dependencies, ())

    def test_readiness_reports_satellite_raster_config_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "train.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"map_location":"boston-seaport",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            satellite_config = root / "satellite_config.json"
            satellite_config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/missing.png",'
                '"origin_ego_xy_m":[0.0,0.0],'
                '"origin_pixel_xy":[0.0,0.0],'
                '"meters_per_pixel":1.0'
                "}}\n",
                encoding="utf-8",
            )
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=manifest,
                satellite_raster_config_path=satellite_config,
                device="cpu",
            )

            report = check_training_readiness(
                config,
                dependency_probe=lambda: (
                    DependencyStatus("torch", True, "2.0"),
                    DependencyStatus("PIL", True, "10.0"),
                    DependencyStatus("numpy", True, "1.0"),
                    DependencyStatus("yaml", True, "6.0"),
                ),
                device_probe=lambda device: True,
            )

        self.assertFalse(report.ready)
        self.assertIn("satellite raster missing", report.config_errors[0])

    def test_readiness_accepts_ready_satellite_raster_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "rasters").mkdir()
            (root / "rasters/boston.png").write_text("raster", encoding="utf-8")
            manifest = root / "train.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"map_location":"boston-seaport",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            satellite_config = root / "satellite_config.json"
            satellite_config.write_text(
                '{"boston-seaport":{'
                '"raster_path":"rasters/boston.png",'
                '"origin_ego_xy_m":[0.0,0.0],'
                '"origin_pixel_xy":[0.0,0.0],'
                '"meters_per_pixel":1.0'
                "}}\n",
                encoding="utf-8",
            )
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=manifest,
                satellite_raster_config_path=satellite_config,
                device="cpu",
            )

            report = check_training_readiness(
                config,
                dependency_probe=lambda: (
                    DependencyStatus("torch", True, "2.0"),
                    DependencyStatus("PIL", True, "10.0"),
                    DependencyStatus("numpy", True, "1.0"),
                    DependencyStatus("yaml", True, "6.0"),
                ),
                device_probe=lambda device: True,
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.config_errors, ())

    def test_readiness_reports_external_adapter_without_module_path(self) -> None:
        config = ExperimentRunConfig(
            training_mode="synthetic",
            model_family="external",
            device="cpu",
        )

        report = check_training_readiness(
            config,
            dependency_probe=lambda: (
                DependencyStatus("torch", True, "2.0"),
                DependencyStatus("PIL", True, "10.0"),
                DependencyStatus("numpy", True, "1.0"),
                DependencyStatus("yaml", True, "6.0"),
            ),
            device_probe=lambda device: True,
        )

        self.assertFalse(report.ready)
        self.assertIn("adapter_module_path", report.config_errors[0])

    def test_readiness_reports_missing_external_adapter_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter.py"
            adapter.write_text("def build_model(**kwargs):\n    return None\n", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="synthetic",
                model_family="external",
                adapter_module_path=adapter,
                weights_path=root / "missing.pt",
                device="cpu",
            )

            report = check_training_readiness(
                config,
                dependency_probe=lambda: (
                    DependencyStatus("torch", True, "2.0"),
                    DependencyStatus("PIL", True, "10.0"),
                    DependencyStatus("numpy", True, "1.0"),
                    DependencyStatus("yaml", True, "6.0"),
                ),
                device_probe=lambda device: True,
            )

        self.assertFalse(report.ready)
        self.assertIn("weights_path", report.missing_paths)

    def test_readiness_reports_missing_reference_root(self) -> None:
        config = ExperimentRunConfig(
            training_mode="synthetic",
            model_family="g3t-vggt",
            adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
            use_reference_adapter=True,
            reference_root=Path("missing-reference-root"),
            device="cpu",
        )

        report = check_training_readiness(
            config,
            dependency_probe=lambda: (
                DependencyStatus("torch", True, "2.0"),
                DependencyStatus("PIL", True, "10.0"),
                DependencyStatus("numpy", True, "1.0"),
                DependencyStatus("yaml", True, "6.0"),
            ),
            device_probe=lambda device: True,
        )

        self.assertFalse(report.ready)
        self.assertIn("reference_root", report.missing_paths)

    def test_readiness_requires_reference_root_when_reference_adapter_is_enabled(self) -> None:
        config = ExperimentRunConfig(
            training_mode="synthetic",
            model_family="g3t-vggt",
            adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
            use_reference_adapter=True,
            reference_root=None,
            device="cpu",
        )

        report = check_training_readiness(
            config,
            dependency_probe=lambda: (
                DependencyStatus("torch", True, "2.0"),
                DependencyStatus("PIL", True, "10.0"),
                DependencyStatus("numpy", True, "1.0"),
                DependencyStatus("yaml", True, "6.0"),
            ),
            device_probe=lambda device: True,
        )

        self.assertFalse(report.ready)
        self.assertIn("reference_root", report.config_errors[0])


if __name__ == "__main__":
    unittest.main()
