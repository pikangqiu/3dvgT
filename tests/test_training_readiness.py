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
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
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

    def test_readiness_rejects_train_manifest_with_missing_referenced_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            train_manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/missing.jpg"],'
                '"satellite_patch_path":"sat/missing.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            eval_manifest.write_text("", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
                satellite_raster_config_path=None,
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
        self.assertIn("train_manifest_path references missing files", report.config_errors[0])
        self.assertIn("camera.image_path", report.config_errors[0])

    def test_readiness_rejects_eval_manifest_with_missing_referenced_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            train_manifest.write_text("", encoding="utf-8")
            eval_manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/missing.jpg"],'
                '"satellite_patch_path":"sat/missing.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
                satellite_raster_config_path=None,
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
        self.assertIn("eval_manifest_path references missing files", report.config_errors[0])
        self.assertIn("satellite_patch_path", report.config_errors[0])

    def test_readiness_reports_invalid_manifest_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            train_manifest.write_text("{not-json}\n", encoding="utf-8")
            eval_manifest.write_text("", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
                satellite_raster_config_path=None,
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
        self.assertIn("train_manifest_path is invalid", report.config_errors[0])

    def test_readiness_reports_satellite_raster_config_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/sample-1.png").write_text("sat", encoding="utf-8")
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
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
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
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "rasters/boston.png").write_text("raster", encoding="utf-8")
            (root / "samples/CAM_FRONT/a.jpg").write_text("image", encoding="utf-8")
            (root / "sat/sample-1.png").write_text("sat", encoding="utf-8")
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
                output_dir=root / "outputs",
                checkpoint=root / "outputs" / "manifest_smoke_scaffold.pt",
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

    def test_readiness_rejects_checkpoint_directory_weights_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter.py"
            adapter.write_text("def build_model(**kwargs):\n    return None\n", encoding="utf-8")
            weights_dir = root / "weights"
            weights_dir.mkdir()
            (weights_dir / "model.pt").write_text("placeholder", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="synthetic",
                model_family="external",
                adapter_module_path=adapter,
                weights_path=weights_dir,
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
        self.assertIn("weights_path points to a directory", report.config_errors[0])
        self.assertIn("model.pt", report.config_errors[0])

    def test_readiness_rejects_non_checkpoint_weights_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter.py"
            adapter.write_text("def build_model(**kwargs):\n    return None\n", encoding="utf-8")
            weights_file = root / "config.json"
            weights_file.write_text("{}", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="synthetic",
                model_family="external",
                adapter_module_path=adapter,
                weights_path=weights_file,
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
        self.assertIn("weights_path must be a .pt, .pth, or .bin file", report.config_errors[0])

    def test_readiness_rejects_unloadable_checkpoint_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter.py"
            adapter.write_text("def build_model(**kwargs):\n    return None\n", encoding="utf-8")
            weights_file = root / "model.pt"
            weights_file.write_text("not a torch checkpoint", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="synthetic",
                model_family="external",
                adapter_module_path=adapter,
                weights_path=weights_file,
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
                checkpoint_probe=lambda path: f"could not load checkpoint {path.name}",
            )

        self.assertFalse(report.ready)
        self.assertIn("weights_path checkpoint inspection failed", report.config_errors[0])
        self.assertIn("could not load checkpoint model.pt", report.config_errors[0])

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

    def test_readiness_reports_unknown_fine_tuning_policy(self) -> None:
        config = ExperimentRunConfig(
            training_mode="synthetic",
            fine_tuning_policy="unknown-policy",
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
        self.assertIn("fine_tuning_policy", report.config_errors[0])

    def test_readiness_rejects_output_dir_that_is_a_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            output_file = root / "output-file"
            train_manifest.write_text("", encoding="utf-8")
            eval_manifest.write_text("", encoding="utf-8")
            output_file.write_text("not a directory", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                output_dir=output_file,
                checkpoint=output_file / "manifest_smoke_scaffold.pt",
                satellite_raster_config_path=None,
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
        self.assertIn("output_dir must be a directory", report.config_errors[0])

    def test_readiness_rejects_eval_checkpoint_that_training_will_not_write(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "val.jsonl"
            output_dir = root / "outputs"
            train_manifest.write_text("", encoding="utf-8")
            eval_manifest.write_text("", encoding="utf-8")
            config = ExperimentRunConfig(
                training_mode="manifest-smoke",
                train_manifest_path=train_manifest,
                eval_manifest_path=eval_manifest,
                output_dir=output_dir,
                checkpoint=root / "other.pt",
                satellite_raster_config_path=None,
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
        self.assertIn("evaluation checkpoint must match training output", report.config_errors[0])


if __name__ == "__main__":
    unittest.main()
