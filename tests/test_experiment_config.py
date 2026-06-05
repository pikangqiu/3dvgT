import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from vggt_project.experiments import (
    ExperimentRunConfig,
    load_experiment_config,
    run_experiment_from_config,
)


class ExperimentConfigTest(unittest.TestCase):
    def test_experiment_config_uses_runtime_defaults(self) -> None:
        config = ExperimentRunConfig.from_mapping({})

        self.assertEqual(config.training_mode, "synthetic")
        self.assertEqual(config.output_dir, Path("outputs/synthetic"))
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.image_size, 32)
        self.assertEqual(config.point_count, 128)

    def test_experiment_config_reads_training_and_data_fields(self) -> None:
        config = ExperimentRunConfig.from_mapping(
            {
                "runtime": {
                    "device": "cpu",
                    "seed": 123,
                    "data": {
                        "manifest_path": "data/manifests/train.jsonl",
                        "train_manifest_path": "data/manifests/train.split.jsonl",
                        "eval_manifest_path": "data/manifests/val.split.jsonl",
                        "satellite_raster_config_path": "data/satellite_rasters/config.json",
                        "image_size": 64,
                        "point_count": 256,
                    },
                    "training": {
                        "mode": "manifest-smoke",
                        "output_dir": "outputs/train",
                        "epochs": 2,
                        "batch_size": 3,
                        "learning_rate": 0.0002,
                    },
                    "evaluation": {
                        "checkpoint": "outputs/train/manifest_smoke_scaffold.pt",
                    },
                }
            }
        )

        self.assertEqual(config.training_mode, "manifest-smoke")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.manifest_path, Path("data/manifests/train.jsonl"))
        self.assertEqual(config.train_manifest_path, Path("data/manifests/train.split.jsonl"))
        self.assertEqual(config.eval_manifest_path, Path("data/manifests/val.split.jsonl"))
        self.assertEqual(config.satellite_raster_config_path, Path("data/satellite_rasters/config.json"))
        self.assertEqual(config.output_dir, Path("outputs/train"))
        self.assertEqual(config.checkpoint, Path("outputs/train/manifest_smoke_scaffold.pt"))
        self.assertEqual(config.epochs, 2)
        self.assertEqual(config.batch_size, 3)
        self.assertEqual(config.learning_rate, 0.0002)
        self.assertEqual(config.image_size, 64)
        self.assertEqual(config.point_count, 256)

    def test_train_and_eval_dispatch_use_split_manifests(self) -> None:
        from vggt_project import experiments

        seen: dict[str, Path] = {}
        devices: dict[str, str] = {}
        seeds: dict[str, int] = {}

        def fake_train_manifest_smoke(**kwargs) -> dict[str, float]:
            seen["train_manifest"] = kwargs["manifest_path"]
            devices["train"] = kwargs["device"]
            seeds["train"] = kwargs["seed"]
            return {"loss": 1.0}

        def fake_evaluate_manifest_smoke(**kwargs) -> dict[str, float]:
            seen["eval_manifest"] = kwargs["manifest_path"]
            devices["eval"] = kwargs["device"]
            return {"loss": 0.5}

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            train_manifest_path=Path("train.jsonl"),
            eval_manifest_path=Path("val.jsonl"),
            checkpoint=Path("checkpoint.pt"),
            device="cpu",
            seed=123,
        )
        original_train = experiments.train_manifest_smoke
        original_eval = experiments.evaluate_manifest_smoke
        experiments.train_manifest_smoke = fake_train_manifest_smoke
        experiments.evaluate_manifest_smoke = fake_evaluate_manifest_smoke
        try:
            experiments.train_from_config(config)
            experiments.evaluate_from_config(config)
        finally:
            experiments.train_manifest_smoke = original_train
            experiments.evaluate_manifest_smoke = original_eval

        self.assertEqual(seen["train_manifest"], Path("train.jsonl"))
        self.assertEqual(seen["eval_manifest"], Path("val.jsonl"))
        self.assertEqual(devices["train"], "cpu")
        self.assertEqual(devices["eval"], "cpu")
        self.assertEqual(seeds["train"], 123)

    @unittest.skipUnless(find_spec("yaml"), "PyYAML is required to load YAML configs")
    def test_load_experiment_config_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "experiment.yaml"
            config_path.write_text(
                "runtime:\n"
                "  training:\n"
                "    mode: synthetic\n"
                "    output_dir: outputs/from-yaml\n"
                "    epochs: 5\n",
                encoding="utf-8",
            )

            config = load_experiment_config(config_path)

        self.assertEqual(config.training_mode, "synthetic")
        self.assertEqual(config.output_dir, Path("outputs/from-yaml"))
        self.assertEqual(config.epochs, 5)

    def test_run_experiment_from_config_writes_report(self) -> None:
        def train_fn(config: ExperimentRunConfig) -> dict[str, float]:
            self.assertEqual(config.training_mode, "synthetic")
            return {"loss": 1.0, "checkpoint": str(config.checkpoint)}

        def evaluate_fn(config: ExperimentRunConfig) -> dict[str, float]:
            return {"loss": 0.5, "depth_mae": 0.25}

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "report.json"
            config = ExperimentRunConfig(
                training_mode="synthetic",
                output_dir=Path(temp_dir) / "out",
                checkpoint=Path(temp_dir) / "out" / "synthetic_scaffold.pt",
                seed=123,
            )

            report = run_experiment_from_config(
                config,
                report_path=report_path,
                train_fn=train_fn,
                evaluate_fn=evaluate_fn,
            )
            written_report = report_path.read_text(encoding="utf-8")
            report_exists = report_path.exists()

        self.assertTrue(report_exists)
        self.assertIn('"train_metrics"', written_report)
        self.assertEqual(report["mode"], "synthetic")
        self.assertEqual(report["config"]["seed"], 123)
        self.assertEqual(report["train_metrics"]["loss"], 1.0)
        self.assertEqual(report["eval_metrics"]["depth_mae"], 0.25)

    @unittest.skipUnless(
        find_spec("PIL") and find_spec("torch") and find_spec("yaml"),
        "Pillow, torch, and PyYAML are required for config-driven smoke training",
    )
    def test_config_driven_manifest_smoke_train_and_eval(self) -> None:
        from PIL import Image

        from vggt_project.experiments import evaluate_from_config, train_from_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "samples/CAM_FRONT").mkdir(parents=True)
            (root / "sat").mkdir()
            (root / "targets").mkdir()
            Image.new("RGB", (8, 8), color=(255, 0, 0)).save(root / "samples/CAM_FRONT/a.png")
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(root / "sat/patch.png")
            Image.new("L", (8, 8), color=128).save(root / "targets/depth.png")
            Image.new("L", (8, 8), color=255).save(root / "targets/mask.png")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.png"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"lidar_depth_path":"targets/depth.png",'
                '"valid_area_mask_path":"targets/mask.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            config_path = root / "experiment.yaml"
            config_path.write_text(
                "runtime:\n"
                "  data:\n"
                f"    manifest_path: {manifest}\n"
                "    image_size: 16\n"
                "    point_count: 4\n"
                "  training:\n"
                "    mode: manifest-smoke\n"
                f"    output_dir: {root / 'out'}\n"
                "    epochs: 1\n"
                "    batch_size: 1\n"
                "  evaluation:\n"
                f"    checkpoint: {root / 'out' / 'manifest_smoke_scaffold.pt'}\n",
                encoding="utf-8",
            )
            config = load_experiment_config(config_path)

            train_metrics = train_from_config(config)
            eval_metrics = evaluate_from_config(config)

        self.assertIn("checkpoint", train_metrics)
        self.assertIn("loss", eval_metrics)
        self.assertIn("depth_mae", eval_metrics)


if __name__ == "__main__":
    unittest.main()
