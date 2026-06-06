import unittest
from pathlib import Path

from vggt_project.experiments import ExperimentRunConfig


class TrainingPlanTest(unittest.TestCase):
    def test_plan_lists_preprocessing_commands_for_missing_manifest_split(self) -> None:
        from vggt_project.training_plan import build_training_run_plan, format_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("data/manifests/nuscenes-mini.supervised.jsonl"),
            train_manifest_path=Path("data/manifests/nuscenes-mini.train.jsonl"),
            eval_manifest_path=Path("data/manifests/nuscenes-mini.val.jsonl"),
            satellite_raster_config_path=Path("data/satellite_rasters/config.json"),
            image_size=32,
            point_count=128,
        )

        plan = build_training_run_plan(config)
        rendered = format_training_run_plan(plan)

        self.assertFalse(plan.ready_to_train)
        self.assertIn("scripts/generate_manifest.py", rendered)
        self.assertIn("scripts/materialize_satellite_crops.py", rendered)
        self.assertIn("scripts/generate_camera_pose_targets.py", rendered)
        self.assertIn("--pointmap-target-frame camera", rendered)
        self.assertIn("scripts/generate_lidar_occupancy_targets.py", rendered)
        self.assertIn("scripts/generate_reference_supervision_targets.py", rendered)
        self.assertIn("scripts/inspect_manifest_sample.py", rendered)
        self.assertIn("scripts/check_model_adapter.py", rendered)
        self.assertIn("scripts/split_manifest.py", rendered)
        self.assertIn("data/manifests/nuscenes-mini.train.jsonl", plan.missing_outputs)
        self.assertIn("data/manifests/nuscenes-mini.val.jsonl", plan.missing_outputs)
        self.assertIn("scripts/train.py --config configs/reconstruction_first.yaml", rendered)

    def test_plan_is_ready_when_required_outputs_exist(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(
            config,
            path_exists=lambda path: path.name in {"ready.supervised.jsonl", "ready.train.jsonl", "ready.val.jsonl"},
        )

        self.assertTrue(plan.ready_to_train)
        self.assertEqual(plan.missing_outputs, ())


if __name__ == "__main__":
    unittest.main()
