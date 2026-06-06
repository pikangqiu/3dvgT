import json
import unittest
from pathlib import Path

from vggt_project.experiments import ExperimentRunConfig
from vggt_project.training_readiness import DependencyStatus


class TrainingLaunchTest(unittest.TestCase):
    def test_launch_packet_combines_readiness_plan_and_next_commands(self) -> None:
        from vggt_project.training_launch import build_training_launch_packet, format_training_launch_packet

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("data/manifests/nuscenes-mini.supervised.jsonl"),
            train_manifest_path=Path("data/manifests/nuscenes-mini.train.jsonl"),
            eval_manifest_path=Path("data/manifests/nuscenes-mini.val.jsonl"),
            satellite_raster_config_path=Path("data/satellite_rasters/config.json"),
            image_size=32,
            point_count=128,
        )

        packet = build_training_launch_packet(
            config,
            config_path=Path("configs/reconstruction_first.json"),
            dependency_probe=lambda: (
                DependencyStatus("torch", False, None),
                DependencyStatus("PIL", True, "test"),
                DependencyStatus("numpy", True, "test"),
                DependencyStatus("yaml", True, "test"),
            ),
            device_probe=lambda device: True,
            plan_path_exists=lambda path: False,
        )
        payload = json.loads(packet.to_json())
        rendered = format_training_launch_packet(packet)

        self.assertFalse(packet.ready_to_launch)
        self.assertIn("missing_path: manifest_path", packet.blockers)
        self.assertIn("plan_missing_output: generate_base_manifest", " ".join(packet.blockers))
        self.assertEqual(payload["readiness"]["ready"], False)
        self.assertEqual(payload["plan"]["ready_to_train"], False)
        self.assertIn("remediation_commands", payload)
        self.assertTrue(
            any("scripts/setup_env.sh" in command for command in payload["remediation_commands"])
        )
        self.assertTrue(
            any("scripts/prepare_nuscenes.sh" in command for command in payload["remediation_commands"])
        )
        self.assertTrue(
            any("scripts/prepare_satellite_rasters.sh" in command for command in payload["remediation_commands"])
        )
        self.assertFalse(any("scripts/train.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/evaluate.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/attach_occ3d_labels.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/validate_occupancy_labels.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/export_occupancy_predictions.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/evaluate_occupancy_benchmark.py" in command for command in payload["remediation_commands"]))
        self.assertFalse(any("scripts/verify_training_artifacts.py" in command for command in payload["remediation_commands"]))
        self.assertIn("scripts/check_nuscenes.py", payload["next_commands"][0])
        self.assertTrue(
            any("scripts/generate_manifest.py" in command for command in payload["next_commands"])
        )
        self.assertIn("ready_to_launch: false", rendered)
        self.assertIn("remediation_commands:", rendered)
        self.assertIn("next_commands:", rendered)

    def test_launch_packet_recommends_weight_preparation_for_missing_checkpoint(self) -> None:
        from vggt_project.training_launch import build_training_launch_packet

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=None,
            train_manifest_path=None,
            eval_manifest_path=None,
            weights_path=Path("checkpoints/g3t/model.pt"),
            image_size=32,
            point_count=128,
        )

        packet = build_training_launch_packet(
            config,
            config_path=Path("configs/reconstruction_first.json"),
            dependency_probe=lambda: (),
            device_probe=lambda device: True,
            plan_path_exists=lambda path: True,
        )

        self.assertIn("missing_path: weights_path", packet.blockers)
        self.assertTrue(
            any("scripts/prepare_model_weights.sh" in command for command in packet.remediation_commands)
        )


if __name__ == "__main__":
    unittest.main()
