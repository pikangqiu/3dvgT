import unittest
from pathlib import Path

from vggt_project.config import ProjectConfig
from vggt_project.references import collect_reference_status


class ProjectScaffoldTest(unittest.TestCase):
    def test_project_config_is_reconstruction_first(self) -> None:
        config = ProjectConfig()

        self.assertEqual(config.task_framing, "3d_reconstruction_first")
        self.assertIn("pointmap_reconstruction", config.objective.primary_losses)
        self.assertIn("long_sequence_alignment_drift", config.objective.primary_metrics)

    def test_reference_statuses_are_named(self) -> None:
        statuses = collect_reference_status()
        by_name = {status.name: status for status in statuses}

        self.assertEqual(by_name["g3t"].path, Path("refs/g3t").resolve())
        self.assertIn("g3t", by_name)
        self.assertTrue(by_name["look_from_above_notes"].exists)
        self.assertIn("pseudomaptrainer_component", by_name)
        self.assertIn("maptr_component", by_name)
        self.assertIn("e3d_bench_reference", by_name)
        self.assertIn("open_occupancy_reference", by_name)
        self.assertIn("surround_occ_reference", by_name)
        self.assertTrue(by_name["look_from_above_paper"].exists)


if __name__ == "__main__":
    unittest.main()
