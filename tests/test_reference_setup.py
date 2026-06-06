import tempfile
import unittest
from pathlib import Path

from vggt_project.reference_setup import (
    reference_clone_plans,
    reference_specs,
)


class ReferenceSetupTest(unittest.TestCase):
    def test_reference_specs_include_core_and_benchmark_repositories(self) -> None:
        names = {spec.name for spec in reference_specs()}

        self.assertIn("g3t", names)
        self.assertIn("pseudomaptrainer_component", names)
        self.assertIn("maptr_component", names)
        self.assertIn("e3d_bench_reference", names)
        self.assertIn("dggt_reference", names)
        self.assertIn("drivingforward_reference", names)
        self.assertIn("gaussianocc_reference", names)
        self.assertIn("openscene_reference", names)
        self.assertIn("uniocc_reference", names)

    def test_clone_plans_are_relative_to_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            plans = reference_clone_plans(root=root)

        g3t_plan = next(plan for plan in plans if plan.spec.name == "g3t")
        self.assertEqual(g3t_plan.path, root / "refs/g3t")
        self.assertEqual(
            g3t_plan.command,
            [
                "git",
                "clone",
                "https://github.com/g3t-paper/g3t.git",
                str(root / "refs/g3t"),
            ],
        )

    def test_benchmark_clone_plans_use_benchmark_reference_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            plans = reference_clone_plans(root=root)

        expected = {
            "dggt_reference": (
                "https://github.com/xiaomi-research/dggt.git",
                "refs/benchmarks/DGGT",
            ),
            "drivingforward_reference": (
                "https://github.com/fangzhou2000/DrivingForward.git",
                "refs/benchmarks/DrivingForward",
            ),
            "gaussianocc_reference": (
                "https://github.com/GANWANSHUI/GaussianOcc.git",
                "refs/benchmarks/GaussianOcc",
            ),
            "openscene_reference": (
                "https://github.com/OpenDriveLab/OpenScene.git",
                "refs/benchmarks/OpenScene",
            ),
            "uniocc_reference": (
                "https://github.com/tasl-lab/UniOcc.git",
                "refs/benchmarks/UniOcc",
            ),
        }

        plans_by_name = {plan.spec.name: plan for plan in plans}
        for name, (url, path) in expected.items():
            with self.subTest(name=name):
                plan = plans_by_name[name]
                self.assertEqual(plan.path, root / path)
                self.assertEqual(plan.command, ["git", "clone", url, str(root / path)])

    def test_existing_repository_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "refs/g3t/.git").mkdir(parents=True)
            plans = reference_clone_plans(root=root)

        g3t_plan = next(plan for plan in plans if plan.spec.name == "g3t")
        self.assertTrue(g3t_plan.exists)


if __name__ == "__main__":
    unittest.main()
