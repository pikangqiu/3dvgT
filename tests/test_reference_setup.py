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

    def test_existing_repository_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "refs/g3t/.git").mkdir(parents=True)
            plans = reference_clone_plans(root=root)

        g3t_plan = next(plan for plan in plans if plan.spec.name == "g3t")
        self.assertTrue(g3t_plan.exists)


if __name__ == "__main__":
    unittest.main()
