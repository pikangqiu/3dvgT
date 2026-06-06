import unittest

from vggt_project.training_bootstrap import (
    format_training_bootstrap_report,
    run_training_bootstrap,
)
from vggt_project.training_plan import TrainingPlanStep, TrainingRunPlan


class TrainingBootstrapTest(unittest.TestCase):
    def test_dry_run_reports_commands_without_executing(self) -> None:
        calls: list[str] = []
        plan = TrainingRunPlan(
            steps=(
                TrainingPlanStep(name="prepare", command="echo prepare", ready=False),
                TrainingPlanStep(name="train", command="echo train", ready=False),
            ),
            missing_outputs=(),
            ready_to_train=False,
        )

        report = run_training_bootstrap(
            plan,
            execute=False,
            command_runner=lambda command: calls.append(command) or 0,
        )
        rendered = format_training_bootstrap_report(report)

        self.assertEqual(calls, [])
        self.assertFalse(report.executed)
        self.assertEqual(tuple(step.status for step in report.steps), ("dry-run", "dry-run"))
        self.assertIn("prepare: dry-run", rendered)
        self.assertIn("run: echo train", rendered)

    def test_execute_runs_pending_steps_until_requested_step(self) -> None:
        calls: list[str] = []
        plan = TrainingRunPlan(
            steps=(
                TrainingPlanStep(name="already_ready", command="echo ready", ready=True),
                TrainingPlanStep(name="prepare", command="echo prepare", ready=False),
                TrainingPlanStep(name="train", command="echo train", ready=False),
            ),
            missing_outputs=(),
            ready_to_train=False,
        )

        report = run_training_bootstrap(
            plan,
            execute=True,
            until="prepare",
            command_runner=lambda command: calls.append(command) or 0,
        )

        self.assertEqual(calls, ["echo prepare"])
        self.assertEqual(tuple(step.status for step in report.steps), ("skipped-ready", "passed"))
        self.assertTrue(report.executed)
        self.assertEqual(report.exit_code, 0)

    def test_execute_stops_on_first_failed_command(self) -> None:
        calls: list[str] = []
        plan = TrainingRunPlan(
            steps=(
                TrainingPlanStep(name="prepare", command="echo prepare", ready=False),
                TrainingPlanStep(name="train", command="echo train", ready=False),
            ),
            missing_outputs=(),
            ready_to_train=False,
        )

        report = run_training_bootstrap(
            plan,
            execute=True,
            command_runner=lambda command: calls.append(command) or 17,
        )

        self.assertEqual(calls, ["echo prepare"])
        self.assertEqual(tuple(step.status for step in report.steps), ("failed",))
        self.assertEqual(report.exit_code, 17)


if __name__ == "__main__":
    unittest.main()
