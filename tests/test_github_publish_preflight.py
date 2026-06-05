import unittest
from subprocess import CompletedProcess


class GitHubPublishPreflightTest(unittest.TestCase):
    def test_preflight_reports_missing_auth_and_origin(self) -> None:
        from vggt_project.github_publish import run_publish_preflight

        def runner(command):
            joined = " ".join(command)
            if joined == "git status --porcelain":
                return CompletedProcess(command, 0, stdout="", stderr="")
            if joined == "git branch --show-current":
                return CompletedProcess(command, 0, stdout="main\n", stderr="")
            if joined == "git remote get-url origin":
                return CompletedProcess(command, 2, stdout="", stderr="no origin\n")
            if joined == "gh auth status":
                return CompletedProcess(command, 1, stdout="", stderr="not logged in\n")
            raise AssertionError(command)

        report = run_publish_preflight(runner=runner)

        self.assertTrue(report.clean_worktree)
        self.assertEqual(report.current_branch, "main")
        self.assertFalse(report.has_origin)
        self.assertFalse(report.gh_authenticated)
        self.assertFalse(report.ready_to_publish)
        self.assertIn("gh auth login", report.next_steps)

    def test_preflight_ready_when_clean_authenticated_and_origin_exists(self) -> None:
        from vggt_project.github_publish import run_publish_preflight

        def runner(command):
            joined = " ".join(command)
            if joined == "git status --porcelain":
                return CompletedProcess(command, 0, stdout="", stderr="")
            if joined == "git branch --show-current":
                return CompletedProcess(command, 0, stdout="main\n", stderr="")
            if joined == "git remote get-url origin":
                return CompletedProcess(command, 0, stdout="git@github.com:user/VggT.git\n", stderr="")
            if joined == "gh auth status":
                return CompletedProcess(command, 0, stdout="Logged in\n", stderr="")
            raise AssertionError(command)

        report = run_publish_preflight(runner=runner)

        self.assertTrue(report.ready_to_publish)
        self.assertEqual(report.origin_url, "git@github.com:user/VggT.git")

    def test_preflight_ready_when_clean_and_origin_exists_without_gh_auth(self) -> None:
        from vggt_project.github_publish import run_publish_preflight

        def runner(command):
            joined = " ".join(command)
            if joined == "git status --porcelain":
                return CompletedProcess(command, 0, stdout="", stderr="")
            if joined == "git branch --show-current":
                return CompletedProcess(command, 0, stdout="main\n", stderr="")
            if joined == "git remote get-url origin":
                return CompletedProcess(command, 0, stdout="https://github.com/user/VggT.git\n", stderr="")
            if joined == "gh auth status":
                return CompletedProcess(command, 1, stdout="", stderr="not logged in\n")
            raise AssertionError(command)

        report = run_publish_preflight(runner=runner)

        self.assertTrue(report.has_origin)
        self.assertFalse(report.gh_authenticated)
        self.assertTrue(report.ready_to_publish)
        self.assertNotIn("gh auth login", report.next_steps)


if __name__ == "__main__":
    unittest.main()
