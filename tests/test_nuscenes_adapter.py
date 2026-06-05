import tempfile
import unittest
from pathlib import Path

from vggt_project.data.nuscenes_adapter import NuScenesAdapterConfig, inspect_nuscenes_root


class NuScenesAdapterTest(unittest.TestCase):
    def test_missing_root_reports_required_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "nuscenes"

            status = inspect_nuscenes_root(NuScenesAdapterConfig(root=missing, version="v1.0-mini"))

        self.assertFalse(status.ready)
        self.assertIn("root", status.missing)
        self.assertIn("samples", status.expected_layout)
        self.assertIn("maps", status.expected_layout)

    def test_existing_mini_layout_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "nuscenes"
            for child in ("samples", "sweeps", "maps", "v1.0-mini"):
                (root / child).mkdir(parents=True)

            status = inspect_nuscenes_root(NuScenesAdapterConfig(root=root, version="v1.0-mini"))

        self.assertTrue(status.ready)
        self.assertEqual(status.version, "v1.0-mini")
        self.assertEqual(status.missing, ())


if __name__ == "__main__":
    unittest.main()
