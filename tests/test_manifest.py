import tempfile
import unittest
from pathlib import Path

from vggt_project.data.manifest import load_manifest


class ManifestTest(unittest.TestCase):
    def test_load_manifest_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"camera_names":["CAM_FRONT"],'
                '"satellite_patch_path":"sat/patch.png",'
                '"occupancy_path":"targets/occupancy.npy",'
                '"pointmap_path":"targets/pointmap.npy",'
                '"pointmap_paths":{"CAM_FRONT":"targets/CAM_FRONT_pointmap.npy"},'
                '"camera_local_camera_to_gravity_poses":{"CAM_FRONT":[1.0,0.0,0.0,0.0]},'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )

            samples = load_manifest(manifest)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].token, "sample-1")
        self.assertEqual(samples[0].cameras[0].image_path, root / "samples/CAM_FRONT/a.jpg")
        self.assertEqual(samples[0].satellite_patch_path, root / "sat/patch.png")
        self.assertEqual(samples[0].occupancy_path, root / "targets/occupancy.npy")
        self.assertEqual(samples[0].pointmap_path, root / "targets/pointmap.npy")
        self.assertEqual(samples[0].pointmap_paths["CAM_FRONT"], root / "targets/CAM_FRONT_pointmap.npy")
        self.assertEqual(
            samples[0].camera_local_camera_to_gravity_poses["CAM_FRONT"],
            (1.0, 0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
