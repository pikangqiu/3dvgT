import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("numpy"), "numpy is required for nuScenes depth tests")
class NuScenesDepthTest(unittest.TestCase):
    def test_rasterize_camera_depth_keeps_nearest_projected_point(self) -> None:
        import numpy as np

        from vggt_project.data.nuscenes_depth import rasterize_camera_depth

        points = np.asarray(
            [
                [0.0, 0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0, 0.0],
                [10.0, 5.0, 10.0, -1.0],
            ],
            dtype=np.float32,
        )
        intrinsic = np.asarray(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        depth = rasterize_camera_depth(
            points_camera=points,
            camera_intrinsic=intrinsic,
            image_width=5,
            image_height=5,
            max_depth_m=10.0,
        )

        self.assertEqual(depth[2, 2], 128)
        self.assertEqual(int(depth.sum()), 128)

    @unittest.skipUnless(find_spec("PIL"), "Pillow is required for lidar depth manifest tests")
    def test_materialize_lidar_depth_manifest_updates_output_manifest(self) -> None:
        import numpy as np
        from PIL import Image

        from vggt_project.data import materialize_lidar_depth_manifest
        from vggt_project.data.nuscenes_depth import rasterize_camera_depth as real_rasterize
        import vggt_project.data.nuscenes_depth as depth_module

        class FakeNuScenes:
            dataroot = ""

        def fake_render(*args, **kwargs):
            depth = real_rasterize(
                points_camera=np.asarray([[0.0], [0.0], [5.0]], dtype=np.float32),
                camera_intrinsic=np.eye(3, dtype=np.float32),
                image_width=3,
                image_height=3,
                max_depth_m=10.0,
            )
            return Image.fromarray(depth, mode="L")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.depth.jsonl"
            original_render = depth_module.render_nuscenes_lidar_depth
            depth_module.render_nuscenes_lidar_depth = fake_render
            try:
                report = materialize_lidar_depth_manifest(
                    FakeNuScenes(),
                    manifest,
                    depth_dir=Path("depth"),
                    output_manifest_path=output,
                    camera_name="CAM_FRONT",
                )
            finally:
                depth_module.render_nuscenes_lidar_depth = original_render

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report.depth_maps_written, 1)
        self.assertEqual(record["lidar_depth_path"], "depth/sample-1_CAM_FRONT.png")


if __name__ == "__main__":
    unittest.main()
