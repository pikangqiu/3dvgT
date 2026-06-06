import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("numpy"), "numpy is required for supervision pipeline tests")
class SupervisionPipelineTest(unittest.TestCase):
    def test_supervision_pipeline_chains_depth_then_pointmap_manifests(self) -> None:
        import vggt_project.data.supervision_pipeline as pipeline
        from vggt_project.data.nuscenes_depth import LidarDepthReport
        from vggt_project.data.nuscenes_pointmap import LidarPointmapReport
        from vggt_project.data.supervision_pipeline import materialize_lidar_supervision_manifest

        class FakeNuScenes:
            pass

        def fake_depth(
            nusc,
            manifest_path,
            *,
            camera_name,
            camera_names,
            depth_dir,
            output_manifest_path,
            max_depth_m,
            overwrite,
        ):
            records = _read_jsonl(manifest_path)
            for record in records:
                record["lidar_depth_path"] = "depth/sample-1_CAM_FRONT.png"
            _write_jsonl(records, output_manifest_path)
            return LidarDepthReport(
                manifest_path=manifest_path,
                output_manifest_path=output_manifest_path,
                camera_name=",".join(camera_names or (camera_name,)),
                sample_count=len(records),
                depth_maps_written=1,
                camera_names=tuple(camera_names or (camera_name,)),
            )

        def fake_pointmap(
            nusc,
            manifest_path,
            *,
            pointmap_dir,
            output_manifest_path,
            max_points,
            overwrite,
        ):
            records = _read_jsonl(manifest_path)
            self.assertEqual(records[0]["lidar_depth_path"], "depth/sample-1_CAM_FRONT.png")
            for record in records:
                record["pointmap_path"] = "pointmaps/sample-1_LIDAR_TOP.npy"
            _write_jsonl(records, output_manifest_path)
            return LidarPointmapReport(
                manifest_path=manifest_path,
                output_manifest_path=output_manifest_path,
                sample_count=len(records),
                pointmaps_written=1,
            )

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
            output = root / "samples.supervised.jsonl"
            original_depth = pipeline.materialize_lidar_depth_manifest
            original_pointmap = pipeline.materialize_lidar_pointmap_manifest
            pipeline.materialize_lidar_depth_manifest = fake_depth
            pipeline.materialize_lidar_pointmap_manifest = fake_pointmap
            try:
                report = materialize_lidar_supervision_manifest(
                    FakeNuScenes(),
                    manifest,
                    output_manifest_path=output,
                    camera_names=("CAM_FRONT", "CAM_BACK"),
                )
            finally:
                pipeline.materialize_lidar_depth_manifest = original_depth
                pipeline.materialize_lidar_pointmap_manifest = original_pointmap

            record = json.loads(output.read_text(encoding="utf-8"))
            depth_manifest_exists = report.depth_manifest_path.exists()

        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.depth_maps_written, 1)
        self.assertEqual(report.pointmaps_written, 1)
        self.assertTrue(depth_manifest_exists)
        self.assertEqual(record["lidar_depth_path"], "depth/sample-1_CAM_FRONT.png")
        self.assertEqual(record["pointmap_path"], "pointmaps/sample-1_LIDAR_TOP.npy")

    def test_supervision_pipeline_can_write_camera_pointmap_paths(self) -> None:
        import vggt_project.data.supervision_pipeline as pipeline
        from vggt_project.data.nuscenes_depth import LidarDepthReport
        from vggt_project.data.nuscenes_pointmap import LidarPointmapReport
        from vggt_project.data.supervision_pipeline import materialize_lidar_supervision_manifest

        class FakeNuScenes:
            pass

        def fake_depth(
            nusc,
            manifest_path,
            *,
            camera_name,
            camera_names,
            depth_dir,
            output_manifest_path,
            max_depth_m,
            overwrite,
        ):
            records = _read_jsonl(manifest_path)
            for record in records:
                record["lidar_depth_paths"] = {
                    "CAM_FRONT": "depth/sample-1_CAM_FRONT.png",
                    "CAM_BACK": "depth/sample-1_CAM_BACK.png",
                }
            _write_jsonl(records, output_manifest_path)
            return LidarDepthReport(
                manifest_path=manifest_path,
                output_manifest_path=output_manifest_path,
                camera_name="CAM_FRONT,CAM_BACK",
                sample_count=len(records),
                depth_maps_written=2,
                camera_names=("CAM_FRONT", "CAM_BACK"),
            )

        def fake_camera_pointmap(
            nusc,
            manifest_path,
            *,
            camera_name,
            camera_names,
            pointmap_dir,
            output_manifest_path,
            max_points,
            overwrite,
        ):
            records = _read_jsonl(manifest_path)
            self.assertIn("lidar_depth_paths", records[0])
            for record in records:
                record["pointmap_paths"] = {
                    "CAM_FRONT": "camera_pointmaps/sample-1_CAM_FRONT.npy",
                    "CAM_BACK": "camera_pointmaps/sample-1_CAM_BACK.npy",
                }
            _write_jsonl(records, output_manifest_path)
            return LidarPointmapReport(
                manifest_path=manifest_path,
                output_manifest_path=output_manifest_path,
                sample_count=len(records),
                pointmaps_written=2,
                camera_names=tuple(camera_names or (camera_name,)),
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                '{"token":"sample-1","scene_token":"scene-1","timestamp_us":10,'
                '"camera_paths":["samples/CAM_FRONT/a.jpg","samples/CAM_BACK/b.jpg"],'
                '"camera_names":["CAM_FRONT","CAM_BACK"],'
                '"satellite_patch_path":"sat/sample-1.png",'
                '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
                '"satellite_frame":"satellite"}\n',
                encoding="utf-8",
            )
            output = root / "samples.supervised.jsonl"
            original_depth = pipeline.materialize_lidar_depth_manifest
            original_camera_pointmap = pipeline.materialize_camera_lidar_pointmap_manifest
            pipeline.materialize_lidar_depth_manifest = fake_depth
            pipeline.materialize_camera_lidar_pointmap_manifest = fake_camera_pointmap
            try:
                report = materialize_lidar_supervision_manifest(
                    FakeNuScenes(),
                    manifest,
                    output_manifest_path=output,
                    camera_names=("CAM_FRONT", "CAM_BACK"),
                    pointmap_target_frame="camera",
                )
            finally:
                pipeline.materialize_lidar_depth_manifest = original_depth
                pipeline.materialize_camera_lidar_pointmap_manifest = original_camera_pointmap

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report.depth_maps_written, 2)
        self.assertEqual(report.pointmaps_written, 2)
        self.assertEqual(record["pointmap_paths"]["CAM_FRONT"], "camera_pointmaps/sample-1_CAM_FRONT.npy")
        self.assertEqual(record["pointmap_paths"]["CAM_BACK"], "camera_pointmaps/sample-1_CAM_BACK.npy")


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    unittest.main()
