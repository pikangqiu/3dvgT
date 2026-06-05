import json
import tempfile
import unittest
from pathlib import Path

from vggt_project.data.manifest_split import split_manifest_by_scene


class ManifestSplitTest(unittest.TestCase):
    def test_split_manifest_by_scene_keeps_scenes_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                _record("a-1", "scene-a")
                + _record("a-2", "scene-a")
                + _record("b-1", "scene-b")
                + _record("c-1", "scene-c"),
                encoding="utf-8",
            )
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "eval.jsonl"

            report = split_manifest_by_scene(
                manifest,
                train_output_path=train_manifest,
                eval_output_path=eval_manifest,
                eval_fraction=0.34,
                seed="unit-test",
            )

            train_records = _read_jsonl(train_manifest)
            eval_records = _read_jsonl(eval_manifest)

        train_scenes = {record["scene_token"] for record in train_records}
        eval_scenes = {record["scene_token"] for record in eval_records}
        self.assertEqual(report.sample_count, 4)
        self.assertEqual(report.train_sample_count + report.eval_sample_count, 4)
        self.assertEqual(train_scenes & eval_scenes, set())
        self.assertGreaterEqual(report.eval_scene_count, 1)

    def test_split_manifest_by_explicit_eval_scene_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "samples.jsonl"
            manifest.write_text(
                _record("a-1", "scene-a")
                + _record("b-1", "scene-b")
                + _record("c-1", "scene-c"),
                encoding="utf-8",
            )
            train_manifest = root / "train.jsonl"
            eval_manifest = root / "eval.jsonl"

            report = split_manifest_by_scene(
                manifest,
                train_output_path=train_manifest,
                eval_output_path=eval_manifest,
                eval_scene_tokens={"scene-b"},
            )

            train_records = _read_jsonl(train_manifest)
            eval_records = _read_jsonl(eval_manifest)

        self.assertEqual(report.train_scene_count, 2)
        self.assertEqual(report.eval_scene_count, 1)
        self.assertEqual([record["token"] for record in eval_records], ["b-1"])
        self.assertEqual([record["token"] for record in train_records], ["a-1", "c-1"])


def _record(token: str, scene_token: str) -> str:
    return (
        f'{{"token":"{token}","scene_token":"{scene_token}","timestamp_us":10,'
        '"camera_paths":["samples/CAM_FRONT/a.jpg"],'
        '"satellite_patch_path":"sat/sample.png",'
        '"ego_pose_frame":"ego","bev_frame":"bev","gravity_frame":"gravity",'
        '"satellite_frame":"satellite"}\n'
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
