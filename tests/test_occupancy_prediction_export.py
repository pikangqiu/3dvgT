import json
import tempfile
import unittest
from pathlib import Path


class OccupancyPredictionExportTest(unittest.TestCase):
    def test_materialize_occupancy_predictions_writes_benchmark_manifest(self) -> None:
        from vggt_project.data.occupancy_predictions import materialize_occupancy_prediction_manifest

        def fake_prediction(record: dict, index: int) -> dict:
            self.assertEqual(record["token"], "sample-1")
            self.assertEqual(index, 0)
            return {"bev_occupancy": [[0.1, 0.8], [0.7, 0.2]]}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "targets").mkdir()
            (root / "targets/gt.json").write_text(json.dumps([[0, 1], [1, 0]]), encoding="utf-8")
            manifest = root / "samples.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "token": "sample-1",
                        "occupancy_path": "targets/gt.json",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "samples.predicted.jsonl"

            report = materialize_occupancy_prediction_manifest(
                manifest,
                prediction_fn=fake_prediction,
                target_dir=Path("occupancy_predictions"),
                output_manifest_path=output,
                array_format="json",
                binary_threshold=0.5,
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            prediction_path = root / record["predicted_occupancy_path"]
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))

        self.assertEqual(report.sample_count, 1)
        self.assertEqual(report.prediction_maps_written, 1)
        self.assertEqual(record["occupancy_path"], "targets/gt.json")
        self.assertEqual(record["predicted_occupancy_path"], "occupancy_predictions/sample-1.json")
        self.assertEqual(prediction, [[0, 1], [1, 0]])

    def test_output_manifest_relative_paths_resolve_from_output_directory(self) -> None:
        from vggt_project.data.occupancy_predictions import materialize_occupancy_prediction_manifest

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "benchmark"
            input_dir.mkdir()
            manifest = input_dir / "samples.jsonl"
            manifest.write_text(json.dumps({"token": "sample-1"}) + "\n", encoding="utf-8")
            output = output_dir / "samples.predicted.jsonl"

            materialize_occupancy_prediction_manifest(
                manifest,
                prediction_fn=lambda _record, _index: {"bev_occupancy": [[0.9]]},
                target_dir=Path("predictions"),
                output_manifest_path=output,
                array_format="json",
            )

            record = json.loads(output.read_text(encoding="utf-8"))
            prediction_path = output.parent / record["predicted_occupancy_path"]
            prediction_exists = prediction_path.exists()

        self.assertEqual(record["predicted_occupancy_path"], "predictions/sample-1.json")
        self.assertTrue(prediction_exists)


if __name__ == "__main__":
    unittest.main()
