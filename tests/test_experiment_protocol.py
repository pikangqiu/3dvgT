import unittest


class ExperimentProtocolTest(unittest.TestCase):
    def test_protocol_contains_reconstruction_first_metrics_and_baselines(self) -> None:
        from vggt_project.experiment_protocol import build_experiment_protocol

        protocol = build_experiment_protocol()
        metric_names = {metric.name for metric in protocol.primary_metrics}
        baseline_names = {baseline.name for baseline in protocol.baselines}

        self.assertIn("depth_mae", metric_names)
        self.assertIn("scale_aligned_pointmap_chamfer", metric_names)
        self.assertIn("gravity_error_deg", metric_names)
        self.assertIn("bev_occupancy_iou", {metric.name for metric in protocol.auxiliary_metrics})
        self.assertIn("VGGT", baseline_names)
        self.assertIn("G3T", baseline_names)
        self.assertIn("BEV+Satellite+G3T", baseline_names)
        self.assertIn("Cross3R", baseline_names)
        self.assertIn("ReconDrive", baseline_names)
        self.assertIn("DynamicVGGT", baseline_names)
        self.assertIn("Sat3DGen", baseline_names)
        self.assertIn("SA-Occ", baseline_names)
        self.assertIn("DriveTok", baseline_names)
        self.assertIn("DGGT", baseline_names)

    def test_protocol_recommends_benchmarks_by_role(self) -> None:
        from vggt_project.experiment_protocol import build_experiment_protocol, format_experiment_protocol

        protocol = build_experiment_protocol()
        rendered = format_experiment_protocol(protocol)
        benchmark_names = {benchmark.name for benchmark in protocol.benchmarks}

        self.assertIn("E3D-Bench", benchmark_names)
        self.assertIn("Occ3D-nuScenes", benchmark_names)
        self.assertIn("Sky2Ground", benchmark_names)
        self.assertIn("CrossGeo", benchmark_names)
        self.assertIn("Sat3DGen-VIGOR-OOD-DSM", benchmark_names)
        self.assertIn("DynamicVGGT", benchmark_names)
        self.assertIn("SA-Occ", benchmark_names)
        self.assertIn("DriveTok", benchmark_names)
        self.assertIn("M2-Occ", benchmark_names)
        self.assertIn("PAGE-4D", benchmark_names)
        self.assertIn("UniOcc", benchmark_names)
        self.assertIn("OpenScene", benchmark_names)
        self.assertIn("SG-BEV", benchmark_names)
        self.assertIn("satellite/cross-view reconstruction table", rendered)
        self.assertIn("driving reconstruction table", rendered)
        self.assertIn("primary reconstruction table", rendered)
        self.assertIn("auxiliary satellite/BEV alignment", rendered)

    def test_protocol_marks_first_wave_experiment_priorities(self) -> None:
        from vggt_project.experiment_protocol import build_experiment_protocol, format_experiment_protocol

        protocol = build_experiment_protocol()
        baseline_by_name = {baseline.name: baseline for baseline in protocol.baselines}
        benchmark_by_name = {benchmark.name: benchmark for benchmark in protocol.benchmarks}
        rendered = format_experiment_protocol(protocol)

        self.assertEqual("phase1_core", baseline_by_name["G3T"].experiment_phase)
        self.assertEqual("phase1_core", baseline_by_name["BEV+Satellite+G3T"].experiment_phase)
        self.assertEqual("phase1_external", benchmark_by_name["SA-Occ"].experiment_phase)
        self.assertEqual("phase1_external", benchmark_by_name["Occ3D-nuScenes"].experiment_phase)
        self.assertEqual("phase2_tracking", baseline_by_name["DynamicVGGT"].experiment_phase)
        self.assertIn("phase=phase1_core", rendered)
        self.assertIn("phase=phase1_external", rendered)


if __name__ == "__main__":
    unittest.main()
