"""
GOLIATH Model Optimization Pipeline - Integration Tests & Benchmarks
"""

import os
import sys
import unittest

# Ensure root path in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from goliath_optimization.quantizer import ModelQuantizer
from goliath_optimization.benchmark import OptimizationBenchmark
from goliath_optimization.pipeline import ModelOptimizationPipeline

class TestGoliathOptimization(unittest.TestCase):

    def test_01_dynamic_int8_quantization_and_benchmark(self):
        """Quantizes baseline yolov8n.onnx to INT8, verifying size compression and cosine fidelity."""
        raw_onnx = os.path.join(ROOT_DIR, "yolov8n.onnx")
        self.assertTrue(os.path.exists(raw_onnx), "Baseline yolov8n.onnx must exist")

        quantizer = ModelQuantizer(output_dir="test_optimized_models")
        int8_path = quantizer.quantize_dynamic_int8(raw_onnx)
        self.assertTrue(os.path.exists(int8_path))

        # Check compression ratio
        raw_size_mb = os.path.getsize(raw_onnx) / (1024 * 1024)
        int8_size_mb = os.path.getsize(int8_path) / (1024 * 1024)
        
        print(f"\n[Compression] Raw FP32: {raw_size_mb:.2f}MB | INT8 Quantized: {int8_size_mb:.2f}MB")
        self.assertTrue(int8_size_mb < raw_size_mb, "INT8 model size must be significantly smaller than FP32")

        # Run benchmark
        results = OptimizationBenchmark.run_benchmark(
            fp32_model_path=raw_onnx,
            quantized_model_path=int8_path,
            iterations=15,
            warmup_iterations=3,
        )

        print(
            f"[Quality Benchmark] Cosine Similarity = {results['cosine_similarity']:.4f} | "
            f"MAE = {results['mean_absolute_error']:.5f} | Speedup = {results['speedup_ratio']:.2f}x"
        )
        self.assertTrue(results["cosine_similarity"] >= 0.98, "Quantized model must maintain >= 0.98 cosine fidelity")

    def test_02_end_to_end_pipeline_and_registry(self):
        """Runs full automated optimization pipeline and validates manifest registration."""
        raw_onnx = os.path.join(ROOT_DIR, "yolov8n.onnx")
        pipeline = ModelOptimizationPipeline(
            output_dir="test_optimized_models",
            registry_storage_root="test_models_storage",
        )

        manifest = pipeline.optimize_and_register(
            raw_onnx_path=raw_onnx,
            model_name="yolov8n-detector",
            version_tag="v1.2.0-int8",
            target_accelerators=["wasm", "webgl", "cpu"],
        )

        self.assertEqual(manifest["version"], "v1.2.0-int8")
        self.assertEqual(len(manifest["sha256"]), 64)
        self.assertIn("benchmark_metrics", manifest)
        self.assertTrue(manifest["benchmark_metrics"]["quality_gate_passed"])


if __name__ == "__main__":
    unittest.main()
