"""
GOLIATH Model Optimization Pipeline - Benchmarking & Quality Assurance Engine
Validates latency speedups and numerical fidelity (Cosine Similarity, MAE) across quantized variants.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("goliath.optimization.benchmark")

class OptimizationBenchmark:
    """
    Rigorously benchmarks inference latency and numerical accuracy between baseline and optimized models.
    """

    @staticmethod
    def run_benchmark(
        fp32_model_path: str,
        quantized_model_path: str,
        input_shape: List[int] = [1, 3, 640, 640],
        iterations: int = 50,
        warmup_iterations: int = 10,
    ) -> Dict[str, Any]:
        import onnxruntime as ort

        # Initialize sessions
        session_fp32 = ort.InferenceSession(fp32_model_path, providers=["CPUExecutionProvider"])
        session_quant = ort.InferenceSession(quantized_model_path, providers=["CPUExecutionProvider"])

        input_name_fp32 = session_fp32.get_inputs()[0].name
        input_name_quant = session_quant.get_inputs()[0].name

        # Create reproducible synthetic test image
        np.random.seed(42)
        dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

        # Warm-up passes
        for _ in range(warmup_iterations):
            session_fp32.run(None, {input_name_fp32: dummy_input})
            session_quant.run(None, {input_name_quant: dummy_input})

        # Benchmark FP32 Latency
        fp32_latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out_fp32 = session_fp32.run(None, {input_name_fp32: dummy_input})[0]
            fp32_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Benchmark Quantized Latency
        quant_latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            out_quant = session_quant.run(None, {input_name_quant: dummy_input})[0]
            quant_latencies.append((time.perf_counter() - t0) * 1000.0)

        # Compute numerical metrics
        flat_fp32 = out_fp32.flatten().astype(np.float64)
        flat_quant = out_quant.flatten().astype(np.float64)

        mae = float(np.mean(np.abs(flat_fp32 - flat_quant)))
        max_error = float(np.max(np.abs(flat_fp32 - flat_quant)))
        
        # Cosine Similarity
        norm_fp32 = np.linalg.norm(flat_fp32)
        norm_quant = np.linalg.norm(flat_quant)
        if norm_fp32 > 0 and norm_quant > 0:
            cosine_sim = float(np.dot(flat_fp32, flat_quant) / (norm_fp32 * norm_quant))
        else:
            cosine_sim = 1.0

        avg_fp32_ms = float(np.mean(fp32_latencies))
        avg_quant_ms = float(np.mean(quant_latencies))
        p95_quant_ms = float(np.percentile(quant_latencies, 95))
        speedup = avg_fp32_ms / max(avg_quant_ms, 1e-5)

        logger.info(
            f"[Benchmark] FP32 Avg: {avg_fp32_ms:.2f}ms | Quantized Avg: {avg_quant_ms:.2f}ms "
            f"(Speedup: {speedup:.2f}x) | Cosine Sim: {cosine_sim:.4f} | MAE: {mae:.5f}"
        )

        return {
            "baseline_avg_latency_ms": avg_fp32_ms,
            "quantized_avg_latency_ms": avg_quant_ms,
            "quantized_p95_latency_ms": p95_quant_ms,
            "speedup_ratio": speedup,
            "cosine_similarity": cosine_sim,
            "mean_absolute_error": mae,
            "max_absolute_error": max_error,
            "quality_gate_passed": bool(cosine_sim >= 0.98),
        }
