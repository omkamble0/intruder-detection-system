"""
GOLIATH Model Optimization Pipeline - End-to-End Orchestrator
Coordinates graph compilation, quantization, benchmarking, and registry deployment.
"""

import os
import logging
from typing import Dict, Any, List

from goliath_optimization.quantizer import ModelQuantizer
from goliath_optimization.benchmark import OptimizationBenchmark
from goliath_control.model_registry import ModelRegistry

logger = logging.getLogger("goliath.optimization.pipeline")

class ModelOptimizationPipeline:
    """
    Automated pipeline transforming raw FP32 weights into certified, benchmarked edge artifacts.
    """

    def __init__(
        self,
        output_dir: str = "optimized_models",
        registry_storage_root: str = "models_storage",
    ):
        self.quantizer = ModelQuantizer(output_dir=output_dir)
        self.registry = ModelRegistry(storage_root=registry_storage_root)

    def optimize_and_register(
        self,
        raw_onnx_path: str,
        model_name: str,
        version_tag: str,
        target_accelerators: List[str] = ["wasm", "webgl", "cpu"],
    ) -> Dict[str, Any]:
        """
        Executes end-to-end optimization:
        1. Produces dynamic INT8 artifact.
        2. Benchmarks against FP32 baseline.
        3. Generates signed manifest with SHA-256 and registers in ModelRegistry.
        """
        logger.info(f"[Pipeline] Starting optimization pipeline for {model_name}:{version_tag}...")

        # 1. Quantization
        int8_artifact_path = self.quantizer.quantize_dynamic_int8(raw_onnx_path)

        # 2. Benchmarking & Quality Assurance
        bench_results = OptimizationBenchmark.run_benchmark(
            fp32_model_path=raw_onnx_path,
            quantized_model_path=int8_artifact_path,
            iterations=30,
        )

        if not bench_results["quality_gate_passed"]:
            logger.warning(
                f"[Pipeline] Quality gate check failed (Cosine Sim: {bench_results['cosine_similarity']:.4f}). "
                "Manual review advised."
            )

        # 3. Model Registry Publishing
        manifest = self.registry.register_model_version(
            model_name=model_name,
            version=version_tag,
            artifact_path=int8_artifact_path,
            input_shape=[1, 3, 640, 640],
            confidence_threshold=0.5,
            iou_threshold=0.45,
            target_accelerators=target_accelerators,
        )

        manifest["benchmark_metrics"] = bench_results
        return manifest
