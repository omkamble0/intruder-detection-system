"""
GOLIATH Model Optimization Pipeline - Quantization & Compression Engine
Supports FP16 half-precision conversion, dynamic INT8 quantization, and graph optimization.
"""

import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("goliath.optimization.quantizer")

class ModelQuantizer:
    """
    Automated optimizer converting FP32 ONNX models into high-performance INT8 / FP16 variants.
    """

    def __init__(self, output_dir: str = "optimized_models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def quantize_dynamic_int8(self, input_model_path: str, output_model_path: Optional[str] = None) -> str:
        """
        Applies dynamic INT8 quantization (weights INT8, activations quantized on-the-fly).
        Ideal for WASM / CPU / Edge runtimes with minimal accuracy loss.
        """
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType

        if not os.path.exists(input_model_path):
            raise FileNotFoundError(f"Input model not found: {input_model_path}")

        if output_model_path is None:
            base_name = os.path.splitext(os.path.basename(input_model_path))[0]
            output_model_path = os.path.join(self.output_dir, f"{base_name}_int8.onnx")

        logger.info(f"[Quantizer] Starting Dynamic INT8 Quantization on {input_model_path}...")
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
        )

        orig_size = os.path.getsize(input_model_path) / (1024 * 1024)
        quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
        logger.info(
            f"[Quantizer] INT8 Quantization completed: {orig_size:.2f}MB -> {quant_size:.2f}MB "
            f"({(1.0 - quant_size / orig_size) * 100:.1f}% reduction)"
        )
        return output_model_path

    def convert_fp16(self, input_model_path: str, output_model_path: Optional[str] = None) -> str:
        """
        Converts model parameters to IEEE 754 Float16 for WebGPU / CUDA / TensorRT accelerators.
        """
        import onnx
        from onnxconverter_common import float16

        if not os.path.exists(input_model_path):
            raise FileNotFoundError(f"Input model not found: {input_model_path}")

        if output_model_path is None:
            base_name = os.path.splitext(os.path.basename(input_model_path))[0]
            output_model_path = os.path.join(self.output_dir, f"{base_name}_fp16.onnx")

        logger.info(f"[Quantizer] Converting {input_model_path} to FP16 half-precision...")
        model = onnx.load(input_model_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, output_model_path)

        orig_size = os.path.getsize(input_model_path) / (1024 * 1024)
        fp16_size = os.path.getsize(output_model_path) / (1024 * 1024)
        logger.info(
            f"[Quantizer] FP16 Conversion completed: {orig_size:.2f}MB -> {fp16_size:.2f}MB "
            f"({(1.0 - fp16_size / orig_size) * 100:.1f}% reduction)"
        )
        return output_model_path
