"""
Quantization and Optimization for QNN Models.

This module provides utilities for quantizing and optimizing models
for deployment on Qualcomm Hexagon NPU.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class CalibrationDataGenerator:
    """
    Generate calibration data for quantization.
    """

    def __init__(self, data_loader, num_calibration_batches=100):
        """
        Args:
            data_loader: PyTorch DataLoader for calibration data
            num_calibration_batches: Number of batches to use for calibration
        """
        self.data_loader = data_loader
        self.num_calibration_batches = num_calibration_batches

    def generate(self, output_dir: str, input_name: str = "image") -> str:
        """
        Generate calibration data files.

        Args:
            output_dir: Directory to save calibration data
            input_name: Name of the input tensor

        Returns:
            Path to calibration input list file
        """
        os.makedirs(output_dir, exist_ok=True)

        input_list_path = os.path.join(output_dir, "calibration_input_list.txt")
        input_files = []

        for i, batch in enumerate(self.data_loader):
            if i >= self.num_calibration_batches:
                break

            # Assume batch is a tuple of (images, ...)
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            # Save each image in the batch
            for j in range(images.shape[0]):
                image = images[j].numpy()
                filename = f"calib_{i:04d}_{j:03d}.raw"
                filepath = os.path.join(output_dir, filename)

                # Save as raw binary
                image.astype(np.float32).tofile(filepath)
                input_files.append(f"{input_name}:={filepath}")

        # Write input list file
        with open(input_list_path, 'w') as f:
            for input_file in input_files:
                f.write(f"{input_file}\n")

        print(f"Generated {len(input_files)} calibration samples")
        print(f"Calibration input list: {input_list_path}")

        return input_list_path


class QuantizationConfigGenerator:
    """
    Generate quantization configuration for QNN.
    """

    @staticmethod
    def create_config(
        output_path: str,
        activation_bitwidth: int = 8,
        weight_bitwidth: int = 8,
        bias_bitwidth: int = 32,
        use_symmetric: bool = True,
        use_per_channel: bool = True,
        algorithms: Optional[List[str]] = None
    ) -> str:
        """
        Create quantization configuration file.

        Args:
            output_path: Path to save config
            activation_bitwidth: Bitwidth for activations (8 or 16)
            weight_bitwidth: Bitwidth for weights (8 or 16)
            bias_bitwidth: Bitwidth for biases (usually 32)
            use_symmetric: Use symmetric quantization
            use_per_channel: Use per-channel quantization for weights
            algorithms: List of quantization algorithms to try

        Returns:
            Path to config file
        """
        if algorithms is None:
            algorithms = ["minmax", "entropy", "mse"]

        config = {
            "activation_bitwidth": activation_bitwidth,
            "weight_bitwidth": weight_bitwidth,
            "bias_bitwidth": bias_bitwidth,
            "use_symmetric_quantization": use_symmetric,
            "use_per_channel_quantization": use_per_channel,
            "quantization_algorithms": algorithms,
            "use_enhanced_quantizer": True,
            "use_native_input_files": True,
            "use_native_output_files": True
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Quantization config saved to {output_path}")
        return output_path


class ModelOptimizer:
    """
    Optimize models for QNN deployment.
    """

    @staticmethod
    def optimize_onnx_for_qnn(
        onnx_model_path: str,
        output_path: str,
        optimizations: Optional[List[str]] = None
    ) -> str:
        """
        Optimize ONNX model for QNN conversion.

        Args:
            onnx_model_path: Path to input ONNX model
            output_path: Path to save optimized model
            optimizations: List of optimization passes to apply

        Returns:
            Path to optimized model
        """
        import onnx
        from onnx import optimizer

        if optimizations is None:
            # Standard optimizations for QNN
            optimizations = [
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_transpose',
                'eliminate_nop_pad',
                'extract_constant_to_initializer',
                'fuse_add_bias_into_conv',
                'fuse_bn_into_conv',
                'fuse_consecutive_concats',
                'fuse_consecutive_reduce_unsqueeze',
                'fuse_consecutive_squeezes',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_pad_into_conv',
                'fuse_transpose_into_gemm'
            ]

        # Load model
        model = onnx.load(onnx_model_path)

        # Apply optimizations
        optimized_model = optimizer.optimize(model, optimizations)

        # Save optimized model
        onnx.save(optimized_model, output_path)

        print(f"Optimized ONNX model saved to {output_path}")
        return output_path

    @staticmethod
    def analyze_model_performance(
        model_path: str,
        input_shape: Tuple[int, ...],
        backend: str = "cpu"
    ) -> Dict:
        """
        Analyze model performance characteristics.

        Args:
            model_path: Path to model (ONNX or QNN)
            input_shape: Input tensor shape
            backend: Backend to use for analysis

        Returns:
            Dict with performance metrics
        """
        import onnxruntime as ort

        # Create session
        providers = ['CPUExecutionProvider']
        if backend == "cuda":
            providers.insert(0, 'CUDAExecutionProvider')

        session = ort.InferenceSession(model_path, providers=providers)

        # Get model info
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(10):
            session.run(output_names, {input_name: dummy_input})

        # Benchmark
        import time
        num_runs = 100
        start = time.time()
        for _ in range(num_runs):
            session.run(output_names, {input_name: dummy_input})
        end = time.time()

        avg_time = (end - start) / num_runs
        fps = 1.0 / avg_time

        metrics = {
            "average_inference_time_ms": avg_time * 1000,
            "fps": fps,
            "input_shape": input_shape,
            "backend": backend,
            "num_runs": num_runs
        }

        print("\nModel Performance Metrics:")
        print(f"  Average inference time: {avg_time * 1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        return metrics


class AccuracyValidator:
    """
    Validate quantized model accuracy.
    """

    @staticmethod
    def compare_outputs(
        pytorch_model,
        qnn_outputs: np.ndarray,
        test_inputs: torch.Tensor,
        tolerance: float = 0.1
    ) -> Dict:
        """
        Compare outputs between PyTorch and QNN models.

        Args:
            pytorch_model: Original PyTorch model
            qnn_outputs: Outputs from QNN model
            test_inputs: Test input tensors
            tolerance: Acceptable error tolerance (0-1)

        Returns:
            Dict with accuracy metrics
        """
        pytorch_model.eval()

        with torch.no_grad():
            pytorch_outputs = pytorch_model(test_inputs).cpu().numpy()

        # Compute metrics
        mae = np.mean(np.abs(pytorch_outputs - qnn_outputs))
        mse = np.mean((pytorch_outputs - qnn_outputs) ** 2)
        max_error = np.max(np.abs(pytorch_outputs - qnn_outputs))

        # Compute relative error
        relative_error = mae / (np.mean(np.abs(pytorch_outputs)) + 1e-8)

        # Check if within tolerance
        within_tolerance = relative_error < tolerance

        metrics = {
            "mae": float(mae),
            "mse": float(mse),
            "max_error": float(max_error),
            "relative_error": float(relative_error),
            "within_tolerance": bool(within_tolerance),
            "tolerance_threshold": tolerance
        }

        print("\nAccuracy Metrics:")
        print(f"  MAE: {mae:.6f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  Max Error: {max_error:.6f}")
        print(f"  Relative Error: {relative_error * 100:.2f}%")
        print(f"  Within Tolerance ({tolerance * 100}%): {within_tolerance}")

        return metrics


def run_complete_quantization_pipeline(
    pytorch_model,
    onnx_model_path: str,
    calibration_data_loader,
    output_dir: str,
    qnn_sdk_root: str = "/opt/qcom/aistack/qnn"
) -> Dict:
    """
    Run complete quantization pipeline for a model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_model_path: Path to ONNX model
        calibration_data_loader: DataLoader for calibration
        output_dir: Output directory
        qnn_sdk_root: QNN SDK root path

    Returns:
        Dict with results and metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Step 1: Generate calibration data
    print("\n" + "=" * 80)
    print("Step 1: Generating calibration data...")
    print("=" * 80)

    calib_gen = CalibrationDataGenerator(
        calibration_data_loader,
        num_calibration_batches=100
    )
    calib_dir = os.path.join(output_dir, "calibration_data")
    calib_input_list = calib_gen.generate(calib_dir)
    results["calibration_input_list"] = calib_input_list

    # Step 2: Create quantization config
    print("\n" + "=" * 80)
    print("Step 2: Creating quantization configuration...")
    print("=" * 80)

    quant_config = os.path.join(output_dir, "quantization_config.json")
    QuantizationConfigGenerator.create_config(
        quant_config,
        activation_bitwidth=8,
        weight_bitwidth=8
    )
    results["quantization_config"] = quant_config

    # Step 3: Optimize ONNX model
    print("\n" + "=" * 80)
    print("Step 3: Optimizing ONNX model...")
    print("=" * 80)

    optimized_onnx = os.path.join(output_dir, "model_optimized.onnx")
    ModelOptimizer.optimize_onnx_for_qnn(
        onnx_model_path,
        optimized_onnx
    )
    results["optimized_onnx"] = optimized_onnx

    # Step 4: Convert to QNN
    print("\n" + "=" * 80)
    print("Step 4: Converting to QNN format...")
    print("=" * 80)

    from convert_to_qnn import QNNConverter

    converter = QNNConverter(qnn_sdk_root)
    qnn_output = converter.convert_onnx_to_qnn(
        onnx_model_path=optimized_onnx,
        output_dir=os.path.join(output_dir, "qnn_model"),
        input_list=[("image", [1, 3, 240, 320])],
        quantization_config=quant_config
    )
    results["qnn_model"] = qnn_output

    # Step 5: Analyze performance
    print("\n" + "=" * 80)
    print("Step 5: Analyzing model performance...")
    print("=" * 80)

    perf_metrics = ModelOptimizer.analyze_model_performance(
        optimized_onnx,
        input_shape=(1, 3, 240, 320)
    )
    results["performance_metrics"] = perf_metrics

    # Save results
    results_file = os.path.join(output_dir, "quantization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("Quantization pipeline complete!")
    print(f"Results saved to {results_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    print("Quantization and Optimization Utilities for QNN")
    print("=" * 80)
    print("\nAvailable tools:")
    print("  - CalibrationDataGenerator: Generate calibration data")
    print("  - QuantizationConfigGenerator: Create quantization configs")
    print("  - ModelOptimizer: Optimize models for QNN")
    print("  - AccuracyValidator: Validate quantized model accuracy")
    print("\nUsage: Import this module and use the provided classes")
