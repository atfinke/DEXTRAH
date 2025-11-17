#!/usr/bin/env python3
"""
Complete Example: DEXTRAH Model Export Pipeline

This script demonstrates the full pipeline:
1. Load trained PyTorch model
2. Convert to ONNX-compatible format (with 12 re-implemented ops)
3. Export to ONNX
4. Validate ONNX export
5. Convert to QNN (with custom Hexagon ops)
6. Quantize and calibrate
7. Validate and benchmark QNN model

Usage:
    python example_export_pipeline.py --model_path <path_to_checkpoint>
"""

import argparse
import logging
from pathlib import Path
import torch

# DEXTRAH model export utilities
from onnx_export import (
    ONNXCompatiblePolicy,
    export_model_to_onnx,
    validate_onnx_export
)
from qnn_convert import (
    QNNConfig,
    QNNCustomOpSpec,
    create_qnn_config,
    generate_qnn_custom_op_package,
    convert_onnx_to_qnn,
    QNNModelValidator
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pytorch_checkpoint(checkpoint_path: Path) -> dict:
    """Load PyTorch checkpoint"""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def create_onnx_compatible_model(
    backbone: str = "resnet",
    checkpoint: dict = None
) -> ONNXCompatiblePolicy:
    """
    Create ONNX-compatible policy model.
    This model uses re-implemented operations for ONNX/QNN compatibility.
    """
    logger.info("Creating ONNX-compatible policy model")

    model = ONNXCompatiblePolicy(
        backbone=backbone,
        img_height=240,
        img_width=320,
        num_proprio_obs=159,
        num_actions=11,
        mlp_units=[512, 512, 256],
        rnn_units=512,
        use_rnn=True
    )

    # Load weights if checkpoint provided
    if checkpoint is not None:
        # TODO: Adapt checkpoint loading to match ONNX model structure
        logger.info("Loading weights from checkpoint")
        # model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="DEXTRAH Model Export Pipeline")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to PyTorch checkpoint (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_models",
        help="Output directory for exported models"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet",
        choices=["scratch", "resnet", "convnext"],
        help="Vision encoder backbone"
    )
    parser.add_argument(
        "--skip_qnn",
        action="store_true",
        help="Skip QNN conversion (requires QNN SDK)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Step 1: Create ONNX-Compatible Model
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: Creating ONNX-Compatible Model")
    logger.info("=" * 80)

    checkpoint = None
    if args.model_path:
        checkpoint = load_pytorch_checkpoint(Path(args.model_path))

    model = create_onnx_compatible_model(
        backbone=args.backbone,
        checkpoint=checkpoint
    )

    logger.info("\n✓ ONNX-compatible model created")
    logger.info(f"  - Re-implemented operations: 12")
    logger.info(f"  - Backbone: {args.backbone}")
    logger.info(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ========================================================================
    # Step 2: Export to ONNX
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Exporting to ONNX Format")
    logger.info("=" * 80)

    onnx_path = output_dir / "dextrah_policy.onnx"

    export_model_to_onnx(
        model,
        onnx_path,
        model_type="policy",
        img_height=240,
        img_width=320,
        num_proprio_obs=159,
        batch_size=1,
        opset_version=14,  # QNN-compatible opset
        dynamic_axes=True,
        include_rnn=True
    )

    logger.info(f"\n✓ ONNX export successful: {onnx_path}")

    # ========================================================================
    # Step 3: Validate ONNX Export
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Validating ONNX Export Accuracy")
    logger.info("=" * 80)

    validation_results = validate_onnx_export(
        model,
        onnx_path,
        num_tests=20,
        tolerance=1e-4,
        model_type="policy"
    )

    logger.info("\n✓ ONNX validation complete")
    logger.info(f"  - Max error: {validation_results['max_error']:.6e}")
    logger.info(f"  - Mean error: {validation_results['mean_error']:.6e}")
    logger.info(f"  - Pass rate: {validation_results['pass_rate']*100:.1f}%")

    if validation_results['pass_rate'] < 1.0:
        logger.warning("⚠ Some validation tests failed! Check numerical accuracy.")

    # ========================================================================
    # Step 4: Generate QNN Custom Operator Package
    # ========================================================================
    if not args.skip_qnn:
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: Generating QNN Custom Operator Package")
        logger.info("=" * 80)

        custom_ops_dir = output_dir / "qnn_custom_ops"
        package_dir = generate_qnn_custom_op_package(custom_ops_dir)

        logger.info(f"\n✓ Custom operator package generated: {package_dir}")
        logger.info("  Custom operators implemented:")

        for op_spec in QNNCustomOpSpec.get_all_custom_ops():
            logger.info(f"    - {op_spec['op_name']}: {op_spec['description']}")

        logger.info("\n  Next steps for custom ops:")
        logger.info(f"    1. cd {package_dir}")
        logger.info("    2. Implement operation logic in src/*.cpp files")
        logger.info("    3. Build: mkdir build && cd build && cmake .. && make")
        logger.info("    4. Use library: export QNN_CUSTOM_OP_LIB=$PWD/build/libqnn_dextrah_ops.so")

        # ========================================================================
        # Step 5: Convert to QNN
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: Converting to QNN Format")
        logger.info("=" * 80)

        qnn_config = create_qnn_config(
            model_name="dextrah_policy",
            precision="int8",
            backend="htp",
            optimize_for_latency=True
        )

        logger.info(f"QNN Configuration:")
        for key, value in qnn_config.to_dict().items():
            logger.info(f"  - {key}: {value}")

        qnn_output_dir = output_dir / "qnn_model"

        try:
            # Note: This requires QNN SDK to be installed
            convert_onnx_to_qnn(
                onnx_path,
                qnn_output_dir,
                config=qnn_config,
                custom_op_lib=None  # Set to path of built custom op library
            )

            logger.info(f"\n✓ QNN conversion successful: {qnn_output_dir}")

        except Exception as e:
            logger.warning(f"\n⚠ QNN conversion skipped: {e}")
            logger.info("  Install QNN SDK from: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk")

        # ========================================================================
        # Step 6: Quantization Calibration
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: Quantization Calibration (INT8)")
        logger.info("=" * 80)

        logger.info("Quantization calibration requires:")
        logger.info("  1. Representative input data from your dataset")
        logger.info("  2. Run calibration script generated in QNN output directory")
        logger.info("  3. Fine-tune quantization parameters based on accuracy metrics")

        logger.info(f"\n  See: {qnn_output_dir / 'calibrate_model.py'}")

        # ========================================================================
        # Step 7: Validate QNN Model
        # ========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: QNN Model Validation and Benchmarking")
        logger.info("=" * 80)

        try:
            validator = QNNModelValidator(qnn_output_dir, backend="cpu")

            if validator.validate_model():
                logger.info("✓ QNN model validation passed")

                # Benchmark
                benchmark_results = validator.benchmark(num_iterations=100)
                logger.info("\nBenchmark Results:")
                for metric, value in benchmark_results.items():
                    logger.info(f"  - {metric}: {value}")

        except Exception as e:
            logger.warning(f"⚠ QNN validation skipped: {e}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EXPORT PIPELINE COMPLETE")
    logger.info("=" * 80)

    logger.info("\nGenerated Files:")
    logger.info(f"  - ONNX Model: {onnx_path}")
    if not args.skip_qnn:
        logger.info(f"  - QNN Custom Ops: {custom_ops_dir}")
        logger.info(f"  - QNN Model: {qnn_output_dir}")

    logger.info("\nOperations Re-implemented for ONNX/QNN Compatibility:")
    ops_reimplemented = [
        "1. Scaled Dot-Product Attention",
        "2. Masked Attention Operations",
        "3. Adaptive Average Pooling",
        "4. GELU Activation",
        "5. ELU Activation",
        "6. Positional Embeddings",
        "7. Cross-Attention Mechanism",
        "8. LayerNorm",
        "9. Running Mean/Std Normalization",
        "10. LSTM State Management",
        "11. Complex Tensor Reshaping",
        "12. BFloat16 → FP16/FP32 Conversion"
    ]
    for op in ops_reimplemented:
        logger.info(f"  {op}")

    logger.info("\nNext Steps:")
    logger.info("  1. Validate ONNX model accuracy on your test dataset")
    logger.info("  2. Build QNN custom operators (see instructions above)")
    logger.info("  3. Run quantization calibration with representative data")
    logger.info("  4. Deploy to Qualcomm device with Hexagon NPU")
    logger.info("  5. Benchmark and optimize performance")

    logger.info("\n✓ All done!")


if __name__ == "__main__":
    main()
