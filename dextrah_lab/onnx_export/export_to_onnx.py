"""
ONNX Export Utilities for DEXTRAH Models.

This module provides utilities to export PyTorch models to ONNX format
and validate the accuracy of the exported models.
"""

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from distillation.encoders_onnx import (
    MonoEncoderONNX,
    StereoEncoderONNX,
    CustomCNNONNX,
    ResnetEncoderONNX,
    ConvNextEncoderONNX,
)


class ONNXExporter:
    """
    Utility class for exporting PyTorch models to ONNX format.
    """

    def __init__(self, opset_version=17):
        """
        Args:
            opset_version: ONNX opset version to use (default: 17)
        """
        self.opset_version = opset_version

    def export_model(
        self,
        model,
        dummy_input,
        output_path,
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        verbose=False
    ):
        """
        Export a PyTorch model to ONNX format.

        Args:
            model: PyTorch model to export
            dummy_input: Example input tensor(s) for tracing
            output_path: Path to save ONNX model
            input_names: List of input names
            output_names: List of output names
            dynamic_axes: Dict specifying dynamic axes
            verbose: Whether to print export details
        """
        model.eval()

        # Default input/output names
        if input_names is None:
            input_names = ['input']
        if output_names is None:
            output_names = ['output']

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )

        print(f"Model exported to {output_path}")

        # Verify the exported model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model is valid!")
        except Exception as e:
            print(f"Error validating ONNX model: {e}")
            raise

        return output_path

    def validate_onnx_model(
        self,
        pytorch_model,
        onnx_path,
        test_inputs,
        rtol=1e-3,
        atol=1e-5
    ):
        """
        Validate ONNX model accuracy against PyTorch model.

        Args:
            pytorch_model: Original PyTorch model
            onnx_path: Path to ONNX model
            test_inputs: Test input tensor(s)
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            bool: True if validation passes
        """
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            if isinstance(test_inputs, (list, tuple)):
                pytorch_output = pytorch_model(*test_inputs)
            else:
                pytorch_output = pytorch_model(test_inputs)

        # Convert to numpy
        if isinstance(pytorch_output, torch.Tensor):
            pytorch_output_np = pytorch_output.cpu().numpy()
        else:
            pytorch_output_np = [o.cpu().numpy() for o in pytorch_output]

        # ONNX inference
        ort_session = ort.InferenceSession(onnx_path)

        # Prepare ONNX inputs
        if isinstance(test_inputs, (list, tuple)):
            ort_inputs = {
                ort_session.get_inputs()[i].name: inp.cpu().numpy()
                for i, inp in enumerate(test_inputs)
            }
        else:
            ort_inputs = {
                ort_session.get_inputs()[0].name: test_inputs.cpu().numpy()
            }

        onnx_output = ort_session.run(None, ort_inputs)

        # Compare outputs
        if isinstance(pytorch_output_np, list):
            for i, (pt_out, onnx_out) in enumerate(zip(pytorch_output_np, onnx_output)):
                if not np.allclose(pt_out, onnx_out, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(pt_out - onnx_out))
                    print(f"Output {i} validation FAILED! Max difference: {max_diff}")
                    return False
                else:
                    max_diff = np.max(np.abs(pt_out - onnx_out))
                    print(f"Output {i} validation PASSED! Max difference: {max_diff}")
        else:
            onnx_out = onnx_output[0]
            if not np.allclose(pytorch_output_np, onnx_out, rtol=rtol, atol=atol):
                max_diff = np.max(np.abs(pytorch_output_np - onnx_out))
                print(f"Output validation FAILED! Max difference: {max_diff}")
                return False
            else:
                max_diff = np.max(np.abs(pytorch_output_np - onnx_out))
                print(f"Output validation PASSED! Max difference: {max_diff}")

        return True


def export_mono_encoder(
    backbone="resnet",
    img_height=240,
    img_width=320,
    n_embd=128,
    n_head=4,
    output_dir="./onnx_models",
    checkpoint_path=None
):
    """
    Export MonoEncoder to ONNX format.

    Args:
        backbone: Backbone type ("scratch", "resnet", "convnext")
        img_height: Input image height
        img_width: Input image width
        n_embd: Embedding dimension
        n_head: Number of attention heads
        output_dir: Directory to save ONNX model
        checkpoint_path: Path to PyTorch checkpoint (optional)

    Returns:
        Path to exported ONNX model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = MonoEncoderONNX(
        backbone=backbone,
        img_height=img_height,
        img_width=img_width,
        n_embd=n_embd,
        n_head=n_head
    )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_height, img_width)

    # Export configuration
    output_path = os.path.join(output_dir, f"mono_encoder_{backbone}.onnx")
    input_names = ['image']
    output_names = ['embedding']
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'embedding': {0: 'batch_size'}
    }

    # Export
    exporter = ONNXExporter(opset_version=17)
    exporter.export_model(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Validate
    test_input = torch.randn(2, 3, img_height, img_width)
    exporter.validate_onnx_model(model, output_path, test_input)

    return output_path


def export_stereo_encoder(
    backbone="resnet",
    img_height=320,
    img_width=256,
    n_embd=128,
    n_head=4,
    output_dir="./onnx_models",
    checkpoint_path=None
):
    """
    Export StereoEncoder to ONNX format.

    Args:
        backbone: Backbone type ("scratch", "resnet", "convnext")
        img_height: Input image height
        img_width: Input image width
        n_embd: Embedding dimension
        n_head: Number of attention heads
        output_dir: Directory to save ONNX model
        checkpoint_path: Path to PyTorch checkpoint (optional)

    Returns:
        Path to exported ONNX model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create model
    model = StereoEncoderONNX(
        backbone=backbone,
        img_height=img_height,
        img_width=img_width,
        n_embd=n_embd,
        n_head=n_head
    )

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # Create dummy input (stacked left and right images)
    dummy_input = torch.randn(2, 3, img_height, img_width)

    # Export configuration
    output_path = os.path.join(output_dir, f"stereo_encoder_{backbone}.onnx")
    input_names = ['stereo_images']
    output_names = ['embedding']
    dynamic_axes = {
        'stereo_images': {0: 'batch_size_x2'},
        'embedding': {0: 'batch_size'}
    }

    # Export
    exporter = ONNXExporter(opset_version=17)
    exporter.export_model(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Validate
    test_input = torch.randn(4, 3, img_height, img_width)  # 2 batches
    exporter.validate_onnx_model(model, output_path, test_input)

    return output_path


def export_cnn_encoder(
    encoder_type="resnet",
    img_height=240,
    img_width=320,
    output_dir="./onnx_models"
):
    """
    Export CNN encoders to ONNX format.

    Args:
        encoder_type: Type of encoder ("scratch", "resnet", "convnext")
        img_height: Input image height
        img_width: Input image width
        output_dir: Directory to save ONNX model

    Returns:
        Path to exported ONNX model
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create model based on type
    if encoder_type == "scratch":
        model = CustomCNNONNX(img_height, img_width)
    elif encoder_type == "resnet":
        model = ResnetEncoderONNX(img_height, img_width)
    elif encoder_type == "convnext":
        model = ConvNextEncoderONNX(img_height, img_width)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_height, img_width)

    # Export configuration
    output_path = os.path.join(output_dir, f"cnn_encoder_{encoder_type}.onnx")
    input_names = ['image']
    output_names = ['features']
    dynamic_axes = {
        'image': {0: 'batch_size'},
        'features': {0: 'batch_size'}
    }

    # Export
    exporter = ONNXExporter(opset_version=17)
    exporter.export_model(
        model,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    # Validate
    test_input = torch.randn(2, 3, img_height, img_width)
    exporter.validate_onnx_model(model, output_path, test_input)

    return output_path


def export_all_encoders(output_dir="./onnx_models"):
    """
    Export all encoder models to ONNX format.

    Args:
        output_dir: Directory to save ONNX models

    Returns:
        List of exported model paths
    """
    exported_models = []

    print("=" * 80)
    print("Exporting MonoEncoder models...")
    print("=" * 80)

    for backbone in ["scratch", "resnet", "convnext"]:
        print(f"\nExporting MonoEncoder with {backbone} backbone...")
        try:
            path = export_mono_encoder(
                backbone=backbone,
                output_dir=output_dir
            )
            exported_models.append(path)
        except Exception as e:
            print(f"Failed to export MonoEncoder ({backbone}): {e}")

    print("\n" + "=" * 80)
    print("Exporting StereoEncoder models...")
    print("=" * 80)

    for backbone in ["scratch", "resnet", "convnext"]:
        print(f"\nExporting StereoEncoder with {backbone} backbone...")
        try:
            path = export_stereo_encoder(
                backbone=backbone,
                output_dir=output_dir
            )
            exported_models.append(path)
        except Exception as e:
            print(f"Failed to export StereoEncoder ({backbone}): {e}")

    print("\n" + "=" * 80)
    print("Exporting CNN Encoder models...")
    print("=" * 80)

    for encoder_type in ["scratch", "resnet", "convnext"]:
        print(f"\nExporting {encoder_type} CNN encoder...")
        try:
            path = export_cnn_encoder(
                encoder_type=encoder_type,
                output_dir=output_dir
            )
            exported_models.append(path)
        except Exception as e:
            print(f"Failed to export CNN encoder ({encoder_type}): {e}")

    print("\n" + "=" * 80)
    print(f"Export complete! {len(exported_models)} models exported.")
    print("=" * 80)

    return exported_models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export DEXTRAH models to ONNX")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_models",
        help="Directory to save ONNX models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mono", "stereo", "cnn", "all"],
        default="all",
        help="Type of model to export"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["scratch", "resnet", "convnext"],
        default="resnet",
        help="Backbone type"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint"
    )

    args = parser.parse_args()

    if args.model_type == "all":
        export_all_encoders(args.output_dir)
    elif args.model_type == "mono":
        export_mono_encoder(
            backbone=args.backbone,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint
        )
    elif args.model_type == "stereo":
        export_stereo_encoder(
            backbone=args.backbone,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint
        )
    elif args.model_type == "cnn":
        export_cnn_encoder(
            encoder_type=args.backbone,
            output_dir=args.output_dir
        )
