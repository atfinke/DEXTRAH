"""
ONNX Export Utilities for DEXTRAH Models

This module provides functions to export PyTorch models to ONNX format
and validate the exported models for accuracy.
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging

from .onnx_models import (
    ONNXStereoEncoder,
    ONNXPolicyMLP,
    ONNXLSTMSequential,
    ONNXRunningMeanStd
)

logger = logging.getLogger(__name__)


class ONNXCompatiblePolicy(torch.nn.Module):
    """
    ONNX-compatible full policy network combining:
    - Stereo vision encoder
    - Proprioceptive MLP
    - LSTM for temporal modeling
    - Action output heads
    """
    def __init__(
        self,
        backbone="resnet",
        img_height=240,
        img_width=320,
        num_proprio_obs=159,
        num_actions=11,
        mlp_units=[512, 512, 256],
        rnn_units=512,
        use_rnn=True
    ):
        super().__init__()

        self.num_proprio_obs = num_proprio_obs
        self.num_actions = num_actions
        self.use_rnn = use_rnn

        # Stereo vision encoder
        self.stereo_encoder = ONNXStereoEncoder(
            backbone=backbone,
            img_height=img_height,
            img_width=img_width,
            n_embd=None,  # Will use default for backbone
            n_head=4
        )

        # Input: proprioception (159) + visual features (64) = 223
        mlp_input_size = num_proprio_obs + 64

        # Policy MLP
        self.actor_mlp = ONNXPolicyMLP(
            input_size=mlp_input_size,
            units=mlp_units,
            activation='elu'
        )

        # LSTM for temporal modeling
        if use_rnn:
            self.rnn = ONNXLSTMSequential(
                input_size=mlp_units[-1],
                hidden_size=rnn_units,
                num_layers=1
            )
            action_input_size = rnn_units
        else:
            action_input_size = mlp_units[-1]

        # Action outputs (continuous control)
        self.mu = torch.nn.Linear(action_input_size, num_actions)
        self.sigma = torch.nn.Parameter(torch.zeros(num_actions))

        # Value function
        self.value = torch.nn.Linear(action_input_size, 1)

    def forward(
        self,
        proprioception,
        img_left,
        img_right,
        hidden_h=None,
        hidden_c=None
    ):
        """
        Forward pass for ONNX export.

        Args:
            proprioception: (B, num_proprio_obs)
            img_left: (B, 3, H, W)
            img_right: (B, 3, H, W)
            hidden_h: (1, B, rnn_units) - LSTM hidden state (optional)
            hidden_c: (1, B, rnn_units) - LSTM cell state (optional)

        Returns:
            mu: (B, num_actions) - action means
            sigma: (B, num_actions) - action std devs
            value: (B, 1) - value estimate
            hidden_h_next: (1, B, rnn_units) - next LSTM hidden state
            hidden_c_next: (1, B, rnn_units) - next LSTM cell state
        """
        batch_size = proprioception.shape[0]

        # Encode stereo images
        # Stack left and right images: (2*B, 3, H, W)
        imgs = torch.cat([img_left, img_right], dim=0)
        visual_features = self.stereo_encoder(imgs)  # (B, 64)

        # Concatenate proprioception and visual features
        obs = torch.cat([proprioception, visual_features], dim=-1)

        # MLP processing
        mlp_out = self.actor_mlp(obs)  # (B, mlp_units[-1])

        # LSTM processing
        if self.use_rnn:
            # Reshape for LSTM: (B, 1, features) for single timestep
            rnn_input = mlp_out.unsqueeze(1)

            if hidden_h is not None and hidden_c is not None:
                hidden_state = (hidden_h, hidden_c)
            else:
                hidden_state = None

            rnn_output, (hidden_h_next, hidden_c_next) = self.rnn(rnn_input, hidden_state)
            rnn_output = rnn_output.squeeze(1)  # (B, rnn_units)
        else:
            rnn_output = mlp_out
            hidden_h_next = torch.zeros(1, batch_size, 1)  # Dummy outputs
            hidden_c_next = torch.zeros(1, batch_size, 1)

        # Action outputs
        mu = self.mu(rnn_output)
        sigma = torch.ones_like(mu) * torch.exp(self.sigma)

        # Value output
        value = self.value(rnn_output)

        return mu, sigma, value, hidden_h_next, hidden_c_next


def export_stereo_encoder_to_onnx(
    model: ONNXStereoEncoder,
    output_path: Union[str, Path],
    img_height: int = 240,
    img_width: int = 320,
    batch_size: int = 1,
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> str:
    """
    Export stereo encoder to ONNX format.

    Args:
        model: ONNXStereoEncoder instance
        output_path: Path to save ONNX model
        img_height: Input image height
        img_width: Input image width
        batch_size: Batch size for export
        opset_version: ONNX opset version (14 recommended for QNN compatibility)
        dynamic_axes: Whether to use dynamic batch size

    Returns:
        Path to saved ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy inputs: (2*B, 3, H, W) - stacked stereo pair
    dummy_imgs = torch.randn(2 * batch_size, 3, img_height, img_width)

    # Define input/output names
    input_names = ["stereo_images"]
    output_names = ["visual_features"]

    # Define dynamic axes if requested
    if dynamic_axes:
        dynamic_axes_dict = {
            "stereo_images": {0: "batch_size"},  # 2*B
            "visual_features": {0: "batch_size"}  # B
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    logger.info(f"Exporting stereo encoder to {output_path}")
    torch.onnx.export(
        model,
        (dummy_imgs,),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"✓ ONNX model validated: {output_path}")

    return str(output_path)


def export_policy_to_onnx(
    model: ONNXCompatiblePolicy,
    output_path: Union[str, Path],
    img_height: int = 240,
    img_width: int = 320,
    num_proprio_obs: int = 159,
    batch_size: int = 1,
    opset_version: int = 14,
    dynamic_axes: bool = True,
    include_rnn: bool = True
) -> str:
    """
    Export full policy to ONNX format.

    Args:
        model: ONNXCompatiblePolicy instance
        output_path: Path to save ONNX model
        img_height: Input image height
        img_width: Input image width
        num_proprio_obs: Number of proprioceptive observations
        batch_size: Batch size for export
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic batch size
        include_rnn: Whether to include LSTM state I/O

    Returns:
        Path to saved ONNX model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create dummy inputs
    dummy_proprio = torch.randn(batch_size, num_proprio_obs)
    dummy_img_left = torch.randn(batch_size, 3, img_height, img_width)
    dummy_img_right = torch.randn(batch_size, 3, img_height, img_width)

    input_names = ["proprioception", "img_left", "img_right"]
    output_names = ["mu", "sigma", "value"]

    if include_rnn and model.use_rnn:
        dummy_h = torch.zeros(1, batch_size, model.rnn.hidden_size)
        dummy_c = torch.zeros(1, batch_size, model.rnn.hidden_size)
        dummy_inputs = (dummy_proprio, dummy_img_left, dummy_img_right, dummy_h, dummy_c)
        input_names.extend(["hidden_h", "hidden_c"])
        output_names.extend(["hidden_h_next", "hidden_c_next"])
    else:
        dummy_inputs = (dummy_proprio, dummy_img_left, dummy_img_right)

    # Define dynamic axes
    if dynamic_axes:
        dynamic_axes_dict = {
            "proprioception": {0: "batch_size"},
            "img_left": {0: "batch_size"},
            "img_right": {0: "batch_size"},
            "mu": {0: "batch_size"},
            "sigma": {0: "batch_size"},
            "value": {0: "batch_size"},
        }
        if include_rnn and model.use_rnn:
            dynamic_axes_dict.update({
                "hidden_h": {1: "batch_size"},
                "hidden_c": {1: "batch_size"},
                "hidden_h_next": {1: "batch_size"},
                "hidden_c_next": {1: "batch_size"},
            })
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    logger.info(f"Exporting policy to {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=False
        )

    # Verify ONNX model
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info(f"✓ ONNX model validated: {output_path}")

    return str(output_path)


def export_model_to_onnx(
    pytorch_model,
    output_path: Union[str, Path],
    model_type: str = "policy",
    **kwargs
) -> str:
    """
    Main export function that routes to appropriate exporter.

    Args:
        pytorch_model: PyTorch model to export
        output_path: Path to save ONNX model
        model_type: Type of model ("encoder" or "policy")
        **kwargs: Additional arguments for specific exporters

    Returns:
        Path to saved ONNX model
    """
    if model_type == "encoder":
        return export_stereo_encoder_to_onnx(pytorch_model, output_path, **kwargs)
    elif model_type == "policy":
        return export_policy_to_onnx(pytorch_model, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def validate_onnx_export(
    pytorch_model,
    onnx_path: Union[str, Path],
    num_tests: int = 10,
    tolerance: float = 1e-4,
    img_height: int = 240,
    img_width: int = 320,
    num_proprio_obs: int = 159,
    model_type: str = "policy"
) -> Dict[str, float]:
    """
    Validate ONNX export by comparing outputs with PyTorch model.

    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_tests: Number of test cases
        tolerance: Numerical tolerance for comparison
        img_height: Image height
        img_width: Image width
        num_proprio_obs: Number of proprioceptive observations
        model_type: Type of model ("encoder" or "policy")

    Returns:
        Dictionary with validation metrics
    """
    logger.info(f"Validating ONNX export: {onnx_path}")

    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))

    pytorch_model.eval()

    max_error = 0.0
    mean_error = 0.0
    num_passed = 0

    with torch.no_grad():
        for i in range(num_tests):
            if model_type == "encoder":
                # Test stereo encoder
                dummy_imgs = torch.randn(2, 3, img_height, img_width)

                # PyTorch inference
                pt_output = pytorch_model(dummy_imgs).cpu().numpy()

                # ONNX inference
                ort_inputs = {"stereo_images": dummy_imgs.cpu().numpy()}
                ort_output = ort_session.run(None, ort_inputs)[0]

                # Compare
                error = np.abs(pt_output - ort_output).max()
                mean_error += np.abs(pt_output - ort_output).mean()

            elif model_type == "policy":
                # Test full policy
                dummy_proprio = torch.randn(1, num_proprio_obs)
                dummy_img_left = torch.randn(1, 3, img_height, img_width)
                dummy_img_right = torch.randn(1, 3, img_height, img_width)

                # PyTorch inference
                if pytorch_model.use_rnn:
                    dummy_h = torch.zeros(1, 1, pytorch_model.rnn.hidden_size)
                    dummy_c = torch.zeros(1, 1, pytorch_model.rnn.hidden_size)
                    pt_outputs = pytorch_model(
                        dummy_proprio, dummy_img_left, dummy_img_right, dummy_h, dummy_c
                    )
                    pt_mu, pt_sigma, pt_value = [x.cpu().numpy() for x in pt_outputs[:3]]

                    # ONNX inference
                    ort_inputs = {
                        "proprioception": dummy_proprio.cpu().numpy(),
                        "img_left": dummy_img_left.cpu().numpy(),
                        "img_right": dummy_img_right.cpu().numpy(),
                        "hidden_h": dummy_h.cpu().numpy(),
                        "hidden_c": dummy_c.cpu().numpy()
                    }
                else:
                    pt_outputs = pytorch_model(dummy_proprio, dummy_img_left, dummy_img_right)
                    pt_mu, pt_sigma, pt_value = [x.cpu().numpy() for x in pt_outputs[:3]]

                    ort_inputs = {
                        "proprioception": dummy_proprio.cpu().numpy(),
                        "img_left": dummy_img_left.cpu().numpy(),
                        "img_right": dummy_img_right.cpu().numpy()
                    }

                ort_outputs = ort_session.run(None, ort_inputs)
                ort_mu, ort_sigma, ort_value = ort_outputs[:3]

                # Compare all outputs
                error = max(
                    np.abs(pt_mu - ort_mu).max(),
                    np.abs(pt_sigma - ort_sigma).max(),
                    np.abs(pt_value - ort_value).max()
                )
                mean_error += (
                    np.abs(pt_mu - ort_mu).mean() +
                    np.abs(pt_sigma - ort_sigma).mean() +
                    np.abs(pt_value - ort_value).mean()
                ) / 3

            max_error = max(max_error, error)

            if error < tolerance:
                num_passed += 1

    mean_error /= num_tests
    pass_rate = num_passed / num_tests

    results = {
        "max_error": float(max_error),
        "mean_error": float(mean_error),
        "pass_rate": float(pass_rate),
        "tolerance": float(tolerance),
        "num_tests": num_tests
    }

    logger.info(f"Validation Results:")
    logger.info(f"  Max Error: {max_error:.6e}")
    logger.info(f"  Mean Error: {mean_error:.6e}")
    logger.info(f"  Pass Rate: {pass_rate*100:.1f}%")

    if pass_rate == 1.0:
        logger.info(f"✓ All tests passed!")
    else:
        logger.warning(f"⚠ {num_tests - num_passed} tests failed")

    return results
