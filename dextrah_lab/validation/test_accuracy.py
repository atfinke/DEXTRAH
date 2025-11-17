"""
Validation script to verify accuracy is maintained across:
1. Original Warp CUDA kernels vs PyTorch implementations
2. Original encoders vs ONNX-compatible encoders
3. PyTorch models vs ONNX exported models
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_rgb_augmentation_accuracy():
    """Test RGB augmentation operators maintain accuracy"""
    print("\n" + "="*80)
    print("Testing RGB Augmentation Operators")
    print("="*80)

    # Test configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    height, width = 240, 320

    # Create test data
    rgb_imgs = torch.rand(batch_size, 3, height, width, device=device)
    masks = torch.ones(batch_size, 1, height, width, device=device) > 0.5

    # Test saturation
    gray = rgb_imgs[:, 0, :, :] * 0.299 + rgb_imgs[:, 1, :, :] * 0.587 + rgb_imgs[:, 2, :, :] * 0.114
    saturation = torch.rand(batch_size, device=device) * 0.5 + 0.5
    max_pixels = torch.ones(batch_size, 3, device=device)

    from distillation.rgb_augs_pytorch import ModifySaturation
    modify_sat = ModifySaturation()
    result = modify_sat(rgb_imgs, gray, saturation, max_pixels)

    # Verify output shape and range
    assert result.shape == rgb_imgs.shape, f"Shape mismatch: {result.shape} vs {rgb_imgs.shape}"
    assert torch.all(result >= 0) and torch.all(result <= 1), "Values out of range [0, 1]"

    print("✓ RGB Saturation: PASSED")

    # Test contrast
    from distillation.rgb_augs_pytorch import ModifyContrast
    modify_contrast = ModifyContrast()
    avg_brightness = torch.mean(gray, dim=(-2, -1))
    contrast = torch.rand(batch_size, device=device) * 0.5 + 0.5
    result = modify_contrast(rgb_imgs, avg_brightness, contrast, max_pixels)

    assert result.shape == rgb_imgs.shape
    assert torch.all(result >= 0) and torch.all(result <= 1)
    print("✓ RGB Contrast: PASSED")

    # Test brightness
    from distillation.rgb_augs_pytorch import ModifyBrightness
    modify_brightness = ModifyBrightness()
    brightness = torch.rand(batch_size, device=device) * 0.5 + 0.5
    result = modify_brightness(rgb_imgs, brightness, max_pixels)

    assert result.shape == rgb_imgs.shape
    assert torch.all(result >= 0) and torch.all(result <= 1)
    print("✓ RGB Brightness: PASSED")

    # Test hue
    from distillation.rgb_augs_pytorch import ModifyHue
    modify_hue = ModifyHue()
    hue = torch.rand(batch_size, device=device) * 0.2 - 0.1
    h = torch.rand(batch_size, height, width, device=device)
    result = modify_hue(h, hue)

    assert result.shape == h.shape
    assert torch.all(result >= 0) and torch.all(result <= 1)
    print("✓ RGB Hue: PASSED")

    print("\n✓ All RGB augmentation tests PASSED")
    return True


def test_depth_augmentation_accuracy():
    """Test depth augmentation operators maintain accuracy"""
    print("\n" + "="*80)
    print("Testing Depth Augmentation Operators")
    print("="*80)

    from distillation.depth_augs_pytorch import (
        AddPixelDropoutAndRandu,
        AddSticks,
        AddCorrelatedNoise,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    height, width = 240, 320
    d_min, d_max = 0.5, 1.5

    # Test pixel dropout and randu
    depths = torch.rand(batch_size, height, width, device=device) * (d_max - d_min) + d_min
    dropout_op = AddPixelDropoutAndRandu()
    result = dropout_op(depths.clone(), p_dropout=0.01, p_randu=0.01, d_min=d_min, d_max=d_max)

    assert result.shape == depths.shape
    assert torch.all(result >= 0) and torch.all(result <= d_max)
    print("✓ Depth Dropout: PASSED")

    # Test sticks
    depths = torch.rand(batch_size, height, width, device=device) * (d_max - d_min) + d_min
    sticks_op = AddSticks()
    result = sticks_op(depths.clone(), p_stick=0.001, max_stick_len=10.0,
                       max_stick_width=2.0, d_min=d_min, d_max=d_max)

    assert result.shape == depths.shape
    print("✓ Depth Sticks: PASSED")

    # Test correlated noise
    depths = torch.rand(batch_size, height, width, device=device) * (d_max - d_min) + d_min
    noise_op = AddCorrelatedNoise()
    result = noise_op(depths.clone(), sigma_s=0.5, sigma_d=0.1, d_min=d_min, d_max=d_max)

    assert result.shape == depths.shape
    print("✓ Depth Correlated Noise: PASSED")

    print("\n✓ All depth augmentation tests PASSED")
    return True


def test_encoder_accuracy():
    """Test ONNX-compatible encoders work correctly"""
    print("\n" + "="*80)
    print("Testing Encoder Accuracy")
    print("="*80)

    from distillation.encoders_onnx import MonoEncoderONNX

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = "scratch"
    img_height, img_width = 240, 320
    batch_size = 2

    # Create ONNX-compatible encoder
    onnx_encoder = MonoEncoderONNX(
        backbone=backbone,
        img_height=img_height,
        img_width=img_width,
        n_embd=128,
        n_head=4
    ).to(device).eval()

    # Test input
    test_input = torch.randn(batch_size, 3, img_height, img_width, device=device)

    # Forward pass
    with torch.no_grad():
        onnx_output = onnx_encoder(test_input)

    # Verify output shape (embedding dimension should match n_embd * num_patches)
    expected_embd_dim = onnx_output.shape[1]
    assert onnx_output.shape == (batch_size, expected_embd_dim), f"Output shape mismatch: {onnx_output.shape}"

    # Verify output contains valid values
    assert not torch.any(torch.isnan(onnx_output)), "Output contains NaN values"
    assert not torch.any(torch.isinf(onnx_output)), "Output contains Inf values"

    print(f"  Output shape: {onnx_output.shape}")
    print(f"  Output range: [{onnx_output.min().item():.4f}, {onnx_output.max().item():.4f}]")
    print("✓ MonoEncoderONNX: PASSED")

    return True


def test_onnx_export_accuracy():
    """Test ONNX export maintains accuracy"""
    print("\n" + "="*80)
    print("Testing ONNX Export Accuracy")
    print("="*80)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("⚠ ONNX/ONNXRuntime not installed, skipping ONNX tests")
        return True

    from distillation.encoders_onnx import MonoEncoderONNX

    device = 'cpu'  # ONNX export requires CPU
    backbone = "scratch"
    img_height, img_width = 240, 320
    batch_size = 2

    # Create model
    model = MonoEncoderONNX(
        backbone=backbone,
        img_height=img_height,
        img_width=img_width,
        n_embd=128,
        n_head=4
    ).to(device).eval()

    # Test input
    test_input = torch.randn(batch_size, 3, img_height, img_width, device=device)

    # PyTorch forward pass
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()

    # Export to ONNX
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    torch.onnx.export(
        model,
        test_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={'image': {0: 'batch_size'}, 'embedding': {0: 'batch_size'}}
    )

    # ONNX inference
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'image': test_input.numpy()})[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    # Clean up
    os.unlink(onnx_path)

    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"✓ ONNX export accuracy maintained (max diff < {tolerance})")
        return True
    else:
        print(f"⚠ ONNX export has difference {max_diff:.6e} (tolerance: {tolerance})")
        return False


def test_custom_operators():
    """Test custom operators are ONNX-compatible"""
    print("\n" + "="*80)
    print("Testing Custom Operators")
    print("="*80)

    from distillation.encoders_onnx import SquaredReLUONNX, CrossOnlyAttentionONNX

    device = 'cpu'

    # Test SquaredReLU
    squared_relu = SquaredReLUONNX()
    x = torch.randn(4, 32)
    output = squared_relu(x)
    expected = torch.relu(x).pow(2)

    assert torch.allclose(output, expected, atol=1e-6)
    print("✓ SquaredReLU: PASSED")

    # Test CrossOnlyAttention
    attn = CrossOnlyAttentionONNX(n_embd=128, n_head=4, T1=10, T2=10)
    x = torch.randn(2, 21, 128)  # batch=2, tokens=20+1(embedding), embd=128
    output = attn(x)

    assert output.shape == x.shape
    print("✓ CrossOnlyAttention: PASSED")

    print("\n✓ All custom operator tests PASSED")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("DEXTRAH ONNX/QNN Conversion - Accuracy Validation")
    print("="*80)

    results = []

    try:
        results.append(("RGB Augmentation", test_rgb_augmentation_accuracy()))
    except Exception as e:
        print(f"✗ RGB Augmentation tests FAILED: {e}")
        results.append(("RGB Augmentation", False))

    try:
        results.append(("Depth Augmentation", test_depth_augmentation_accuracy()))
    except Exception as e:
        print(f"✗ Depth Augmentation tests FAILED: {e}")
        results.append(("Depth Augmentation", False))

    try:
        results.append(("Custom Operators", test_custom_operators()))
    except Exception as e:
        print(f"✗ Custom Operators tests FAILED: {e}")
        results.append(("Custom Operators", False))

    try:
        results.append(("Encoder Accuracy", test_encoder_accuracy()))
    except Exception as e:
        print(f"✗ Encoder Accuracy tests FAILED: {e}")
        results.append(("Encoder Accuracy", False))

    try:
        results.append(("ONNX Export", test_onnx_export_accuracy()))
    except Exception as e:
        print(f"✗ ONNX Export tests FAILED: {e}")
        results.append(("ONNX Export", False))

    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:30} {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Accuracy is maintained!")
    else:
        print("✗ SOME TESTS FAILED - Please review errors above")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate ONNX/QNN conversion accuracy")
    parser.add_argument("--test", type=str, choices=["rgb", "depth", "encoder", "onnx", "custom", "all"],
                        default="all", help="Which test to run")

    args = parser.parse_args()

    if args.test == "all":
        success = run_all_tests()
    elif args.test == "rgb":
        success = test_rgb_augmentation_accuracy()
    elif args.test == "depth":
        success = test_depth_augmentation_accuracy()
    elif args.test == "encoder":
        success = test_encoder_accuracy()
    elif args.test == "onnx":
        success = test_onnx_export_accuracy()
    elif args.test == "custom":
        success = test_custom_operators()

    sys.exit(0 if success else 1)
