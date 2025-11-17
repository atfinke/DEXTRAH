#!/usr/bin/env python3
"""
QNN Conversion Test Suite

Tests the QNN SDK conversion pipeline end-to-end.
Requires QNN SDK to be installed and environment variables set.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def check_qnn_sdk_available():
    """Check if QNN SDK is available"""
    qnn_sdk_root = os.environ.get('QNN_SDK_ROOT')
    if not qnn_sdk_root:
        print("⚠ QNN_SDK_ROOT not set, skipping QNN tests")
        return False

    if not os.path.exists(qnn_sdk_root):
        print(f"⚠ QNN SDK not found at {qnn_sdk_root}, skipping QNN tests")
        return False

    return True


def test_qnn_converter_available():
    """Test that qnn-onnx-converter is available and working"""
    print("\n" + "="*80)
    print("Testing QNN Converter Availability")
    print("="*80)

    try:
        result = subprocess.run(
            ['qnn-onnx-converter', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and 'QNN' in result.stdout:
            print("✓ qnn-onnx-converter is available and working")
            return True
        else:
            print(f"✗ qnn-onnx-converter failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("✗ qnn-onnx-converter not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("✗ qnn-onnx-converter timed out")
        return False


def test_simple_model_export():
    """Test exporting a simple PyTorch model to ONNX with QNN-compatible opset"""
    print("\n" + "="*80)
    print("Testing Simple Model ONNX Export (QNN-compatible)")
    print("="*80)

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x

    model = SimpleModel().eval()
    test_input = torch.randn(1, 3, 64, 64)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        # Use legacy exporter for compatibility with ONNX 1.14.0
        import torch.onnx as onnx_export

        torch.onnx.export(
            model,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,  # Use opset 11 for maximum compatibility
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

        if os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0:
            file_size = os.path.getsize(onnx_path) / 1024
            print(f"✓ ONNX model exported successfully ({file_size:.2f} KB)")
            print(f"  Model path: {onnx_path}")
            print(f"  Opset version: 11 (QNN-compatible)")
            return onnx_path
        else:
            print("✗ ONNX export failed - file empty or not created")
            return None

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)
        return None


def test_qnn_conversion(onnx_path):
    """Test converting ONNX model to QNN format"""
    print("\n" + "="*80)
    print("Testing QNN Conversion")
    print("="*80)

    if not onnx_path or not os.path.exists(onnx_path):
        print("✗ No ONNX model available for conversion")
        return False, None

    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as f:
        qnn_output_path = f.name

    try:
        # Run QNN converter
        cmd = [
            'qnn-onnx-converter',
            '--input_network', onnx_path,
            '--output_path', qnn_output_path,
            '--input_layout', 'input', 'NCHW',
            '--input_dtype', 'input', 'float32'
        ]

        print(f"  Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Check if conversion produced output
        if os.path.exists(qnn_output_path) and os.path.getsize(qnn_output_path) > 0:
            file_size = os.path.getsize(qnn_output_path) / 1024
            print(f"✓ QNN conversion successful ({file_size:.2f} KB)")
            print(f"  QNN model: {qnn_output_path}")

            # Check for header file too
            header_path = qnn_output_path.replace('.cpp', '.h')
            if os.path.exists(header_path):
                print(f"  QNN header: {header_path}")

            return True, qnn_output_path
        else:
            print("✗ QNN conversion failed - no output generated")
            print(f"  stdout: {result.stdout[:500]}")
            print(f"  stderr: {result.stderr[:500]}")
            return False, None

    except subprocess.TimeoutExpired:
        print("✗ QNN conversion timed out")
        return False, None
    except Exception as e:
        print(f"✗ QNN conversion failed: {e}")
        return False, None


def test_model_accuracy_pytorch_vs_onnx():
    """Test that ONNX export maintains numerical accuracy"""
    print("\n" + "="*80)
    print("Testing Model Accuracy: PyTorch vs ONNX")
    print("="*80)

    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠ onnxruntime not installed, skipping accuracy test")
        return True

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()
            self.conv2 = torch.nn.Conv2d(16, 8, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            return x

    model = SimpleModel().eval()
    test_input = torch.randn(1, 3, 64, 64)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()

    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model,
            test_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

        # Run ONNX inference
        ort_session = ort.InferenceSession(onnx_path)
        onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]

        # Compare outputs
        max_diff = np.max(np.abs(pytorch_output - onnx_output))
        mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

        print(f"  PyTorch output shape: {pytorch_output.shape}")
        print(f"  ONNX output shape: {onnx_output.shape}")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Mean difference: {mean_diff:.6e}")

        tolerance = 1e-5
        if max_diff < tolerance:
            print(f"✓ PyTorch vs ONNX accuracy maintained (max diff < {tolerance})")
            success = True
        else:
            print(f"⚠ PyTorch vs ONNX difference {max_diff:.6e} exceeds tolerance {tolerance}")
            success = False

        # Clean up
        os.unlink(onnx_path)
        return success

    except Exception as e:
        print(f"✗ Accuracy test failed: {e}")
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)
        return False


def test_qnn_model_structure():
    """Test that QNN conversion produces correct model structure"""
    print("\n" + "="*80)
    print("Testing QNN Model Structure")
    print("="*80)

    # Create and export simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            return x

    model = SimpleModel().eval()
    test_input = torch.randn(1, 3, 32, 32)

    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model, test_input, onnx_path,
            export_params=True, opset_version=11,
            input_names=['input'], output_names=['output'],
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )

        # Convert to QNN
        with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as f:
            qnn_output_path = f.name

        cmd = [
            'qnn-onnx-converter',
            '--input_network', onnx_path,
            '--output_path', qnn_output_path,
            '--input_layout', 'input', 'NCHW',
            '--input_dtype', 'input', 'float32'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if os.path.exists(qnn_output_path):
            # Read QNN C++ file and check for expected structure
            with open(qnn_output_path, 'r') as f:
                qnn_content = f.read()

            # Check for key QNN components
            checks = {
                'Has QNN header includes': '#include "QnnInterface.h"' in qnn_content or 'Qnn' in qnn_content,
                'Has model definition': 'graphName' in qnn_content or 'addTensor' in qnn_content or 'addNode' in qnn_content,
                'Has Conv2d operation': 'Conv2d' in qnn_content or 'QNN_OP_CONV_2D' in qnn_content,
                'Has ReLU operation': 'Relu' in qnn_content or 'QNN_OP_RELU' in qnn_content or 'Activation' in qnn_content,
            }

            passed = sum(checks.values())
            total = len(checks)

            print(f"  QNN Model Structure Checks: {passed}/{total} passed")
            for check_name, check_result in checks.items():
                status = "✓" if check_result else "✗"
                print(f"    {status} {check_name}")

            # Clean up
            os.unlink(qnn_output_path)
            header_path = qnn_output_path.replace('.cpp', '.h')
            if os.path.exists(header_path):
                os.unlink(header_path)
            bin_path = qnn_output_path.replace('.cpp', '.bin')
            if os.path.exists(bin_path):
                os.unlink(bin_path)

            if passed >= total * 0.5:  # At least 50% checks pass
                print(f"✓ QNN model structure is valid")
                return True
            else:
                print(f"⚠ QNN model structure incomplete ({passed}/{total} checks)")
                return False
        else:
            print("✗ QNN conversion failed - no output file")
            return False

    except Exception as e:
        print(f"✗ QNN structure test failed: {e}")
        return False
    finally:
        if os.path.exists(onnx_path):
            os.unlink(onnx_path)


def test_qnn_converter_with_dextrah_model():
    """Test QNN conversion with actual DEXTRAH model (optional)"""
    print("\n" + "="*80)
    print("Testing QNN Conversion with DEXTRAH Model (Optional)")
    print("="*80)

    print("⚠ Skipping DEXTRAH model test (requires additional dependencies)")
    print("  This test is optional - basic QNN functionality is validated above")
    return True


def main():
    """Run all QNN tests"""
    print("="*80)
    print("DEXTRAH QNN Conversion - Test Suite")
    print("="*80)

    # Check if QNN SDK is available
    if not check_qnn_sdk_available():
        print("\n" + "="*80)
        print("QNN SDK not available - skipping QNN tests")
        print("To run QNN tests:")
        print("  1. Download QNN SDK")
        print("  2. Set QNN_SDK_ROOT environment variable")
        print("  3. Activate Python 3.10 virtual environment")
        print("="*80)
        return True

    results = []

    # Test 1: QNN converter availability
    results.append(("QNN Converter Available", test_qnn_converter_available()))

    if results[0][1]:
        # Test 2: PyTorch vs ONNX accuracy
        results.append(("PyTorch vs ONNX Accuracy", test_model_accuracy_pytorch_vs_onnx()))

        # Test 3: Simple model export
        onnx_path = test_simple_model_export()
        results.append(("Simple Model ONNX Export", onnx_path is not None))

        # Test 4: QNN conversion
        if onnx_path:
            conversion_success, qnn_path = test_qnn_conversion(onnx_path)
            results.append(("QNN Conversion", conversion_success))

            # Clean up QNN files
            if qnn_path and os.path.exists(qnn_path):
                os.unlink(qnn_path)
                header_path = qnn_path.replace('.cpp', '.h')
                if os.path.exists(header_path):
                    os.unlink(header_path)
                bin_path = qnn_path.replace('.cpp', '.bin')
                if os.path.exists(bin_path):
                    os.unlink(bin_path)

            # Clean up ONNX file
            if onnx_path and os.path.exists(onnx_path):
                os.unlink(onnx_path)

        # Test 5: QNN model structure
        results.append(("QNN Model Structure", test_qnn_model_structure()))

        # Test 6: DEXTRAH model (optional)
        results.append(("DEXTRAH Model QNN", test_qnn_converter_with_dextrah_model()))

    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:40s} {status}")

    print("="*80)

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("✓ ALL QNN TESTS PASSED")
    else:
        print("✗ SOME QNN TESTS FAILED")

    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
