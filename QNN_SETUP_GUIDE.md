# QNN SDK Setup Guide - Ubuntu 24.04

Complete step-by-step guide for setting up Qualcomm QNN SDK on Ubuntu 24.04 and converting ONNX models to QNN format.

## Prerequisites

- Ubuntu 24.04 (or 22.04 LTS)
- Python 3.10 installed
- ~2GB free disk space for QNN SDK
- Internet connection for downloads

## Step 1: Download QNN SDK v2.35.0

Download the QNN SDK Community Edition (no registration required):

```bash
# Download QNN SDK (1.2 GB)
wget -O /tmp/qnn_sdk.zip "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.35.0.250530/v2.35.0.250530.zip"

# Extract SDK
cd /tmp && unzip -q qnn_sdk.zip

# Verify extraction
ls /tmp/qairt/2.35.0.250530/
```

## Step 2: Install System Dependencies

```bash
# Update package list
sudo apt-get update

# Install required system libraries
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    libc++-dev \
    libc++abi-dev
```

## Step 3: Create Python 3.10 Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv /tmp/qnn_env

# Activate environment
source /tmp/qnn_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 4: Install Compatible Python Dependencies

The QNN SDK v2.35.0 requires specific library versions:

```bash
# Install core dependencies (CRITICAL: use these exact versions)
pip install 'numpy==1.26.3' \
            'onnx==1.14.0' \
            'protobuf<5.0.0' \
            pyyaml \
            packaging \
            sympy \
            pandas

# Install PyTorch 2.1.0 (for model export)
pip install 'torch==2.1.0' --index-url https://download.pytorch.org/whl/cpu
```

**Important Version Notes:**
- `onnx==1.14.0`: QNN SDK v2.35.0 requires this specific version
- `numpy==1.26.3`: Must be <2.0.0 for compatibility
- `torch==2.1.0`: Compatible with ONNX 1.14.0
- `protobuf<5.0.0`: Required for ONNX 1.14.0

## Step 5: Set Environment Variables

Add these to your shell session or `~/.bashrc`:

```bash
# QNN SDK environment variables
export QNN_SDK_ROOT=/tmp/qairt/2.35.0.250530
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
export PATH=${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}
```

To add permanently:

```bash
echo 'export QNN_SDK_ROOT=/tmp/qairt/2.35.0.250530' >> ~/.bashrc
echo 'export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export PATH=${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}' >> ~/.bashrc
source ~/.bashrc
```

## Step 6: Verify Installation

Test that the QNN converter is working:

```bash
# Should show full usage information
qnn-onnx-converter --help
```

Expected output:
```
usage: qnn-onnx-converter [--out_node OUT_NAMES] ...
Script to convert ONNX model into QNN
...
```

## Step 7: Run Test Suite

Verify the complete QNN pipeline:

```bash
cd /home/user/DEXTRAH
./run_qnn_tests.sh
```

Expected output:
```
✓ QNN Converter Available         PASSED
✓ Simple Model ONNX Export         PASSED
✓ QNN Conversion                   PASSED
✓ ALL QNN TESTS PASSED
```

## Usage Example: Convert PyTorch Model to QNN

### 1. Export PyTorch Model to ONNX

```python
import torch
import torch.nn as nn

# Create your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

model = MyModel().eval()
test_input = torch.randn(1, 3, 224, 224)

# Export to ONNX (use opset 11-13 for QNN compatibility)
torch.onnx.export(
    model,
    test_input,
    "my_model.onnx",
    export_params=True,
    opset_version=11,  # CRITICAL: Use opset 11-13
    input_names=['input'],
    output_names=['output'],
    do_constant_folding=True,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
)
```

### 2. Convert ONNX to QNN

```bash
# Activate QNN environment
source /tmp/qnn_env/bin/activate

# Convert ONNX to QNN
qnn-onnx-converter \
    --input_network my_model.onnx \
    --output_path my_model_qnn.cpp \
    --input_layout "input" NCHW \
    --input_dtype "input" float32
```

This generates:
- `my_model_qnn.cpp`: C++ model definition
- `my_model_qnn.h`: Header file
- `my_model_qnn.bin`: Model weights

### 3. (Optional) Quantize for Hexagon NPU

```bash
# Quantize to INT8 for edge deployment
qnn-quantizer \
    --input_network my_model_qnn.cpp \
    --output_network my_model_qnn_quantized.cpp \
    --weights_bitwidth 8 \
    --act_bitwidth 8
```

## Important Notes

### ONNX Opset Compatibility

**QNN SDK v2.35.0 works best with ONNX opset 11-13:**

| PyTorch Version | Default Opset | QNN Compatibility | Solution |
|----------------|---------------|-------------------|----------|
| PyTorch 2.1.0  | 13-14        | ✓ Good           | Use as-is |
| PyTorch 2.4+   | 17-18        | ⚠ Limited        | Specify `opset_version=11` |

Always specify `opset_version=11` or `opset_version=13` when exporting:

```python
torch.onnx.export(
    model, input, "model.onnx",
    opset_version=11  # Add this!
)
```

### Ubuntu 24.04 Compatibility

While QNN SDK officially supports Ubuntu 22.04, it works on Ubuntu 24.04 with the Python 3.10 virtual environment approach described above.

### Newer QNN SDK Versions

For better ONNX opset 17-18 support, consider:
- **QNN SDK v2.37.0**: Improved ONNX compatibility
- **QNN SDK v2.38.0**: Latest version (September 2025)

## Troubleshooting

### Error: "No module named 'yaml'"
```bash
pip install pyyaml
```

### Error: "No module named 'pandas'"
```bash
pip install pandas
```

### Error: "libc++.so.1: cannot open shared object file"
```bash
sudo apt-get install libc++-dev libc++abi-dev
```

### Error: "Python version mismatch"
Ensure you're using Python 3.10 in the virtual environment:
```bash
which python3  # Should show /tmp/qnn_env/bin/python3
python3 --version  # Should show Python 3.10.x
```

### Error: "AttributeError: 'NoneType' object has no attribute 'AttributeProto'"
This means ONNX version mismatch. Reinstall with correct version:
```bash
pip uninstall onnx
pip install 'onnx==1.14.0'
```

### Error: "Unsupported value of 1 provided as allowzero attribute for Reshape"
Your ONNX model uses features not supported by QNN v2.35.0:
- Solution 1: Export with lower opset version (`opset_version=11`)
- Solution 2: Upgrade to QNN SDK v2.37+ for better compatibility

### Warning: "Unsupported Ubuntu version 24.04"
This warning can be ignored. The SDK works on Ubuntu 24.04 with proper Python dependencies.

## Testing Your Setup

Use the included test suite:

```bash
# Run QNN conversion tests
./run_qnn_tests.sh
```

This validates:
1. QNN SDK tools are accessible
2. ONNX export works with compatible settings
3. QNN conversion pipeline functions correctly

## Additional Resources

- **QNN SDK Documentation**: Check `/tmp/qairt/2.35.0.250530/docs/`
- **ONNX Opset Versions**: https://github.com/onnx/onnx/blob/main/docs/Versioning.md
- **PyTorch ONNX Export**: https://pytorch.org/docs/stable/onnx.html

## Quick Reference Commands

```bash
# Activate QNN environment
source /tmp/qnn_env/bin/activate

# Check QNN converter
qnn-onnx-converter --help

# List available QNN tools
ls ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/

# Run tests
./run_qnn_tests.sh

# Export PyTorch to ONNX (QNN-compatible)
torch.onnx.export(model, input, "model.onnx", opset_version=11)

# Convert ONNX to QNN
qnn-onnx-converter --input_network model.onnx --output_path model_qnn.cpp
```

## Production Deployment Checklist

- [ ] ONNX model exported with opset 11-13
- [ ] QNN conversion successful (generates .cpp and .h files)
- [ ] Quantization applied if deploying to Hexagon NPU
- [ ] Model tested on target hardware
- [ ] Performance benchmarks validated

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17
**QNN SDK Version**: v2.35.0.250530
**Tested On**: Ubuntu 24.04 LTS
