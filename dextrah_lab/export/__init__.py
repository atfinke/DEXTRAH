"""
DEXTRAH Model Export Package

This package provides utilities for exporting DEXTRAH models to:
1. ONNX format (with PyTorch operation compatibility)
2. QNN format (for Qualcomm Hexagon NPU acceleration)

Key operations re-implemented for ONNX/QNN compatibility:
- Scaled dot-product attention
- Custom cross-attention with masks
- LSTM with state management
- RunningMeanStd normalization
- Adaptive pooling operations
- Positional embeddings
- Various activation functions (GELU, ELU)
- Complex tensor reshaping operations
"""

from .onnx_export import (
    export_model_to_onnx,
    validate_onnx_export,
    ONNXCompatibleStereoEncoder,
    ONNXCompatiblePolicy
)

from .qnn_convert import (
    convert_onnx_to_qnn,
    create_qnn_config,
    QNNModelValidator
)

__all__ = [
    'export_model_to_onnx',
    'validate_onnx_export',
    'ONNXCompatibleStereoEncoder',
    'ONNXCompatiblePolicy',
    'convert_onnx_to_qnn',
    'create_qnn_config',
    'QNNModelValidator'
]
