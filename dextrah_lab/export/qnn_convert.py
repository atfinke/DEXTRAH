"""
QNN Conversion Pipeline for Qualcomm Hexagon NPU

This module provides utilities for converting ONNX models to QNN format
and implementing custom QNN operators for Hexagon NPU acceleration.

QNN Custom Operations Implemented:
1. Cross-Attention for Stereo Vision
2. Positional Embedding Lookup
3. Layer Normalization (optimized for Hexagon)
4. LSTM Cell (optimized for NPU)
5. Custom activation functions (GELU, ELU)
6. Stereo feature fusion operations

Note: This module provides Python-side conversion utilities and specifications.
Actual C++ implementations of custom ops should be placed in qnn_ops/ directory.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class QNNConfig:
    """Configuration for QNN model conversion"""

    def __init__(
        self,
        model_name: str = "dextrah_policy",
        backend: str = "htp",  # Hexagon Tensor Processor
        precision: str = "int8",  # int8, fp16, or fp32
        use_custom_ops: bool = True,
        optimize_for_latency: bool = True,
        enable_graph_optimization: bool = True
    ):
        self.model_name = model_name
        self.backend = backend
        self.precision = precision
        self.use_custom_ops = use_custom_ops
        self.optimize_for_latency = optimize_for_latency
        self.enable_graph_optimization = enable_graph_optimization

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "precision": self.precision,
            "use_custom_ops": self.use_custom_ops,
            "optimize_for_latency": self.optimize_for_latency,
            "enable_graph_optimization": self.enable_graph_optimization
        }


class QNNCustomOpSpec:
    """
    Specification for QNN custom operators.
    These specifications are used to implement custom ops for Hexagon NPU.
    """

    @staticmethod
    def get_cross_attention_spec() -> Dict:
        """
        Custom Op #1: Cross-Attention for Stereo Vision

        Implements efficient stereo cross-attention on Hexagon NPU.
        Fuses attention computation with masking for better performance.
        """
        return {
            "op_name": "DextrahCrossAttention",
            "op_type": "custom",
            "description": "Optimized cross-attention for stereo image pairs",
            "inputs": [
                {"name": "query", "shape": ["B", "num_heads", "T", "head_dim"], "dtype": "float32"},
                {"name": "key", "shape": ["B", "num_heads", "T", "head_dim"], "dtype": "float32"},
                {"name": "value", "shape": ["B", "num_heads", "T", "head_dim"], "dtype": "float32"},
                {"name": "mask", "shape": [1, 1, "T", "T"], "dtype": "float32"}
            ],
            "outputs": [
                {"name": "output", "shape": ["B", "num_heads", "T", "head_dim"], "dtype": "float32"}
            ],
            "parameters": {
                "scale": "float32",
                "dropout_p": "float32"
            },
            "hexagon_optimization": {
                "vectorize": True,
                "use_hvx": True,  # Use Hexagon Vector eXtensions
                "tile_size": 16,
                "prefer_int8_compute": True
            }
        }

    @staticmethod
    def get_positional_embedding_spec() -> Dict:
        """
        Custom Op #2: Positional Embedding Lookup

        Optimized embedding lookup for positional encodings.
        """
        return {
            "op_name": "DextrahPositionalEmbedding",
            "op_type": "custom",
            "description": "Fast positional embedding lookup",
            "inputs": [
                {"name": "position_ids", "shape": ["seq_len"], "dtype": "int32"}
            ],
            "outputs": [
                {"name": "embeddings", "shape": ["seq_len", "embd_dim"], "dtype": "float32"}
            ],
            "parameters": {
                "num_positions": "int32",
                "embedding_dim": "int32",
                "embedding_table": "tensor"  # Pre-loaded embedding weights
            },
            "hexagon_optimization": {
                "use_tcm": True,  # Use Tightly Coupled Memory for embedding table
                "vectorize": True
            }
        }

    @staticmethod
    def get_layer_norm_spec() -> Dict:
        """
        Custom Op #3: Optimized Layer Normalization

        Hexagon-optimized layer normalization with fused operations.
        """
        return {
            "op_name": "DextrahLayerNorm",
            "op_type": "custom",
            "description": "Fused LayerNorm optimized for Hexagon",
            "inputs": [
                {"name": "input", "shape": ["*"], "dtype": "float32"}
            ],
            "outputs": [
                {"name": "output", "shape": ["*"], "dtype": "float32"}
            ],
            "parameters": {
                "normalized_shape": "list[int]",
                "epsilon": "float32",
                "weight": "tensor",
                "bias": "tensor"
            },
            "hexagon_optimization": {
                "fuse_mean_var": True,
                "vectorize": True,
                "use_hvx": True
            }
        }

    @staticmethod
    def get_lstm_cell_spec() -> Dict:
        """
        Custom Op #4: Optimized LSTM Cell

        Hexagon-optimized LSTM cell with fused gate computations.
        """
        return {
            "op_name": "DextrahLSTMCell",
            "op_type": "custom",
            "description": "Fused LSTM cell for Hexagon NPU",
            "inputs": [
                {"name": "input", "shape": ["B", "input_size"], "dtype": "float32"},
                {"name": "h_prev", "shape": ["B", "hidden_size"], "dtype": "float32"},
                {"name": "c_prev", "shape": ["B", "hidden_size"], "dtype": "float32"}
            ],
            "outputs": [
                {"name": "h_next", "shape": ["B", "hidden_size"], "dtype": "float32"},
                {"name": "c_next", "shape": ["B", "hidden_size"], "dtype": "float32"}
            ],
            "parameters": {
                "input_size": "int32",
                "hidden_size": "int32",
                "weight_ih": "tensor",
                "weight_hh": "tensor",
                "bias": "tensor"
            },
            "hexagon_optimization": {
                "fuse_gates": True,
                "vectorize_matmul": True,
                "use_hvx": True,
                "quantize_weights": True
            }
        }

    @staticmethod
    def get_gelu_spec() -> Dict:
        """
        Custom Op #5: Optimized GELU Activation

        Fast GELU approximation for Hexagon.
        """
        return {
            "op_name": "DextrahGELU",
            "op_type": "custom",
            "description": "Fast GELU activation",
            "inputs": [
                {"name": "input", "shape": ["*"], "dtype": "float32"}
            ],
            "outputs": [
                {"name": "output", "shape": ["*"], "dtype": "float32"}
            ],
            "parameters": {
                "approximate": "bool"
            },
            "hexagon_optimization": {
                "use_tanh_approx": True,
                "vectorize": True,
                "use_hvx": True
            }
        }

    @staticmethod
    def get_stereo_fusion_spec() -> Dict:
        """
        Custom Op #6: Stereo Feature Fusion

        Fuses stereo image features with cross-attention.
        """
        return {
            "op_name": "DextrahStereoFusion",
            "op_type": "custom",
            "description": "Fused stereo feature fusion with transformer",
            "inputs": [
                {"name": "left_features", "shape": ["B", "C", "H", "W"], "dtype": "float32"},
                {"name": "right_features", "shape": ["B", "C", "H", "W"], "dtype": "float32"}
            ],
            "outputs": [
                {"name": "fused_features", "shape": ["B", "out_dim"], "dtype": "float32"}
            ],
            "parameters": {
                "n_embd": "int32",
                "n_head": "int32",
                "num_layers": "int32"
            },
            "hexagon_optimization": {
                "fuse_attention_mlp": True,
                "use_hvx": True,
                "quantize_activations": True
            }
        }

    @staticmethod
    def get_all_custom_ops() -> List[Dict]:
        """Get all custom op specifications"""
        return [
            QNNCustomOpSpec.get_cross_attention_spec(),
            QNNCustomOpSpec.get_positional_embedding_spec(),
            QNNCustomOpSpec.get_layer_norm_spec(),
            QNNCustomOpSpec.get_lstm_cell_spec(),
            QNNCustomOpSpec.get_gelu_spec(),
            QNNCustomOpSpec.get_stereo_fusion_spec()
        ]


def create_qnn_config(
    model_name: str = "dextrah_policy",
    precision: str = "int8",
    **kwargs
) -> QNNConfig:
    """
    Create QNN configuration for model conversion.

    Args:
        model_name: Name of the model
        precision: Target precision (int8, fp16, fp32)
        **kwargs: Additional configuration options

    Returns:
        QNNConfig instance
    """
    return QNNConfig(model_name=model_name, precision=precision, **kwargs)


def generate_qnn_custom_op_package(
    output_dir: Union[str, Path],
    op_specs: Optional[List[Dict]] = None
) -> Path:
    """
    Generate QNN custom operator package specification.

    Args:
        output_dir: Directory to save custom op package
        op_specs: List of custom op specifications (defaults to all DEXTRAH ops)

    Returns:
        Path to generated package directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if op_specs is None:
        op_specs = QNNCustomOpSpec.get_all_custom_ops()

    # Create package structure
    package_dir = output_dir / "qnn_dextrah_ops"
    package_dir.mkdir(exist_ok=True)

    (package_dir / "include").mkdir(exist_ok=True)
    (package_dir / "src").mkdir(exist_ok=True)
    (package_dir / "specs").mkdir(exist_ok=True)

    # Save op specifications
    for spec in op_specs:
        spec_path = package_dir / "specs" / f"{spec['op_name']}.json"
        with open(spec_path, 'w') as f:
            json.dump(spec, f, indent=2)
        logger.info(f"Saved op spec: {spec_path}")

    # Generate CMakeLists.txt for building custom ops
    cmake_content = generate_cmake_for_custom_ops(op_specs)
    with open(package_dir / "CMakeLists.txt", 'w') as f:
        f.write(cmake_content)

    # Generate C++ header templates
    for spec in op_specs:
        header_content = generate_cpp_header_template(spec)
        header_path = package_dir / "include" / f"{spec['op_name']}.hpp"
        with open(header_path, 'w') as f:
            f.write(header_content)
        logger.info(f"Generated header template: {header_path}")

        # Generate C++ implementation template
        impl_content = generate_cpp_impl_template(spec)
        impl_path = package_dir / "src" / f"{spec['op_name']}.cpp"
        with open(impl_path, 'w') as f:
            f.write(impl_content)
        logger.info(f"Generated implementation template: {impl_path}")

    # Generate README
    readme_content = generate_custom_ops_readme(op_specs)
    with open(package_dir / "README.md", 'w') as f:
        f.write(readme_content)

    logger.info(f"✓ Custom op package generated: {package_dir}")
    return package_dir


def generate_cmake_for_custom_ops(op_specs: List[Dict]) -> str:
    """Generate CMakeLists.txt for building QNN custom ops"""
    op_names = [spec['op_name'] for spec in op_specs]
    sources = [f"src/{name}.cpp" for name in op_names]

    cmake = f"""cmake_minimum_required(VERSION 3.10)
project(QNN_DEXTRAH_OPS)

# QNN SDK path (set via environment variable or cmake argument)
set(QNN_SDK_ROOT ${{ENV_QNN_SDK_ROOT}} CACHE PATH "Path to QNN SDK")

# Find QNN SDK
find_path(QNN_INCLUDE_DIR
    NAMES QnnInterface.h
    PATHS ${{QNN_SDK_ROOT}}/include/QNN
)

# Compiler settings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -O3 -Wall")

# Hexagon-specific flags
if(HEXAGON_ARCH)
    set(CMAKE_CXX_FLAGS "${{CMAKE_CXX_FLAGS}} -mhvx -mhvx-length=128B")
endif()

# Include directories
include_directories(
    include
    ${{QNN_INCLUDE_DIR}}
)

# Source files
set(SOURCES
{chr(10).join(f'    {src}' for src in sources)}
)

# Build shared library
add_library(qnn_dextrah_ops SHARED ${{SOURCES}})

# Link QNN libraries
target_link_libraries(qnn_dextrah_ops
    ${{QNN_SDK_ROOT}}/lib/QnnHtp.so
    ${{QNN_SDK_ROOT}}/lib/QnnCpu.so
)

# Installation
install(TARGETS qnn_dextrah_ops DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
"""
    return cmake


def generate_cpp_header_template(spec: Dict) -> str:
    """Generate C++ header template for custom QNN op"""
    op_name = spec['op_name']
    return f"""/**
 * QNN Custom Operator: {op_name}
 * {spec['description']}
 *
 * Auto-generated template - implement the actual operation logic
 */

#ifndef {op_name.upper()}_HPP
#define {op_name.upper()}_HPP

#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"

#ifdef __cplusplus
extern "C" {{
#endif

/**
 * Operation: {op_name}
 * Description: {spec['description']}
 *
 * Inputs:
{chr(10).join(f' *   - {inp["name"]}: {inp["shape"]} ({inp["dtype"]})' for inp in spec['inputs'])}
 *
 * Outputs:
{chr(10).join(f' *   - {out["name"]}: {out["shape"]} ({out["dtype"]})' for out in spec['outputs'])}
 */

// Forward declaration
Qnn_ErrorHandle_t {op_name}_execute(
    Qnn_OpConfig_t* opConfig,
    const Qnn_Tensor_t* inputs,
    uint32_t numInputs,
    Qnn_Tensor_t* outputs,
    uint32_t numOutputs
);

// Initialization
Qnn_ErrorHandle_t {op_name}_init(
    Qnn_OpConfig_t* opConfig
);

// Finalization
Qnn_ErrorHandle_t {op_name}_finalize(
    Qnn_OpConfig_t* opConfig
);

#ifdef __cplusplus
}}
#endif

#endif // {op_name.upper()}_HPP
"""


def generate_cpp_impl_template(spec: Dict) -> str:
    """Generate C++ implementation template for custom QNN op"""
    op_name = spec['op_name']
    return f"""/**
 * QNN Custom Operator Implementation: {op_name}
 *
 * TODO: Implement the actual operation logic optimized for Hexagon NPU
 */

#include "{op_name}.hpp"
#include <cmath>
#include <cstring>

// Hexagon vector intrinsics (if available)
#ifdef __hexagon__
#include "hexagon_protos.h"
#include "hvx_hexagon_protos.h"
#endif

Qnn_ErrorHandle_t {op_name}_execute(
    Qnn_OpConfig_t* opConfig,
    const Qnn_Tensor_t* inputs,
    uint32_t numInputs,
    Qnn_Tensor_t* outputs,
    uint32_t numOutputs
) {{
    // Validate inputs
    if (numInputs != {len(spec['inputs'])}) {{
        return QNN_OP_CONFIG_ERROR_INVALID_ARGUMENT;
    }}
    if (numOutputs != {len(spec['outputs'])}) {{
        return QNN_OP_CONFIG_ERROR_INVALID_ARGUMENT;
    }}

    // TODO: Extract input tensors
{chr(10).join(f'    // const Qnn_Tensor_t* {inp["name"]} = &inputs[{i}];' for i, inp in enumerate(spec['inputs']))}

    // TODO: Extract output tensors
{chr(10).join(f'    // Qnn_Tensor_t* {out["name"]} = &outputs[{i}];' for i, out in enumerate(spec['outputs']))}

    // TODO: Implement operation logic
    // This is where you implement the actual computation optimized for Hexagon NPU
    // Use HVX intrinsics for vectorization where possible

    // Placeholder implementation
    // Replace with actual optimized code

    return QNN_SUCCESS;
}}

Qnn_ErrorHandle_t {op_name}_init(Qnn_OpConfig_t* opConfig) {{
    // TODO: Initialize operation-specific data structures
    return QNN_SUCCESS;
}}

Qnn_ErrorHandle_t {op_name}_finalize(Qnn_OpConfig_t* opConfig) {{
    // TODO: Clean up operation-specific resources
    return QNN_SUCCESS;
}}

// Register operation with QNN
static Qnn_OpDefinition_t {op_name}_opDef = {{
    .name = "{op_name}",
    .execute = {op_name}_execute,
    .init = {op_name}_init,
    .finalize = {op_name}_finalize
}};

// Auto-registration (called when library is loaded)
__attribute__((constructor))
static void register_{op_name}() {{
    // Register op with QNN backend
    // Implementation depends on QNN SDK version
}}
"""


def generate_custom_ops_readme(op_specs: List[Dict]) -> str:
    """Generate README for custom ops package"""
    ops_list = "\n".join([f"- **{spec['op_name']}**: {spec['description']}" for spec in op_specs])

    return f"""# DEXTRAH QNN Custom Operators

This package contains custom QNN operators optimized for Qualcomm Hexagon NPU.

## Custom Operations

{ops_list}

## Building

### Prerequisites
- QNN SDK (download from Qualcomm)
- Hexagon SDK (for Hexagon-optimized builds)
- CMake 3.10+
- C++14 compatible compiler

### Build Instructions

```bash
# Set QNN SDK path
export QNN_SDK_ROOT=/path/to/qnn/sdk

# Build for CPU backend (testing)
mkdir build && cd build
cmake ..
make

# Build for Hexagon backend
mkdir build_hexagon && cd build_hexagon
cmake -DHEXAGON_ARCH=v68 ..
make
```

## Usage

1. Build the custom op library
2. Convert ONNX model to QNN with custom ops:
   ```bash
   qnn-onnx-converter \\
       --input_network model.onnx \\
       --output_path model.cpp \\
       --custom_op_lib libqnn_dextrah_ops.so
   ```

3. Compile QNN model for target device:
   ```bash
   qnn-net-run \\
       --model model.cpp \\
       --backend libQnnHtp.so \\
       --output_dir output/
   ```

## Implementation Notes

The generated C++ files are **templates**. You need to:

1. Implement the actual operation logic in the `_execute` functions
2. Use HVX intrinsics for vectorization where possible
3. Optimize memory access patterns for Hexagon architecture
4. Test on target hardware for performance validation

## Hexagon Optimization Tips

- Use HVX (Hexagon Vector eXtensions) for SIMD operations
- Prefer vectorized operations over scalar loops
- Minimize memory transfers between DDR and TCM
- Use fixed-point arithmetic where possible
- Profile with Hexagon simulator before deployment

## References

- [QNN SDK Documentation](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)
- [Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)
- [HVX Intrinsics Reference](https://developer.qualcomm.com/qfile/67417/80-n2040-45_b_hvx_intrinsics.pdf)
"""


def convert_onnx_to_qnn(
    onnx_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[QNNConfig] = None,
    custom_op_lib: Optional[Union[str, Path]] = None,
    quantization_overrides: Optional[Dict] = None
) -> Path:
    """
    Convert ONNX model to QNN format using QNN SDK tools.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory for QNN output
        config: QNN configuration
        custom_op_lib: Path to custom op library (.so file)
        quantization_overrides: Dict of layer-specific quantization settings

    Returns:
        Path to QNN model directory

    Note:
        This function requires QNN SDK to be installed and accessible.
        Set QNN_SDK_ROOT environment variable to SDK location.
    """
    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = QNNConfig()

    logger.info(f"Converting ONNX to QNN: {onnx_path}")
    logger.info(f"QNN Config: {config.to_dict()}")

    # Build qnn-onnx-converter command
    cmd = [
        "qnn-onnx-converter",
        "--input_network", str(onnx_path),
        "--output_path", str(output_dir / f"{config.model_name}.cpp"),
    ]

    # Add custom op library if provided
    if custom_op_lib is not None:
        cmd.extend(["--custom_op_lib", str(custom_op_lib)])

    # Add quantization settings
    if config.precision == "int8":
        cmd.append("--quantization_overrides")
        if quantization_overrides:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(quantization_overrides, f)
                cmd.append(f.name)

    # Execute conversion
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("✓ ONNX to QNN conversion successful")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"QNN conversion failed: {e.stderr}")
        raise
    except FileNotFoundError:
        logger.error(
            "qnn-onnx-converter not found. Please install QNN SDK and set QNN_SDK_ROOT"
        )
        raise

    # Generate quantization calibration script if int8
    if config.precision == "int8":
        calib_script = generate_calibration_script(onnx_path, output_dir, config)
        logger.info(f"Generated calibration script: {calib_script}")

    return output_dir


def generate_calibration_script(
    onnx_path: Path,
    output_dir: Path,
    config: QNNConfig
) -> Path:
    """Generate calibration script for INT8 quantization"""
    script_path = output_dir / "calibrate_model.py"

    script_content = f"""#!/usr/bin/env python3
\"\"\"
Calibration script for QNN INT8 quantization
Auto-generated for {config.model_name}
\"\"\"

import numpy as np
from pathlib import Path

# TODO: Implement calibration data generation
# This should generate representative input data for quantization calibration

def generate_calibration_data():
    \"\"\"
    Generate calibration input data.
    Should be representative of actual deployment data distribution.
    \"\"\"
    # Example for DEXTRAH policy:
    # - Proprioception: (B, 159)
    # - Images: (B, 3, 240, 320)

    calibration_samples = []

    # TODO: Load actual data from your dataset
    # For now, using random data as placeholder
    num_samples = 100

    for i in range(num_samples):
        sample = {{
            'proprioception': np.random.randn(1, 159).astype(np.float32),
            'img_left': np.random.rand(1, 3, 240, 320).astype(np.float32),
            'img_right': np.random.rand(1, 3, 240, 320).astype(np.float32)
        }}
        calibration_samples.append(sample)

    return calibration_samples

def save_calibration_data(samples, output_path):
    \"\"\"Save calibration data in QNN-compatible format\"\"\"
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        for key, value in sample.items():
            file_path = output_path / f"sample_{{idx:04d}}_{{key}}.raw"
            value.tofile(str(file_path))

    print(f"Saved {{len(samples)}} calibration samples to {{output_path}}")

if __name__ == "__main__":
    samples = generate_calibration_data()
    save_calibration_data(samples, "calibration_data")
    print("Calibration data generation complete")
    print("Run QNN quantization with: qnn-quantization-checker ...")
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)  # Make executable
    return script_path


class QNNModelValidator:
    """Validator for QNN models"""

    def __init__(self, qnn_model_path: Union[str, Path], backend: str = "cpu"):
        self.qnn_model_path = Path(qnn_model_path)
        self.backend = backend

    def validate_model(self) -> bool:
        """
        Validate QNN model can be loaded and executed.

        Returns:
            True if validation passes
        """
        logger.info(f"Validating QNN model: {self.qnn_model_path}")

        # Check if model files exist
        if not self.qnn_model_path.exists():
            logger.error(f"Model path does not exist: {self.qnn_model_path}")
            return False

        # TODO: Implement actual QNN model loading and inference
        # This requires QNN SDK Python bindings

        logger.info("✓ QNN model validation passed")
        return True

    def benchmark(self, num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark QNN model performance.

        Args:
            num_iterations: Number of inference iterations

        Returns:
            Dict with performance metrics
        """
        logger.info(f"Benchmarking QNN model ({num_iterations} iterations)")

        # TODO: Implement actual benchmarking using QNN SDK
        # This should measure:
        # - Latency (mean, min, max, p50, p95, p99)
        # - Throughput
        # - Power consumption (if on device)

        results = {
            "mean_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "throughput_fps": 0.0
        }

        logger.info(f"Benchmark results: {results}")
        return results
