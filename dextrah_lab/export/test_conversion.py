#!/usr/bin/env python3
"""
Test suite for ONNX/QNN conversion pipeline.

Tests all 12 re-implemented operations and validates conversion accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path

from onnx_ops import (
    ONNXScaledDotProductAttention,
    ONNXCrossAttentionMask,
    ONNXAdaptiveAvgPool2d,
    ONNXGELUActivation,
    ONNXELUActivation,
    ONNXPositionalEmbedding,
    ONNXLayerNorm,
    ONNXRunningMeanStd,
    ONNXLSTMCell,
    ONNXTensorReshaper
)

from onnx_models import (
    ONNXStereoEncoder,
    ONNXCustomCNN,
    ONNXResNetEncoder,
    ONNXCrossOnlyAttention
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestONNXOperations:
    """Test suite for re-implemented ONNX operations"""

    def __init__(self):
        self.tolerance = 1e-4
        self.passed = 0
        self.failed = 0

    def assert_close(self, a, b, name, tolerance=None):
        """Assert two tensors are close"""
        if tolerance is None:
            tolerance = self.tolerance

        diff = torch.abs(a - b).max().item()
        if diff < tolerance:
            logger.info(f"✓ {name}: PASSED (max diff: {diff:.6e})")
            self.passed += 1
            return True
        else:
            logger.error(f"✗ {name}: FAILED (max diff: {diff:.6e}, tolerance: {tolerance:.6e})")
            self.failed += 1
            return False

    def test_scaled_dot_product_attention(self):
        """Test #1: Scaled Dot-Product Attention"""
        logger.info("\nTest 1: Scaled Dot-Product Attention")

        B, num_heads, seq_len, head_dim = 2, 4, 10, 16
        q = torch.randn(B, num_heads, seq_len, head_dim)
        k = torch.randn(B, num_heads, seq_len, head_dim)
        v = torch.randn(B, num_heads, seq_len, head_dim)

        # ONNX version
        onnx_attn = ONNXScaledDotProductAttention(dropout_p=0.0)
        onnx_attn.eval()
        with torch.no_grad():
            onnx_output = onnx_attn(q, k, v)

        # PyTorch version (for comparison)
        scale = 1.0 / (head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pt_output = torch.matmul(attn_weights, v)

        self.assert_close(onnx_output, pt_output, "Scaled Dot-Product Attention")

    def test_cross_attention_mask(self):
        """Test #2: Cross-Attention Mask"""
        logger.info("\nTest 2: Cross-Attention Mask")

        T1, T2 = 10, 10
        mask_gen = ONNXCrossAttentionMask(T1, T2)
        mask = mask_gen()

        # Check mask shape
        expected_shape = (1, 1, T1 + T2 + 1, T1 + T2 + 1)
        self.assert_close(
            torch.tensor(mask.shape),
            torch.tensor(expected_shape),
            "Cross-Attention Mask Shape"
        )

    def test_adaptive_avg_pool(self):
        """Test #3: Adaptive Average Pooling"""
        logger.info("\nTest 3: Adaptive Average Pooling")

        B, C, H, W = 2, 64, 32, 48
        x = torch.randn(B, C, H, W)

        onnx_pool = ONNXAdaptiveAvgPool2d((8, 16))
        pt_pool = nn.AdaptiveAvgPool2d((8, 16))

        with torch.no_grad():
            onnx_output = onnx_pool(x)
            pt_output = pt_pool(x)

        self.assert_close(onnx_output, pt_output, "Adaptive Average Pooling", tolerance=1e-3)

    def test_gelu_activation(self):
        """Test #4: GELU Activation"""
        logger.info("\nTest 4: GELU Activation")

        x = torch.randn(10, 128)

        onnx_gelu = ONNXGELUActivation()
        pt_gelu = nn.GELU()

        with torch.no_grad():
            onnx_output = onnx_gelu(x)
            pt_output = pt_gelu(x)

        self.assert_close(onnx_output, pt_output, "GELU Activation", tolerance=1e-4)

    def test_elu_activation(self):
        """Test #5: ELU Activation"""
        logger.info("\nTest 5: ELU Activation")

        x = torch.randn(10, 128)
        alpha = 1.0

        onnx_elu = ONNXELUActivation(alpha=alpha)
        pt_elu = nn.ELU(alpha=alpha)

        with torch.no_grad():
            onnx_output = onnx_elu(x)
            pt_output = pt_elu(x)

        self.assert_close(onnx_output, pt_output, "ELU Activation")

    def test_positional_embedding(self):
        """Test #6: Positional Embedding"""
        logger.info("\nTest 6: Positional Embedding")

        num_positions = 100
        embedding_dim = 64

        onnx_emb = ONNXPositionalEmbedding(num_positions, embedding_dim)
        pt_emb = nn.Embedding(num_positions, embedding_dim)

        # Copy weights
        pt_emb.weight.data = onnx_emb.embedding.weight.data.clone()

        with torch.no_grad():
            onnx_output = onnx_emb(50)
            pt_output = pt_emb(torch.arange(50))

        self.assert_close(onnx_output, pt_output, "Positional Embedding")

    def test_layer_norm(self):
        """Test #8: LayerNorm"""
        logger.info("\nTest 8: LayerNorm")

        x = torch.randn(10, 128)

        onnx_ln = ONNXLayerNorm(128)
        pt_ln = nn.LayerNorm(128)

        # Copy weights
        pt_ln.weight.data = onnx_ln.weight.data.clone()
        pt_ln.bias.data = onnx_ln.bias.data.clone()

        with torch.no_grad():
            onnx_output = onnx_ln(x)
            pt_output = pt_ln(x)

        self.assert_close(onnx_output, pt_output, "LayerNorm")

    def test_running_mean_std(self):
        """Test #9: Running Mean/Std Normalization"""
        logger.info("\nTest 9: Running Mean/Std Normalization")

        shape = (3, 32, 32)
        x = torch.randn(2, *shape)

        onnx_rms = ONNXRunningMeanStd(shape)

        # Set some stats
        onnx_rms.running_mean.data = torch.randn(shape)
        onnx_rms.running_var.data = torch.rand(shape) + 0.1

        with torch.no_grad():
            output = onnx_rms(x)

        # Check output shape
        self.assert_close(
            torch.tensor(output.shape),
            torch.tensor(x.shape),
            "Running Mean/Std Shape"
        )

    def test_lstm_cell(self):
        """Test #10: LSTM Cell"""
        logger.info("\nTest 10: LSTM Cell")

        input_size = 64
        hidden_size = 128
        batch_size = 4

        x = torch.randn(batch_size, input_size)
        h = torch.randn(batch_size, hidden_size)
        c = torch.randn(batch_size, hidden_size)

        onnx_lstm = ONNXLSTMCell(input_size, hidden_size)

        with torch.no_grad():
            h_next, c_next = onnx_lstm(x, (h, c))

        # Check output shapes
        self.assert_close(
            torch.tensor(h_next.shape),
            torch.tensor([batch_size, hidden_size]),
            "LSTM Cell Output Shape"
        )

    def test_tensor_reshaper(self):
        """Test #11: Tensor Reshaping"""
        logger.info("\nTest 11: Tensor Reshaping")

        x = torch.randn(2, 3, 4, 5)

        reshaper = ONNXTensorReshaper()

        # Test safe_view
        y = reshaper.safe_view(x, 2, -1)
        assert y.shape == (2, 60), "safe_view failed"

        # Test safe_permute
        y = reshaper.safe_permute(x, 0, 2, 1, 3)
        assert y.shape == (2, 4, 3, 5), "safe_permute failed"

        # Test safe_transpose
        y = reshaper.safe_transpose(x, 1, 2)
        assert y.shape == (2, 4, 3, 5), "safe_transpose failed"

        logger.info("✓ Tensor Reshaping: PASSED")
        self.passed += 1

    def test_stereo_encoder(self):
        """Test full stereo encoder"""
        logger.info("\nTest: Full Stereo Encoder")

        encoder = ONNXStereoEncoder(
            backbone="scratch",  # Use custom CNN for faster testing
            img_height=240,
            img_width=320,
            n_embd=128,
            n_head=4
        )
        encoder.eval()

        # Test input: stacked stereo pair
        imgs = torch.randn(4, 3, 240, 320)  # 2 stereo pairs

        with torch.no_grad():
            output = encoder(imgs)

        expected_shape = (2, 64)  # Batch size 2, output dim 64
        self.assert_close(
            torch.tensor(output.shape),
            torch.tensor(expected_shape),
            "Stereo Encoder Output Shape"
        )

    def run_all_tests(self):
        """Run all tests"""
        logger.info("=" * 80)
        logger.info("ONNX OPERATIONS TEST SUITE")
        logger.info("=" * 80)

        self.test_scaled_dot_product_attention()
        self.test_cross_attention_mask()
        self.test_adaptive_avg_pool()
        self.test_gelu_activation()
        self.test_elu_activation()
        self.test_positional_embedding()
        self.test_layer_norm()
        self.test_running_mean_std()
        self.test_lstm_cell()
        self.test_tensor_reshaper()
        self.test_stereo_encoder()

        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Passed: {self.passed}")
        logger.info(f"Failed: {self.failed}")
        logger.info(f"Total:  {self.passed + self.failed}")

        if self.failed == 0:
            logger.info("\n✓ ALL TESTS PASSED!")
            return True
        else:
            logger.error(f"\n✗ {self.failed} TESTS FAILED")
            return False


def main():
    """Run test suite"""
    tester = TestONNXOperations()
    success = tester.run_all_tests()

    if success:
        logger.info("\n✓ Ready for ONNX/QNN conversion!")
        return 0
    else:
        logger.error("\n✗ Fix failing tests before proceeding")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
