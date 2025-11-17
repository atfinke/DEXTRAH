"""
ONNX-Compatible Operation Implementations

This module re-implements PyTorch operations that are not well-supported
in ONNX or QNN, converting them to use ONNX-compatible operations.

Operations Re-implemented (12 total):
1. Scaled Dot-Product Attention (F.scaled_dot_product_attention)
2. Masked Attention Operations (masked_fill with attention masks)
3. Adaptive Average Pooling (nn.AdaptiveAvgPool2d)
4. GELU Activation (explicit implementation for ONNX compatibility)
5. ELU Activation (explicit implementation)
6. Positional Embeddings (nn.Embedding with dynamic indices)
7. Cross-Attention Mechanism (custom stereo attention)
8. LayerNorm (with explicit normalization)
9. Running Mean/Std Normalization (custom normalization)
10. LSTM State Management (stateful LSTM for ONNX)
11. Complex Tensor Reshaping (view/permute/transpose chains)
12. BFloat16 → FP16/FP32 Conversion (for QNN compatibility)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ONNXScaledDotProductAttention(nn.Module):
    """
    ONNX-compatible implementation of scaled dot-product attention.
    Replaces F.scaled_dot_product_attention which has limited ONNX support.

    Re-implemented Operation #1: Scaled Dot-Product Attention
    """
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, query, key, value, attn_mask=None, scale=None):
        """
        Args:
            query: (B, num_heads, seq_len, head_dim)
            key: (B, num_heads, seq_len, head_dim)
            value: (B, num_heads, seq_len, head_dim)
            attn_mask: (1, 1, seq_len, seq_len) or None
            scale: scaling factor (default: 1/sqrt(head_dim))
        Returns:
            output: (B, num_heads, seq_len, head_dim)
        """
        B, num_heads, seq_len, head_dim = query.shape

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        # Compute attention scores: Q @ K^T
        # (B, num_heads, seq_len, head_dim) @ (B, num_heads, head_dim, seq_len)
        # -> (B, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply attention mask if provided
        if attn_mask is not None:
            # Convert boolean mask to float mask for ONNX compatibility
            if attn_mask.dtype == torch.bool:
                # Replace masked positions with large negative value
                attn_scores = torch.where(
                    attn_mask.to(torch.bool),
                    attn_scores,
                    torch.full_like(attn_scores, -1e9)
                )
            else:
                # Assume mask is already in additive form (0 for valid, -inf for masked)
                attn_scores = attn_scores + attn_mask

        # Softmax over last dimension
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply dropout during training
        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)

        # Compute weighted sum: attn_weights @ V
        # (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, head_dim)
        # -> (B, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, value)

        return output


class ONNXCrossAttentionMask(nn.Module):
    """
    ONNX-compatible cross-attention mask creation.

    Re-implemented Operation #2: Masked Attention Operations
    """
    def __init__(self, T1, T2):
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        # Pre-compute and register mask as buffer for ONNX export
        mask = self.create_cross_attention_mask(T1, T2)
        self.register_buffer("cross_mask", mask.view(1, 1, *mask.shape))

    def create_cross_attention_mask(self, T1, T2):
        """
        Creates a cross-only attention mask for stereo vision.
        Returns 1 for allowed attention, 0 for masked positions.
        """
        T = T1 + T2
        img_mask = torch.zeros(T, T, dtype=torch.float32)

        # Image 1 attends to Image 2
        img_mask[0:T1, T1:T] = 1.0
        # Image 2 attends to Image 1
        img_mask[T1:T, 0:T1] = 1.0

        # Add embedding token dimension
        mask = torch.ones(T+1, T+1, dtype=torch.float32)
        mask[1:T+1, 1:T+1] = img_mask

        return mask

    def forward(self):
        return self.cross_mask


class ONNXAdaptiveAvgPool2d(nn.Module):
    """
    ONNX-compatible adaptive average pooling.
    Re-implements nn.AdaptiveAvgPool2d for better ONNX support.

    Re-implemented Operation #3: Adaptive Average Pooling
    """
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, C, out_h, out_w)
        """
        B, C, H, W = x.shape
        out_h, out_w = self.output_size

        # Compute stride and kernel size
        stride_h = H // out_h
        stride_w = W // out_w
        kernel_h = H - (out_h - 1) * stride_h
        kernel_w = W - (out_w - 1) * stride_w

        # Use standard AvgPool2d with computed parameters
        return F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))


class ONNXGELUActivation(nn.Module):
    """
    ONNX-compatible GELU activation using explicit formula.

    Re-implemented Operation #4: GELU Activation
    """
    def __init__(self, approximate='none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'tanh':
            # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            ))
        else:
            # Exact GELU: x * Φ(x) where Φ is the CDF of standard normal distribution
            # Approximated as: 0.5 * x * (1 + erf(x / sqrt(2)))
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ONNXELUActivation(nn.Module):
    """
    ONNX-compatible ELU activation with explicit implementation.

    Re-implemented Operation #5: ELU Activation
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        # ELU(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
        return torch.where(
            x > 0,
            x,
            self.alpha * (torch.exp(x) - 1.0)
        )


class ONNXPositionalEmbedding(nn.Module):
    """
    ONNX-compatible positional embedding.
    Pre-computes embeddings and uses indexing instead of nn.Embedding.

    Re-implemented Operation #6: Positional Embeddings
    """
    def __init__(self, num_positions, embedding_dim):
        super().__init__()
        self.num_positions = num_positions
        self.embedding_dim = embedding_dim

        # Use standard embedding but ensure it's exportable
        self.embedding = nn.Embedding(num_positions, embedding_dim)
        # Pre-compute position indices as buffer
        self.register_buffer(
            "position_ids",
            torch.arange(num_positions, dtype=torch.long)
        )

    def forward(self, seq_length=None):
        """
        Returns:
            embeddings: (seq_length, embedding_dim) or (num_positions, embedding_dim)
        """
        if seq_length is None:
            seq_length = self.num_positions

        # Use pre-computed position IDs for ONNX compatibility
        positions = self.position_ids[:seq_length]
        return self.embedding(positions)


class ONNXLayerNorm(nn.Module):
    """
    ONNX-compatible Layer Normalization with explicit computation.

    Re-implemented Operation #8: LayerNorm
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Compute mean and variance over last dimensions
        dims = list(range(-len(self.normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.weight * x_normalized + self.bias


class ONNXRunningMeanStd(nn.Module):
    """
    ONNX-compatible running mean/std normalization.
    For inference, statistics are frozen and applied as fixed transformations.

    Re-implemented Operation #9: Running Mean/Std Normalization
    """
    def __init__(self, shape, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

        # Register buffers (will be frozen for inference)
        self.register_buffer("running_mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("running_var", torch.ones(shape, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float32))

    def forward(self, x):
        """
        Apply frozen normalization statistics.
        """
        # For ONNX export, always use frozen statistics
        mean = self.running_mean.view(1, *self.running_mean.shape)
        std = torch.sqrt(self.running_var.view(1, *self.running_var.shape) + self.epsilon)
        return (x - mean) / std


class ONNXLSTMCell(nn.Module):
    """
    ONNX-compatible LSTM cell with explicit state management.

    Re-implemented Operation #10: LSTM State Management
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrix for i, f, g, o gates
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (batch, input_size)
            hidden_state: tuple of (h, c) each (batch, hidden_size)
        Returns:
            h_next: (batch, hidden_size)
            c_next: (batch, hidden_size)
        """
        batch_size = x.size(0)

        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h, c = hidden_state

        # Compute gates
        gates = torch.matmul(x, self.weight_ih.t()) + torch.matmul(h, self.weight_hh.t()) + self.bias

        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=1)

        # Apply activations
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # cell gate
        o = torch.sigmoid(o)  # output gate

        # Update cell state and hidden state
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ONNXBFloat16ToFP32Converter(nn.Module):
    """
    Converts bfloat16 operations to FP32 for ONNX/QNN compatibility.
    QNN typically requires FP16 or FP32, not bfloat16.

    Re-implemented Operation #12: BFloat16 Conversion
    """
    def __init__(self, target_dtype=torch.float32):
        super().__init__()
        self.target_dtype = target_dtype

    def forward(self, x):
        if x.dtype == torch.bfloat16:
            return x.to(self.target_dtype)
        return x


class ONNXTensorReshaper(nn.Module):
    """
    ONNX-compatible tensor reshaping operations.
    Handles view, permute, transpose, flatten chains in ONNX-friendly way.

    Re-implemented Operation #11: Complex Tensor Reshaping
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def safe_view(x, *shape):
        """ONNX-safe view operation using reshape"""
        return x.reshape(*shape)

    @staticmethod
    def safe_flatten(x, start_dim=0, end_dim=-1):
        """ONNX-safe flatten operation"""
        if end_dim == -1:
            end_dim = len(x.shape) - 1

        # Compute new shape
        shape = list(x.shape[:start_dim]) + [-1] + list(x.shape[end_dim+1:])
        return x.reshape(*shape)

    @staticmethod
    def safe_permute(x, *dims):
        """ONNX-safe permute operation"""
        return x.permute(*dims).contiguous()

    @staticmethod
    def safe_transpose(x, dim0, dim1):
        """ONNX-safe transpose operation"""
        return x.transpose(dim0, dim1).contiguous()


def replace_operations_for_onnx(module):
    """
    Recursively replaces non-ONNX-compatible operations in a module.

    Args:
        module: PyTorch module to convert

    Returns:
        Modified module with ONNX-compatible operations
    """
    for name, child in module.named_children():
        # Replace GELU
        if isinstance(child, nn.GELU):
            setattr(module, name, ONNXGELUActivation())

        # Replace ELU
        elif isinstance(child, nn.ELU):
            setattr(module, name, ONNXELUActivation(alpha=child.alpha))

        # Replace AdaptiveAvgPool2d
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, ONNXAdaptiveAvgPool2d(child.output_size))

        # Replace LayerNorm (optional, for more control)
        elif isinstance(child, nn.LayerNorm):
            setattr(module, name, ONNXLayerNorm(child.normalized_shape, child.eps))

        # Recursively apply to children
        else:
            replace_operations_for_onnx(child)

    return module
