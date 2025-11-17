"""
ONNX-Compatible Model Implementations

This module provides ONNX-exportable versions of DEXTRAH models with
all operations converted to ONNX-compatible implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math

from .onnx_ops import (
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


class ONNXStandardTransform(nn.Module):
    """ONNX-compatible image normalization transform"""
    def __init__(self):
        super().__init__()
        # Register normalization parameters as buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class ONNXCustomCNN(nn.Module):
    """
    ONNX-compatible version of CustomCNN encoder.
    Uses ONNX-compatible LayerNorm.
    """
    def __init__(self, input_height, input_width):
        super().__init__()
        num_channel = 3

        # Calculate layer norm shapes
        h, w = input_height, input_width
        shapes = []
        for kernel, stride in [(6, 2), (4, 2), (4, 2), (4, 2)]:
            h = (h - kernel) // stride + 1
            w = (w - kernel) // stride + 1
            shapes.append((h, w))

        # Build CNN with ONNX-compatible operations
        self.conv1 = nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.ln1 = ONNXLayerNorm([16, shapes[0][0], shapes[0][1]])

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.ln2 = ONNXLayerNorm([32, shapes[1][0], shapes[1][1]])

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.ln3 = ONNXLayerNorm([64, shapes[2][0], shapes[2][1]])

        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0)
        self.relu4 = nn.ReLU()
        self.ln4 = ONNXLayerNorm([128, shapes[3][0], shapes[3][1]])

    def forward(self, x):
        x = self.ln1(self.relu1(self.conv1(x)))
        x = self.ln2(self.relu2(self.conv2(x)))
        x = self.ln3(self.relu3(self.conv3(x)))
        x = self.ln4(self.relu4(self.conv4(x)))
        return x


class ONNXResNetEncoder(nn.Module):
    """
    ONNX-compatible ResNet18 encoder with ONNX-compatible operations.
    """
    def __init__(self, input_height, input_width, train_resnet=False):
        super().__init__()

        # Load pretrained ResNet18
        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Replace adaptive pooling with ONNX-compatible version
        self.downproject = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            ONNXAdaptiveAvgPool2d((8, 16))
        )

        self.resnet18.fc = nn.Identity()
        self.resnet18.avgpool = self.downproject

        # Set to eval mode for export
        self.resnet18.eval()

        # Normalization transform
        self.transform = ONNXStandardTransform()

    def forward(self, x):
        x = self.transform(x)
        x = self.resnet18(x)
        return x


class ONNXConvNextEncoder(nn.Module):
    """
    ONNX-compatible ConvNeXt encoder.
    """
    def __init__(self, input_height, input_width, train_resnet=False):
        super().__init__()

        # Load pretrained ConvNeXt
        self.convnext = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        self.convnext.avgpool = nn.Identity()
        self.convnext.classifier = nn.Identity()
        self.convnext.eval()

        self.transform = ONNXStandardTransform()

        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=False),  # inplace=False for ONNX
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.transform(x)
        x = self.convnext(x)
        x = self.reduce_channels(x)
        return x.reshape(batch_size, 128, -1)


class ONNXCrossOnlyAttention(nn.Module):
    """
    ONNX-compatible cross-only attention mechanism for stereo vision.
    Uses explicit attention computation instead of F.scaled_dot_product_attention.

    Re-implemented Operation #7: Cross-Attention Mechanism
    """
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, T1=234, T2=234):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        self.T1 = T1
        self.T2 = T2

        # Key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # ONNX-compatible attention and dropout
        self.attention = ONNXScaledDotProductAttention(dropout_p=attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Cross-attention mask
        self.mask_generator = ONNXCrossAttentionMask(T1, T2)

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) where T = T1 + T2 + 1 (including embedding token)
        Returns:
            y: (B, T, n_embd)
        """
        B, T, C = x.size()
        head_size = C // self.n_head

        # Project to Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Get cross-attention mask
        cross_mask = self.mask_generator()

        # Apply ONNX-compatible attention
        y = self.attention(q, k, v, attn_mask=cross_mask)

        # Reassemble heads: (B, n_head, T, head_size) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class ONNXBlock(nn.Module):
    """ONNX-compatible Transformer block"""
    def __init__(self, n_embd, n_head, n_tokens, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = ONNXLayerNorm(n_embd)
        self.attn = ONNXCrossOnlyAttention(
            n_embd, n_head, attn_pdrop, resid_pdrop, T1=n_tokens, T2=n_tokens
        )
        self.ln2 = ONNXLayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            ONNXGELUActivation(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ONNXTransformer(nn.Module):
    """ONNX-compatible Transformer for stereo vision"""
    def __init__(
        self, in_dim, out_dim, ctx_len, n_embd, n_head, num_layer,
        attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ctx_len = ctx_len
        self.n_embd = n_embd

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, n_embd),
            nn.Dropout(resid_pdrop),
        )

        # ONNX-compatible positional embedding
        self.weight_pos_embed = ONNXPositionalEmbedding(ctx_len, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ONNXBlock(n_embd, n_head, ctx_len // 2, attn_pdrop, resid_pdrop)
            for _ in range(num_layer)
        ])

        self.output_layer = nn.Sequential(
            ONNXLayerNorm(n_embd),
            nn.Linear(n_embd, out_dim),
        )

        # Learnable embedding token
        self.embd_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            output: (B, T+1, out_dim) - includes embedding token
        """
        B = x.shape[0]

        # Project input
        x = self.input_layer(x)

        # Add positional embeddings
        pos_embeds = self.weight_pos_embed(self.ctx_len)
        x = x + pos_embeds.unsqueeze(0)

        # Prepend embedding token
        embd_token = self.embd_token.repeat(B, 1, 1)
        x = torch.cat([embd_token, x], dim=1)

        # Apply transformer blocks sequentially for ONNX compatibility
        for block in self.blocks:
            x = block(x)

        # Output projection
        x = self.output_layer(x)
        return x


class ONNXStereoEncoder(nn.Module):
    """
    ONNX-compatible Stereo Encoder combining CNN backbone + Transformer.
    This is the main vision encoder for the DEXTRAH policy.
    """
    def __init__(
        self, backbone, img_height, img_width, n_embd=None, n_head=4,
        attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.backbone = backbone

        # Select CNN backbone
        if backbone == "scratch":
            self.cnn = ONNXCustomCNN(img_height, img_width)
            n_embd = n_embd or 128
            num_tokens = 2 * 234
        elif backbone == "resnet":
            self.cnn = ONNXResNetEncoder(img_height, img_width)
            n_embd = n_embd or 128
            num_tokens = 2 * 128
        elif backbone == "convnext":
            self.cnn = ONNXConvNextEncoder(img_height, img_width)
            n_embd = n_embd or 48
            num_tokens = 2 * 128
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.num_tokens = num_tokens
        self.n_embd = n_embd
        self.out_embd = n_embd

        # Transformer for stereo fusion
        self.transformer = ONNXTransformer(
            n_embd, self.out_embd, num_tokens, n_embd, n_head, num_layer=2,
            attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop
        )

        # Output projection
        self.out_layer = nn.Sequential(
            nn.Linear(self.out_embd, 128),
            ONNXGELUActivation(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        """
        Args:
            x: (2*B, C, H, W) - stacked left and right images
        Returns:
            features: (B, 64) - stereo visual features
        """
        # Extract batch size (images come as [left1, left2, ..., right1, right2, ...])
        total_batch = x.shape[0]
        batch_size = total_batch // 2

        # Extract features from both images
        x = self.cnn(x)

        # Reshape for stereo processing
        if self.backbone == "convnext":
            # ConvNext already returns (B, 128, num_features)
            x = x.reshape(2, batch_size, -1, self.n_embd)
            x = x.permute(1, 0, 2, 3)  # (B, 2, num_features, n_embd)
            x = x.reshape(batch_size, -1, self.n_embd)
        else:
            # Standard CNN/ResNet: reshape from (2*B, C, H, W)
            x = x.view(2, batch_size, self.n_embd, -1)
            x = x.permute(1, 0, 2, 3)  # (B, 2, n_embd, spatial)
            x = x.reshape(batch_size, -1, self.n_embd)

        # Apply transformer
        x = self.transformer(x)

        # Use only the embedding token output
        x = self.out_layer(x[:, 0, :])

        return x


class ONNXLSTMSequential(nn.Module):
    """
    ONNX-compatible LSTM with sequential processing.
    For ONNX export, we handle state externally or unroll for fixed sequences.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use standard LSTM for ONNX compatibility
        # Note: ONNX supports nn.LSTM well, but custom LSTM variants need special handling
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden_state=None):
        """
        Args:
            x: (B, seq_len, input_size)
            hidden_state: tuple of (h, c) each (num_layers, B, hidden_size)
        Returns:
            output: (B, seq_len, hidden_size)
            hidden_state: tuple of (h, c) each (num_layers, B, hidden_size)
        """
        if hidden_state is None:
            output, hidden_state = self.lstm(x)
        else:
            output, hidden_state = self.lstm(x, hidden_state)

        return output, hidden_state


class ONNXPolicyMLP(nn.Module):
    """ONNX-compatible MLP for policy network"""
    def __init__(self, input_size, units, activation='elu'):
        super().__init__()
        layers = []
        in_size = input_size

        for unit in units:
            layers.append(nn.Linear(in_size, unit))
            if activation == 'elu':
                layers.append(ONNXELUActivation())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(ONNXGELUActivation())
            in_size = unit

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
