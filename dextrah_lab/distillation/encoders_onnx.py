"""
ONNX-compatible implementations of vision encoders.
This ensures all operations are ONNX-exportable without custom ops.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def conv_output_size(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """Utility function to compute the output size of a convolution layer."""
    if isinstance(kernel_size, tuple):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h, kernel_w = kernel_size, kernel_size

    if isinstance(stride, tuple):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride, stride

    if isinstance(pad, tuple):
        pad_h, pad_w = pad
    else:
        pad_h, pad_w = pad, pad

    h = (h_w[0] + 2 * pad_h - dilation * (kernel_h - 1) - 1) // stride_h + 1
    w = (h_w[1] + 2 * pad_w - dilation * (kernel_w - 1) - 1) // stride_w + 1
    return h, w


class CustomCNNONNX(nn.Module):
    """ONNX-compatible Custom CNN Encoder"""

    def __init__(self, input_height, input_width):
        super().__init__()
        num_channel = 3

        # Initial input dimensions
        h, w = input_height, input_width

        # Layer 1
        h, w = conv_output_size((h, w), kernel_size=6, stride=2)
        layer1_norm_shape = [16, h, w]

        # Layer 2
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer2_norm_shape = [32, h, w]

        # Layer 3
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer3_norm_shape = [64, h, w]

        # Layer 4
        h, w = conv_output_size((h, w), kernel_size=4, stride=2)
        layer4_norm_shape = [128, h, w]

        # CNN definition
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channel, 16, kernel_size=6, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer1_norm_shape),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer2_norm_shape),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer3_norm_shape),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LayerNorm(layer4_norm_shape),
        )

    def forward(self, x):
        return self.cnn(x)


class ResnetEncoderONNX(nn.Module):
    """ONNX-compatible ResNet18 Encoder"""

    def __init__(self, input_height, input_width):
        super().__init__()

        # Load pre-trained ResNet18
        self.resnet18 = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )

        # Downproject layer to reduce channels and spatial dimensions
        self.downproject = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 16))
        )

        # Remove fc and replace avgpool
        self.resnet18.fc = nn.Identity()
        self.resnet18.avgpool = self.downproject

        # Normalization parameters (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Normalize
        x = (x - self.mean) / self.std
        # Forward through ResNet
        return self.resnet18(x)


class ConvNextEncoderONNX(nn.Module):
    """ONNX-compatible ConvNeXt Tiny Encoder"""

    def __init__(self, input_height, input_width):
        super().__init__()

        # Load pre-trained ConvNeXt
        self.convnext = torchvision.models.convnext_tiny(
            weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        # Remove avgpool and classifier
        self.convnext.avgpool = nn.Identity()
        self.convnext.classifier = nn.Identity()

        # Channel reduction
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Normalization parameters (ImageNet)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        # Normalize
        x = (x - self.mean) / self.std
        # Forward through ConvNeXt
        convnext_out = self.convnext(x)
        # Reduce channels
        out = self.reduce_channels(convnext_out)
        # Reshape to (B, 128, num_tokens)
        return out.reshape(x.shape[0], 128, -1)


class SquaredReLUONNX(nn.Module):
    """ONNX-compatible SquaredReLU activation"""

    def forward(self, x):
        return F.relu(x).pow(2)


class CrossOnlyAttentionONNX(nn.Module):
    """
    ONNX-compatible Cross-Only Attention.
    Uses standard PyTorch operations without custom masking logic.
    """

    def __init__(
        self,
        n_embd,
        n_head,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        T1=234,
        T2=234
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.T1 = T1
        self.T2 = T2

        # Key, query, value projections
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropouts
        self.attn_pdrop = attn_pdrop
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        # Precompute the cross-only mask
        mask_2d = self.create_cross_attention_mask(T1, T2)
        # Register as buffer so it's part of state_dict and moves with model
        self.register_buffer("cross_mask", mask_2d.view(1, 1, *mask_2d.shape))

    def create_cross_attention_mask(self, T1, T2):
        """
        Create a mask for cross-only attention.
        Returns tensor of shape (T+1, T+1) with 1s for allowed positions, 0s for masked.
        """
        T = T1 + T2
        img_mask = torch.zeros(T, T)

        # Image 1 attends to Image 2
        img_mask[0:T1, T1:T] = 1
        # Image 2 attends to Image 1
        img_mask[T1:T, 0:T1] = 1

        # Full mask including embedding token
        mask = torch.ones(T+1, T+1)
        mask[1:T+1, 1:T+1] = img_mask

        return mask

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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))

        # Apply cross-only mask (convert 0s to -inf for masked positions)
        # For ONNX compatibility, avoid in-place operations
        mask = self.cross_mask
        att = att + (1.0 - mask) * (-1e9)

        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum over values
        y = att @ v

        # Reassemble
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class BlockONNX(nn.Module):
    """ONNX-compatible Transformer Block"""

    def __init__(self, n_embd, n_head, n_tokens, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CrossOnlyAttentionONNX(
            n_embd, n_head, attn_pdrop,
            resid_pdrop, T1=n_tokens, T2=n_tokens
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerONNX(nn.Module):
    """ONNX-compatible Transformer"""

    def __init__(
        self, in_dim, out_dim, ctx_len, n_embd, n_head, num_layer,
        attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ctx_len = ctx_len
        self.n_embd = n_embd
        self.n_head = n_head
        self.num_layer = num_layer

        self.input_layer = nn.Sequential(
            nn.Linear(in_dim, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.weight_pos_embed = nn.Embedding(ctx_len, n_embd)

        self.blocks = nn.Sequential(
            *[
                BlockONNX(
                    n_embd, n_head, ctx_len // 2,
                    attn_pdrop, resid_pdrop
                )
                for _ in range(num_layer)
            ],
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, out_dim),
        )

        self.embd_token = nn.Parameter(torch.randn(1, 1, n_embd))

    def forward(self, x):
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            y: (B, T+1, out_dim)
        """
        B = x.shape[0]

        # Input projection
        x = self.input_layer(x)

        # Add positional embeddings
        pos_embeds = self.weight_pos_embed(
            torch.arange(self.ctx_len, device=x.device, dtype=torch.long)
        ).unsqueeze(0)
        x = x + pos_embeds

        # Prepend embedding token
        x = torch.cat([self.embd_token.repeat(B, 1, 1), x], dim=1)

        # Transformer blocks
        x = self.blocks(x)

        # Output layer
        x = self.output_layer(x)

        return x


class MonoEncoderONNX(nn.Module):
    """ONNX-compatible Monocular Encoder"""

    def __init__(
        self, backbone, img_height, img_width, n_embd, n_head,
        attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.backbone = backbone

        # Model settings
        MODEL_SETTINGS = {
            "scratch": {
                "n_embd": 128,
                "num_tokens": 234,
                "model": CustomCNNONNX,
            },
            "resnet": {
                "n_embd": 128,
                "num_tokens": 128,
                "model": ResnetEncoderONNX,
            },
            "convnext": {
                "n_embd": 48,
                "num_tokens": 128,
                "model": ConvNextEncoderONNX,
            },
        }

        self.cnn = MODEL_SETTINGS[backbone]["model"](img_height, img_width)
        self.num_tokens = MODEL_SETTINGS[backbone]["num_tokens"]

        if n_embd is None:
            n_embd = MODEL_SETTINGS[backbone]["n_embd"]

        self.out_embd = n_embd
        self.n_embd = n_embd

        self.transformer = TransformerONNX(
            n_embd, self.out_embd, self.num_tokens, n_embd, n_head, 2,
            attn_pdrop, resid_pdrop
        )

        self.out_layer = nn.Sequential(
            nn.Linear(self.out_embd, 128),
            nn.GELU(),
            nn.Linear(128, 32),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) - Input images
        Returns:
            y: (B, 32) - Output embeddings
        """
        batch_size = x.shape[0]

        # CNN feature extraction
        x = self.cnn(x)

        # Reshape for transformer
        if self.backbone == "convnext":
            x = x.reshape(batch_size, self.n_embd, -1).transpose(1, 2)
        else:
            x = x.view(batch_size, self.n_embd, -1).transpose(1, 2)

        # Transformer
        x = self.transformer(x)

        # Output layer (use only embedding token)
        x = self.out_layer(x[:, 0, :])

        return x


class StereoEncoderONNX(nn.Module):
    """ONNX-compatible Stereo Encoder"""

    def __init__(
        self, backbone, img_height, img_width, n_embd, n_head,
        attn_pdrop=0.1, resid_pdrop=0.1
    ):
        super().__init__()
        self.backbone = backbone

        # Model settings
        MODEL_SETTINGS = {
            "scratch": {
                "n_embd": 128,
                "num_tokens": 2 * 234,
                "model": CustomCNNONNX,
            },
            "resnet": {
                "n_embd": 128,
                "num_tokens": 2 * 128,
                "model": ResnetEncoderONNX,
            },
            "convnext": {
                "n_embd": 48,
                "num_tokens": 2 * 128,
                "model": ConvNextEncoderONNX,
            },
        }

        self.cnn = MODEL_SETTINGS[backbone]["model"](img_height, img_width)
        self.num_tokens = MODEL_SETTINGS[backbone]["num_tokens"]

        if n_embd is None:
            n_embd = MODEL_SETTINGS[backbone]["n_embd"]

        self.out_embd = n_embd
        self.n_embd = n_embd

        self.transformer = TransformerONNX(
            n_embd, self.out_embd, self.num_tokens, n_embd, n_head, 2,
            attn_pdrop, resid_pdrop
        )

        self.out_layer = nn.Sequential(
            nn.Linear(self.out_embd, 128),
            nn.GELU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        """
        Args:
            x: (2*B, 3, H, W) - Stacked left and right images
        Returns:
            y: (B, 64) - Output embeddings
        """
        batch_size = x.shape[0] // 2

        # CNN feature extraction
        x = self.cnn(x)

        # Reshape for transformer
        if self.backbone == "convnext":
            x = x.reshape(2, batch_size, -1, self.n_embd)
            x = x.permute(1, 0, 2, 3).reshape(batch_size, -1, self.n_embd)
        else:
            x = x.view(2, batch_size, self.n_embd, -1)
            x = x.permute(1, 0, 2, 3).reshape(batch_size, -1, self.n_embd)

        # Transformer
        x = self.transformer(x)

        # Output layer (use only embedding token)
        x = self.out_layer(x[:, 0, :])

        return x
