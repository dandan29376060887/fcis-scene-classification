"""ScratchViT: a small Vision Transformer built from self-defined layers.

Satisfies the amended course requirement that at least one of the four
from-scratch models is a transformer (or a variant). Attention is
implemented manually from Linear layers and scaled dot-product softmax -
no ready-made transformer model classes are used, mirroring how the CNN
baselines stack primitive conv/pool/dense layers.

Architecture: image -> non-overlapping patches (a strided convolution),
flattened to tokens plus a learned [CLS] token and learned positional
embeddings, a stack of transformer encoder blocks (multi-head
self-attention + MLP, each with residual connections and LayerNorm),
then a linear head on the [CLS] token.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ImageClassifier


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product self-attention over image tokens."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"embed dim {dim} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm encoder block: attention residual + MLP residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ScratchViT(ImageClassifier):
    """Compact ViT: strided-conv patchify, encoder blocks, CLS head."""

    def __init__(
        self,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.1,
        patch_size: int = 16,
        depth: int = 4,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        image_size: int = 224,
    ) -> None:
        super().__init__(num_classes)
        dim = base_channels * 4  # token embedding width
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.pos_drop(torch.cat([cls, tokens], dim=1) + self._pos(tokens.shape[1]))
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))

    def _pos(self, num_tokens: int) -> torch.Tensor:
        # Slice supports inputs whose patch count differs from image_size
        # (e.g. the 32x32 tensors used in tests).
        return self.pos_embed[:, : num_tokens + 1]
