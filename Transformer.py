from Attention import MultiHeadSelfAttention
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    mlp_dim: int

    def setup(self):
        self.attn = MultiHeadSelfAttention(self.embed_dim, self.num_heads)
        self.mlp = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dense(self.embed_dim)
        ])
        self.layernorm1 = nn.LayerNorm()
        self.layernorm2 = nn.LayerNorm()

    def __call__(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
