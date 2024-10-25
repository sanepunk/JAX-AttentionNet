import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Callable

class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int

    def setup(self):
        self.qkv_proj = nn.Dense(self.embed_dim * 3)
        self.out_proj = nn.Dense(self.embed_dim)

    def __call__(self, x):
        B, T, _ = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.embed_dim // self.num_heads)
        q, k, v = jnp.transpose(qkv, (2, 0, 3, 1, 4))

        attn_weights = jax.nn.softmax(jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(k.shape[-1]), axis=-1)
        out = jnp.einsum("bhqk,bhvd->bhqd", attn_weights, v).reshape(B, T, self.embed_dim)
        return self.out_proj(out)
