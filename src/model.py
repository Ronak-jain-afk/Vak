import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 384
    n_layers: int = 16
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden_dim: int = 1024
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True
    # Ablation variant flags
    norm_type: str = "rmsnorm"     # "rmsnorm" or "layernorm"
    ffn_type: str = "swiglu"       # "swiglu" or "gelu"
    pos_emb: str = "rope"          # "rope" or "learned"
    use_qk_norm: bool = True       # True or False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps).type_as(x) * self.weight


def create_norm(dim: int, norm_type: str = "rmsnorm", eps: float = 1e-5) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=False)
    return RMSNorm(dim, eps=eps)


def precompute_rope_frequencies(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    seq_len = x.shape[2]
    cos = cos[start_pos : start_pos + seq_len, :].to(dtype=x.dtype, device=x.device)
    sin = sin[start_pos : start_pos + seq_len, :].to(dtype=x.dtype, device=x.device)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.head_dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.gqa_kwargs = {"enable_gqa": self.n_rep > 1}

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        if cfg.use_qk_norm:
            self.q_norm = create_norm(self.head_dim, norm_type=cfg.norm_type, eps=cfg.norm_eps)
            self.k_norm = create_norm(self.head_dim, norm_type=cfg.norm_type, eps=cfg.norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if cos is not None and sin is not None:
            q = apply_rope(q, cos, sin, start_pos=start_pos)
            k = apply_rope(k, cos, sin, start_pos=start_pos)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k[:batch_size, :, start_pos : start_pos + seq_len, :] = k.type_as(cache_k)
            cache_v[:batch_size, :, start_pos : start_pos + seq_len, :] = v.type_as(cache_v)
            k = cache_k[:batch_size, :, : start_pos + seq_len, :].type_as(q)
            v = cache_v[:batch_size, :, : start_pos + seq_len, :].type_as(q)
            new_cache = (cache_k, cache_v)
        else:
            new_cache = None

        if seq_len > 1:
            if start_pos == 0:
                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **self.gqa_kwargs)
            else:
                total_k_len = start_pos + seq_len
                q_pos = torch.arange(seq_len, device=q.device).unsqueeze(1) + start_pos
                k_pos = torch.arange(total_k_len, device=k.device).unsqueeze(0)
                mask = q_pos >= k_pos
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **self.gqa_kwargs)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False, **self.gqa_kwargs)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_out), new_cache


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ffn_type = cfg.ffn_type
        if cfg.ffn_type == "swiglu":
            self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden_dim, bias=False)
            self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden_dim, bias=False)
            self.down_proj = nn.Linear(cfg.ffn_hidden_dim, cfg.d_model, bias=False)
        else:
            self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden_dim, bias=False)
            self.down_proj = nn.Linear(cfg.ffn_hidden_dim, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(F.gelu(self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = create_norm(cfg.d_model, norm_type=cfg.norm_type, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = create_norm(cfg.d_model, norm_type=cfg.norm_type, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_cache = self.attn(self.norm1(x), cos, sin, start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class VakModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)

        if cfg.pos_emb == "learned":
            self.pos_embeddings = nn.Embedding(cfg.max_seq_len, cfg.d_model)
            self.rope_cos = None
            self.rope_sin = None
        else:
            self.pos_embeddings = None
            cos, sin = precompute_rope_frequencies(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = create_norm(cfg.d_model, norm_type=cfg.norm_type, eps=cfg.norm_eps)

        if not cfg.tie_embeddings:
            # Untied embeddings ablation variant
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        else:
            self.lm_head = None

        self._init_weights()

    def _init_weights(self):
        residual_std = self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

        for layer in self.layers:
            nn.init.normal_(layer.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_kv_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        return [
            (
                torch.zeros(batch_size, self.cfg.n_kv_heads, self.cfg.max_seq_len, self.cfg.head_dim, device=device, dtype=dtype),
                torch.zeros(batch_size, self.cfg.n_kv_heads, self.cfg.max_seq_len, self.cfg.head_dim, device=device, dtype=dtype),
            )
            for _ in range(self.cfg.n_layers)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        batch_size, seq_len = input_ids.shape
        x = self.tok_embeddings(input_ids)

        if self.pos_embeddings is not None:
            pos_ids = torch.arange(start_pos, start_pos + seq_len, device=input_ids.device)
            x = x + self.pos_embeddings(pos_ids)

        cos = self.rope_cos
        sin = self.rope_sin

        new_caches = [] if kv_caches is not None else None
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = layer(x, cos, sin, start_pos=start_pos, kv_cache=layer_cache)
            if new_caches is not None:
                new_caches.append(updated_cache)

        x = self.norm(x)
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.tok_embeddings.weight)
        return logits, new_caches
