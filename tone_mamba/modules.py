import torch
from einops import rearrange
from mamba_ssm import Mamba2 as Mamba2Block
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as BlockRMSNorm
from torch import nn
from torch.nn import functional as F

from .config import MODEL_CONFIG, ResetMambaModelConfig


class Mamba2ResidualBlock(nn.Module):
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.norm = BlockRMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mixer = Mamba2Block(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand_factor,
            headdim=cfg.d_head,
            chunk_size=cfg.chunk_size,
        )

    def forward(self, x, seq_idx, cu_seqlens):
        return self.mixer(self.norm(x), seq_idx=seq_idx, cu_seqlens=cu_seqlens) + x


class Mamba2(nn.Module):
    def __init__(self, n_layers: int, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.layers = nn.ModuleList([Mamba2ResidualBlock(cfg) for _ in range(n_layers)])

    def forward(self, x, lengths):
        lengths = lengths.to(torch.int32).reshape(-1)
        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=x.device)
        cu_seqlens[1:] = lengths.cumsum(0)
        ids = torch.arange(len(lengths), dtype=torch.int32, device=x.device)
        seq_idx = torch.repeat_interleave(ids, lengths).reshape(1, -1)
        for layer in self.layers:
            x = layer(x, seq_idx, cu_seqlens)
        return x


class GatedBiFusion(nn.Module):
    # channel-wise gated fusion of forward and backward representations
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.fusion_weight = nn.Parameter(torch.zeros(cfg.d_model, 2, 2))
        self.fusion_bias = nn.Parameter(torch.zeros(cfg.d_model, 2))
        self.gate_norm = nn.RMSNorm(cfg.d_model, eps=cfg.fusion_norm_eps)

    def forward(self, forward, backward):
        h = rearrange(torch.stack([forward, backward], dim=2), 'b t l c -> b t c l')
        h = h * (2**-0.5)
        fused = torch.einsum('btcl,cle->btce', h, self.fusion_weight) + self.fusion_bias
        return fused[..., 0] * F.silu(self.gate_norm(fused[..., 1]))


class GumbelSyllablePooling(nn.Module):
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.scorer = nn.Linear(cfg.d_model, 1)
        self.tau = cfg.pooling_temperature

    def forward(self, x, lengths):
        lengths = lengths.to(torch.int32).reshape(-1)
        ends = lengths.cumsum(0) - 1
        starts = ends - lengths + 1
        frames = torch.arange(x.shape[1], device=x.device).view(1, 1, -1)
        mask = (frames >= starts.view(1, -1, 1)) & (frames <= ends.view(1, -1, 1))
        logits = self.scorer(x).squeeze(-1).unsqueeze(1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        pooled = torch.einsum('bnt,btc->bnc', (logits / self.tau).softmax(-1), x)
        return pooled.float()


class ResetPitchEmbedding(nn.Module):
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.pitch_embeddings = nn.Linear(1, cfg.d_model)
        self.f0_scale = nn.Parameter(torch.tensor(1.0))
        self.token_embeddings = nn.Embedding(2, cfg.d_model)
        self.layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, f0, tokens):
        f0 = f0.masked_fill((tokens == 2).unsqueeze(-1), 0.0)
        x = self.pitch_embeddings(f0 * self.f0_scale)
        return self.layernorm(x + self.token_embeddings(tokens - 1))


class SyllableEmbeddingWithDuration(nn.Module):
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.duration_encoder = nn.Sequential(
            nn.Linear(cfg.rhythm_feature_dim, cfg.rhythm_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.rhythm_hidden_dim, cfg.d_model),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model), nn.Sigmoid()
        )
        self.layernorm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

    def forward(self, x, duration):
        rhythm = self.duration_encoder(duration)
        gate = self.fusion_gate(torch.cat([x, rhythm], dim=-1))
        return self.layernorm(x + gate * rhythm)
