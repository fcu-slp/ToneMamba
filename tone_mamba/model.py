from pathlib import Path

import torch
from torch import nn

from .config import MODEL_CONFIG, ResetMambaModelConfig
from .features import ToneBatch
from .modules import (
    GatedBiFusion,
    GumbelSyllablePooling,
    Mamba2,
    ResetPitchEmbedding,
    SyllableEmbeddingWithDuration,
)


class ToneNetResetBiMamba2WithRhythm(nn.Module):
    def __init__(self, cfg: ResetMambaModelConfig = MODEL_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.contour_encoder = Mamba2(cfg.contour_layers, cfg)
        self.interaction_encoder = Mamba2(cfg.interaction_layers, cfg)
        self.contour_context_fusion = GatedBiFusion(cfg)
        self.interaction_context_fusion = GatedBiFusion(cfg)
        
        self.register_buffer(
            'contour_weight_logits',
            torch.tensor(cfg.contour_level_weights, dtype=torch.float32),
        )
        
        self.register_buffer(
            'syllable_weight_logits',
            torch.tensor(cfg.syllable_level_weights, dtype=torch.float32),
        )
        
        self.syllable_pool = GumbelSyllablePooling(cfg)
        self.pitch_embedding = ResetPitchEmbedding(cfg)
        self.syllable_embedding = SyllableEmbeddingWithDuration(cfg)
        
        self.projection_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Identity(), nn.Linear(cfg.d_model, cfg.d_logits)
        )
    
    @staticmethod
    def _encode(
        x: torch.Tensor,
        encoder: Mamba2,
        fusion: GatedBiFusion,
        lengths_by_level: tuple[torch.Tensor, ...],
        weight_logits: torch.Tensor,
    ):
        backward = x.flip(1)
        result = None
        for weight, lengths in zip(
            weight_logits.softmax(0), lengths_by_level, strict=True
        ):
            forward = encoder(x, lengths)
            reverse = encoder(backward, lengths.flip(-1)).flip(1)
            level = weight * fusion(forward, reverse)
            result = level if result is None else result + level
        return result
    
    @torch.inference_mode()
    def forward(self, batch: ToneBatch) -> torch.Tensor:
        x = self.pitch_embedding(batch.f0, batch.tokens)
        x = self._encode(
            x,
            self.contour_encoder,
            self.contour_context_fusion,
            tuple(
                getattr(batch, f'{level}_len') for level in self.cfg.contour_scan_levels
            ),
            self.contour_weight_logits,
        )
        x = self.syllable_pool(x, batch.syllable_len)
        x = self.syllable_embedding(x, batch.duration)
        x = self._encode(
            x,
            self.interaction_encoder,
            self.interaction_context_fusion,
            tuple(
                getattr(batch, f'{level}_inter_len')
                for level in self.cfg.syllable_scan_levels
            ),
            self.syllable_weight_logits,
        )
        return self.classifier(self.projection_head(x))


def load_model(
    checkpoint: str | Path,
    device: str = 'cuda:0',
    cfg: ResetMambaModelConfig = MODEL_CONFIG,
) -> ToneNetResetBiMamba2WithRhythm:
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ToneNetResetBiMamba2WithRhythm(cfg)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval()
