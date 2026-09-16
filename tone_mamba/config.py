from dataclasses import dataclass


@dataclass(frozen=True)
class ResetMambaModelConfig:
    d_model: int = 256
    d_logits: int = 5
    contour_layers: int = 8
    interaction_layers: int = 4
    
    d_head: int = 32
    d_state: int = 64
    expand_factor: int = 2
    d_conv: int = 4
    chunk_size: int = 256
    
    rms_norm_eps: float = 1e-5
    fusion_norm_eps: float | None = None
    pooling_temperature: float = 1.0
    rhythm_feature_dim: int = 9
    rhythm_hidden_dim: int = 128
    
    contour_scan_levels: tuple[str, ...] = ('utterance', 'word', 'syllable')
    syllable_scan_levels: tuple[str, ...] = ('utterance', 'word')
    contour_level_weights: tuple[float, ...] = (1.0, 2.0, 3.0)
    syllable_level_weights: tuple[float, ...] = (1.0, 2.0)


MODEL_CONFIG = ResetMambaModelConfig()
INFERENCE_MODE = 'packing'
TOKEN_BATCH_SIZE = 80_000
