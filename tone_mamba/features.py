import re
from dataclasses import dataclass, fields

import numpy as np
import torch


@dataclass
class ToneBatch:
    # Shapes after unsqueeze(0); the batch dimension is 1 for single and packing.
    # T: total frames; S: total syllables; W: total words; U: number of IPUs.
    
    f0: torch.Tensor                    # [1, T, 1]
    tokens: torch.Tensor                # [1, T]
    duration: torch.Tensor              # [1, S, 9]
    utterance_len: torch.Tensor         # [1, U], frames per IPU
    utterance_inter_len: torch.Tensor   # [1, U], syllables per IPU
    syllable_len: torch.Tensor          # [1, S], frames per syllable
    word_len: torch.Tensor              # [1, W], frames per word
    word_inter_len: torch.Tensor        # [1, W], syllables per word
    valid_f0_frames: torch.Tensor       # [1, S], valid F0 frames per syllable

    @classmethod
    def as_tensor(cls, **values) -> 'ToneBatch':
        return cls(**{name: torch.as_tensor(value) for name, value in values.items()})
    
    def to(self, device: str) -> 'ToneBatch':
        return ToneBatch(
            **{
                field.name: getattr(self, field.name).to(device)
                for field in fields(self)
            }
        )
    
    def unsqueeze(self, dim: int) -> 'ToneBatch':
        return ToneBatch(
            **{
                field.name: getattr(self, field.name).unsqueeze(dim)
                for field in fields(self)
            }
        )


def duration_features(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy()
    missing = ~np.isfinite(values)
    values[missing] = values[~missing].mean() if (~missing).any() else 0.0
    centered = values - values.mean() if len(values) > 1 else np.zeros_like(values)
    variability = np.zeros_like(values)
    if len(values) > 1:
        denominator = (values[:-1] + values[1:]) / 2
        variability[:-1] = np.divide(
            values[:-1] - values[1:],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        variability[-1] = np.abs(variability[:-1]).mean()
    return np.stack([values, centered, variability], axis=1)


def _numbers(value: str) -> np.ndarray:
    return np.asarray(
        [float(item) for item in re.split(r'_|,', value)], dtype=np.float32
    )


def _durations(value: str) -> np.ndarray:
    return np.asarray(
        [float(item) if item else np.nan for item in value.split(',')], dtype=np.float32
    )


def prepare_input(row: dict, f0_hz, minimum: float, maximum: float) -> ToneBatch | None:
    values = np.asarray(f0_hz, dtype=np.float32)
    valid = np.isfinite(values) & (values >= minimum) & (values <= maximum)
    
    if valid.sum() < 5:
        return None
    
    if not 0 < minimum < maximum < np.inf:
        raise ValueError('speaker F0 range must be finite, positive and increasing')
    
    start, end = float(row['starttime_sentence']), float(row['endtime_sentence'])
    starts, ends = _numbers(row['starttime_word']), _numbers(row['endtime_word'])
    counts = np.asarray([len(word.split('_')) for word in row['tone_word'].split(',')])
    n = len(starts)
    
    if not n or ends.shape != starts.shape or counts.sum() != n:
        raise ValueError('syllable times and tone_word must have matching counts')
    
    indices = np.floor((starts - start) / (end - start) * len(values)).astype(np.int64)
    indices = np.clip(indices, 0, len(values) - 1)
    
    if indices[0] != 0 or np.any(np.diff(indices) <= 0):
        raise ValueError(
            'the first syllable must start at IPU frame 0; later starts must advance'
        )
    
    lengths = np.diff(np.append(indices, len(values)))
    word_lengths = np.asarray(
        [part.sum() for part in np.split(lengths, counts.cumsum()[:-1])]
    )
    
    normalized = np.zeros_like(values)
    normalized[valid] = (np.log1p(values[valid]) - np.log1p(minimum)) / (
        np.log1p(maximum) - np.log1p(minimum)
    )
    duration = [
        duration_features(feature)
        for feature in (
            np.nan_to_num(ends - starts),
            _durations(row['duration_final']),
            _durations(row['duration_v']),
        )
    ]
    valid_counts = [part.sum() for part in np.split(valid, indices[1:])]
    
    return ToneBatch.as_tensor(
        f0                  = normalized[:, None],
        tokens              = (~valid).astype(np.int64) + 1,
        duration            = np.concatenate(duration, axis=-1),
        utterance_len       = [len(values)],
        utterance_inter_len = [n],
        syllable_len        = lengths,
        word_len            = word_lengths,
        word_inter_len      = counts,
        valid_f0_frames     = valid_counts,
    ).unsqueeze(0)
