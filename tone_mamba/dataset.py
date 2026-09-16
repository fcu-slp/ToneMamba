import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from .config import INFERENCE_MODE, TOKEN_BATCH_SIZE
from .features import ToneBatch, prepare_input

DATASET_REPO = 'poyu39/FCU_Voice_47_EX'
DATASET_REVISION = 'main'
DATASET_FILES = [
    'csv/reset/train_filtered.csv',
    'csv/reset/val_filtered.csv',
    'csv/reset/test_filtered.csv',
    'f0/f0_praat.csv',
    'f0/f0_praat.npy',
]


@dataclass
class ToneSample:
    # T: frames; S: syllables in this IPU.
    sample_id: str  # IPU identifier
    inputs: ToneBatch  # U=1; f0 [1, T, 1], duration [1, S, 9]
    tones: torch.Tensor  # [S], labels 0-4; -100 ignored


def collate_fn_packed(samples: list[ToneSample]) -> ToneBatch:
    if not samples:
        raise ValueError('cannot pack an empty batch')
    if len(samples) == 1:
        return samples[0].inputs
    return ToneBatch(
        **{
            field.name: torch.cat(
                [getattr(sample.inputs, field.name) for sample in samples], dim=1
            )
            for field in fields(ToneBatch)
        }
    )


def iter_batches(
    samples: list[ToneSample],
    mode: str = INFERENCE_MODE,
    token_batch_size: int = TOKEN_BATCH_SIZE,
) -> Iterator[list[ToneSample]]:
    if mode not in ('packing', 'single'):
        raise ValueError('mode must be packing or single')
    if token_batch_size <= 0:
        raise ValueError('token_batch_size must be positive')
    if mode == 'single':
        for sample in samples:
            yield [sample]
        return
    batch = []
    frames = 0
    for sample in samples:
        length = sample.inputs.f0.shape[1]
        if batch and frames + length > token_batch_size:
            yield batch
            batch = []
            frames = 0
        batch.append(sample)
        frames += length
    if batch:
        yield batch


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def load_dataset(
    directory: str | Path | None = None, split: str = 'test'
) -> list[ToneSample]:
    if split not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    if directory is None:
        directory = snapshot_download(
            repo_id=DATASET_REPO,
            repo_type='dataset',
            revision=DATASET_REVISION,
            allow_patterns=DATASET_FILES,
        )
    directory = Path(directory)
    rows = _read_csv(directory / 'csv/reset' / f'{split}_filtered.csv')
    metadata = _read_csv(directory / 'f0/f0_praat.csv')
    contours = np.load(directory / 'f0/f0_praat.npy', allow_pickle=True)
    if len(metadata) != len(contours):
        raise ValueError('F0 CSV and NPY must contain the same number of rows')
    indices = {row['filename']: index for index, row in enumerate(metadata)}
    if len(indices) != len(metadata):
        raise ValueError('F0 artifact filenames must be unique')
    
    samples = []
    seen = set()
    for row in rows:
        name = f'{row["filename"]}_{int(row["index_sentence"])}'
        if name in seen:
            raise ValueError(f'duplicate split sample: {name}')
        seen.add(name)
        if name not in indices:
            raise ValueError(f'F0 artifact is missing sample: {name}')
        index = indices[name]
        pitch = np.asarray(contours[index], dtype=np.float32)
        stats = metadata[index]
        if int(stats['val_num']) != int(np.isfinite(pitch).sum()):
            raise ValueError(f'F0 val_num does not match contour: {name}')
        inputs = prepare_input(
            row,
            pitch,
            float(stats['min'] or 'nan'),
            float(stats['max'] or 'nan'),
        )
        if inputs is None:
            continue
        values = re.split(r'_|,', row['tone_word'])
        tones = torch.tensor(
            [
                int(tone) - 1 if tone in {'1', '2', '3', '4', '5'} else -100
                for tone in values
            ]
        )
        if len(re.split(r'[\s,]+', row['pinyin_word'].strip())) != len(tones):
            raise ValueError(f'pinyin and tone counts do not match: {name}')
        tones[inputs.valid_f0_frames[0] < 5] = -100
        if (tones != -100).any():
            samples.append(ToneSample(name, inputs, tones))
    return samples
