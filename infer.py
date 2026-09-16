import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

from tone_mamba.config import INFERENCE_MODE, TOKEN_BATCH_SIZE

CSV_FIELDS = [
    'split',
    'sample_id',
    'syllable_index',
    'tone',
    'reference_tone',
    'prob_1',
    'prob_2',
    'prob_3',
    'prob_4',
    'prob_5',
]


def print_sample(split, sample_id, rows):
    print(f'{split} / {sample_id}')
    print('syllable tone  ref      p1      p2      p3      p4      p5')
    for row in rows:
        scores = ' '.join(
            f'{row[f"prob_{tone}"]:7.4f}'
            if row[f'prob_{tone}'] is not None
            else '      -'
            for tone in range(1, 6)
        )
        print(
            f'{row["syllable_index"]:8d} {row["tone"] or "-":>4} '
            f'{row["reference_tone"] or "-":>4} {scores}'
        )
    print(flush=True)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='local dataset directory; defaults to Hugging Face download',
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--mode', choices=['packing', 'single'], default=INFERENCE_MODE)
    parser.add_argument(
        '--token-batch-size',
        type=int,
        default=TOKEN_BATCH_SIZE,
        help='maximum F0 frames per packed batch (default: %(default)s)',
    )
    parser.add_argument(
        '--sample-index', type=int, help='run one sample (index after F0 filtering)'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=root / 'checkpoints/model.pt',
    )
    parser.add_argument(
        '--device', default='cuda:0', help='CUDA device (default: cuda:0)'
    )
    parser.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument(
        '--output',
        type=Path,
        help='packing mode: write CSV to this file instead of stdout',
    )
    args = parser.parse_args()
    if args.token_batch_size <= 0:
        parser.error('--token-batch-size must be positive')
    if args.mode == 'single' and args.output is not None:
        parser.error('--output is only supported in packing mode')
    return args, parser


def predict(model, samples, args):
    import torch

    from tone_mamba.dataset import collate_fn_packed, iter_batches

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for batch in iter_batches(samples, args.mode, args.token_batch_size):
        inputs = collate_fn_packed(batch)
        with (
            torch.inference_mode(),
            torch.autocast(
                'cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'
            ),
        ):
            logits = model(inputs.to(args.device))
        probabilities = logits.float().softmax(-1)[0].cpu()
        outputs = probabilities.split([len(sample.tones) for sample in batch])
        yield from zip(batch, outputs, strict=True)


def prediction_rows(sample, probabilities, split):
    rows = []
    for index, (probability, count, reference) in enumerate(
        zip(
            probabilities,
            sample.inputs.valid_f0_frames[0].tolist(),
            sample.tones.tolist(),
            strict=True,
        )
    ):
        prediction = int(probability.argmax()) + 1 if count >= 5 else None
        scores = probability.tolist() if count >= 5 else [None] * 5
        rows.append(
            {
                'split': split,
                'sample_id': sample.sample_id,
                'syllable_index': index,
                'tone': prediction,
                'reference_tone': reference + 1 if reference != -100 else None,
                **{f'prob_{tone}': score for tone, score in enumerate(scores, start=1)},
            }
        )
    return rows


def write_csv(rows, output):
    with (
        output.open('w', encoding='utf-8', newline='')
        if output
        else nullcontext(sys.stdout)
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(predictions, args):
    results = []
    sample_count = correct = evaluated = 0
    for sample, probabilities in predictions:
        sample_count += 1
        rows = prediction_rows(sample, probabilities, args.split)
        for row in rows:
            if row['reference_tone'] is not None:
                evaluated += 1
                correct += row['tone'] == row['reference_tone']
        if args.mode == 'single':
            print_sample(args.split, sample.sample_id, rows)
        else:
            results.extend(rows)
    if args.mode == 'packing':
        write_csv(results, args.output)
    accuracy = f'{correct / evaluated:.2%}' if evaluated else 'N/A'
    print(
        f'{args.split}: {sample_count} IPUs, {correct}/{evaluated} syllables correct '
        f'(accuracy: {accuracy})',
        file=sys.stderr,
    )


def main():
    args, parser = parse_args()

    import torch

    from tone_mamba.dataset import load_dataset
    from tone_mamba.model import load_model

    if torch.device(args.device).type != 'cuda' or not torch.cuda.is_available():
        parser.error('Mamba-2/Triton kernels require an NVIDIA CUDA GPU')
    samples = load_dataset(args.data_dir, args.split)
    if args.sample_index is not None:
        if not 0 <= args.sample_index < len(samples):
            parser.error(f'--sample-index must be between 0 and {len(samples) - 1}')
        samples = [samples[args.sample_index]]
    model = load_model(args.checkpoint, args.device)
    write_predictions(predict(model, samples, args), args)


if __name__ == '__main__':
    main()
