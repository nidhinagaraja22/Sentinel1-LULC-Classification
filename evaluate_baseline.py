"""Evaluate the selected winter baseline on the held-out test split.

Save beside train_baseline.py, lulc_dataset.py and lulc_model.py.
This script does not train or change model weights.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from train_baseline import WinterDataset, metrics
from lulc_model import SmallUNet


def fingerprint(path):
    """Identify exactly which file was evaluated, without loading it all into RAM."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Batch size must be positive.')
    root = Path(__file__).resolve().parent
    run = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    checkpoint_path = run / 'best_model.pt'
    manifest = root / 'data/metadata/test.csv'
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    dataset = WinterDataset(root, 'test', augment=False)
    if dataset.config != saved['class_configuration']:
        raise ValueError('Dataset configuration differs from the trained checkpoint.')
    classes = dataset.classes
    count = len(classes)
    if count != saved['num_classes'] or len(dataset) == 0:
        raise ValueError('Class count mismatch or empty test dataset.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallUNet(num_classes=count).to(device)
    model.load_state_dict(saved['model_state_dict'], strict=True)
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    confusion = torch.zeros((count, count), dtype=torch.int64, device=device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.ignore_index, reduction='sum')
    loss_total, valid_total, ignored_total = 0.0, 0, 0
    started = time.monotonic()
    print(f'Test patches: {len(dataset)} | device: {device} | checkpoint epoch: {saved["epoch"]}', flush=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            valid = targets != dataset.ignore_index
            n = int(valid.sum().item())
            ignored_total += targets.numel() - n
            if n:
                scores = model(images)
                if not torch.isfinite(scores).all():
                    raise RuntimeError('Non-finite model output encountered.')
                loss = criterion(scores, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite test loss encountered.')
                predictions = scores.argmax(1)
                codes = count * targets[valid] + predictions[valid]
                confusion += torch.bincount(codes, minlength=count ** 2).reshape(count, count)
                loss_total += loss.item()
                valid_total += n
            if step == 1 or step % 100 == 0 or step == len(loader):
                print(f'Test {step}/{len(loader)} | elapsed {(time.monotonic()-started)/60:.1f} min', flush=True)
    matrix = confusion.cpu()
    oa, miou, ious, f1s = metrics(matrix)
    output = run / datetime.now(timezone.utc).strftime('test_evaluation_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'test_confusion.npy', matrix.numpy())
    names = [c['name'] for c in classes]
    with (output / 'test_confusion.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['reference / predicted'] + names)
        for name, row in zip(names, matrix.tolist()):
            writer.writerow([name] + row)
    with (output / 'test_classes.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['class_id', 'class_name', 'reference_pixels', 'predicted_pixels', 'iou', 'f1'])
        for c in classes:
            i = c['id']
            writer.writerow([i, c['name'], int(matrix[i].sum()), int(matrix[:, i].sum()),
                             ious[i] if math.isfinite(ious[i]) else '',
                             f1s[i] if math.isfinite(f1s[i]) else ''])
    summary = {
        'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_epoch': saved['epoch'],
        'checkpoint_sha256': fingerprint(checkpoint_path),
        'test_manifest_sha256': fingerprint(manifest), 'test_patches': len(dataset),
        'valid_pixels': valid_total, 'ignored_pixels': ignored_total,
        'test_loss': loss_total / valid_total, 'overall_accuracy': oa, 'mean_iou': miou,
        'overall_accuracy_percent': 100 * oa, 'mean_iou_percent': 100 * miou,
        'scope': 'Winter-only nine-class baseline. Snow/Ice is excluded and unsupported.',
        'reference': 'Scores measure agreement with the dataset reference labels, which can be coarse or erroneous.',
        'metric_policy': 'Aggregate confusion counts across all valid test pixels. Rows are reference; columns predicted. IoU with zero union is undefined and excluded from mIoU. Undefined per-class scores are blank in CSV.',
    }
    (output / 'test_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    shutil.copy2(manifest, output / 'test.csv')
    shutil.copy2(Path(__file__), output / 'evaluate_baseline.py')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values = matrix.numpy().astype(float)
    totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0)
    fig, axis = plt.subplots(figsize=(10, 9))
    plot = axis.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    axis.set(xticks=range(count), yticks=range(count), xticklabels=names, yticklabels=names,
             xlabel='Predicted class', ylabel='Reference class', title='Test confusion matrix (row proportions)')
    plt.setp(axis.get_xticklabels(), rotation=60, ha='right')
    fig.colorbar(plot, ax=axis)
    fig.tight_layout()
    fig.savefig(output / 'test_confusion.png', dpi=150)
    plt.close(fig)
    print(f'\nTest OA: {oa:.2%}\nTest mIoU: {miou:.2%}')
    for name, iou, f1 in zip(names, ious, f1s):
        print(f'{name:17} IoU: {iou:.3f} | F1: {f1:.3f}')
    print('Saved evaluation:', output.resolve())


if __name__ == '__main__':
    main()
