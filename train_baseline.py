"""Train the winter-only nine-class baseline. Run from the project folder.

Requires the previously created lulc_dataset.py, lulc_model.py and metadata.
Snow/Ice is excluded; this model cannot identify snow-covered areas reliably.
This script reads train/validation data only and preserves earlier experiments.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import csv
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from lulc_dataset import LULCDataset
from lulc_model import SmallUNet


class WinterDataset(LULCDataset):
    """Apply a separate nine-class configuration without editing classes.json."""
    def __init__(self, project_dir, split, augment=False):
        super().__init__(project_dir, split=split, augment=augment)
        self.config = copy.deepcopy(self.config)
        self.classes = [copy.deepcopy(c) for c in self.classes
                        if c['name'] != 'Snow/Ice']
        for index, definition in enumerate(self.classes):
            definition['id'] = index
        self.config['classes'] = self.classes
        self.config['ignored_source_ids'] = sorted(
            set(self.config['ignored_source_ids']) | {15})
        self.config['classes_without_training_pixels'] = []
        self.config['scope'] = 'winter baseline; Snow/Ice excluded and unsupported'
        self.ignored_source_ids = set(self.config['ignored_source_ids'])
        self.mapping = {raw: c['id'] for c in self.classes
                        for raw in c['source_ids']}


def metrics(matrix):
    """Aggregate pixels before computing OA, IoU and F1."""
    m = matrix.double()
    tp = m.diag()
    actual, predicted = m.sum(1), m.sum(0)
    union = actual + predicted - tp
    iou = torch.where(union > 0, tp / union, float('nan'))
    denominator = actual + predicted
    f1 = torch.where(denominator > 0, 2 * tp / denominator, float('nan'))
    total = m.sum()
    if total == 0:
        raise RuntimeError('No valid pixels were evaluated.')
    return float(tp.sum() / total), float(iou.nanmean()), iou.tolist(), f1.tolist()


def epoch_pass(model, loader, optimizer, device, ignore_index):
    training = optimizer is not None
    model.train(training)
    classes = len(loader.dataset.classes)
    confusion = torch.zeros(classes, classes, dtype=torch.int64, device=device)
    loss_total, pixels = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='sum')
    start = time.monotonic()
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            images = batch['image'].to(device)
            targets = batch['target'].to(device)
            valid = targets != ignore_index
            count = int(valid.sum())
            if count == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            scores = model(images)
            loss_sum = criterion(scores, targets)
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping this run.')
            if training:
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError('Non-finite gradient; stopping this run.')
                optimizer.step()
            with torch.no_grad():
                predictions = scores.argmax(1)
                codes = classes * targets[valid] + predictions[valid]
                confusion += torch.bincount(codes, minlength=classes**2).reshape(classes, classes)
                loss_total += float(loss_sum.detach())
                pixels += count
            if step == 1 or step % 100 == 0 or step == len(loader):
                print(f"  {'train' if training else 'validation'} {step}/{len(loader)}"
                      f' | loss {loss_total / pixels:.4f}'
                      f' | elapsed {(time.monotonic()-start)/60:.1f} min', flush=True)
    if pixels == 0:
        raise RuntimeError('No valid target pixels in this split.')
    return loss_total / pixels, confusion.cpu()


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix('.pt.part')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', type=Path, help='Path to a last_model.pt from this script')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Epochs and batch size must be positive.')
    root = Path(__file__).resolve().parent
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = WinterDataset(root, 'train', augment=True)
    validation = WinterDataset(root, 'validation')
    # Check support using TRAINING counts only. Never choose settings from test results.
    counts = np.zeros(len(train.classes), dtype=np.int64)
    with (root/'data/metadata/raw_label_distribution.csv').open() as source:
        for row in csv.DictReader(source):
            raw_id = int(row['raw_label_id'])
            if row['split'] == 'train' and raw_id in train.mapping:
                counts[train.mapping[raw_id]] += int(row['pixel_count'])
    if (counts == 0).any():
        raise RuntimeError('A retained class has no training examples. Inspect class counts.')
    majority_class = int(counts.argmax())
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, generator=generator)
    val_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    model = SmallUNet(num_classes=len(train.classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best, history, first_epoch = -1.0, [], 1
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['class_configuration'] != train.config:
            raise ValueError('Checkpoint class configuration does not match.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, best = saved['history'], saved['best_validation_miou']
        first_epoch = saved['epoch'] + 1
        # Restore ordering and augmentation state for epoch-boundary continuation.
        torch.set_rng_state(saved['torch_rng_state'])
        generator.set_state(saved['loader_rng_state'])
        if device.type == 'cuda' and saved['cuda_rng_states']:
            torch.cuda.set_rng_state_all(saved['cuda_rng_states'])
        run_dir = args.resume.resolve().parent
    else:
        run_dir = root/'results'/datetime.now(timezone.utc).strftime('winter9_%Y%m%d_%H%M%S_%f')
        run_dir.mkdir(parents=True)
        config = {'seed': 42, 'epochs': args.epochs, 'batch_size': args.batch_size,
                  'learning_rate': 0.001, 'loss': 'unweighted cross entropy',
                  'checkpoint_selection': 'highest full-validation mIoU',
                  'normalization': 'clip [-25,0], then (x+25)/25',
                  'torch_version': str(torch.__version__), 'classes': train.config,
                  'scope': 'winter-only development baseline; no Snow/Ice capability'}
        (run_dir/'configuration.json').write_text(json.dumps(config, indent=2))
        # Preserve exact splits and code for this experiment.
        import shutil
        for name in ('train.csv', 'validation.csv', 'split_description.json'):
            shutil.copy2(root/'data/metadata'/name, run_dir/name)
        for name in ('lulc_dataset.py', 'lulc_model.py', 'train_baseline.py'):
            shutil.copy2(root/name, run_dir/name)
    print('Device:', device, '| Training:', len(train), '| Validation:', len(validation), flush=True)
    print('Results:', run_dir, flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}', flush=True)
        train_loss, train_cm = epoch_pass(model, train_loader, optimizer, device, train.ignore_index)
        val_loss, val_cm = epoch_pass(model, val_loader, None, device, train.ignore_index)
        train_oa, train_miou, _, _ = metrics(train_cm)
        val_oa, val_miou, iou, f1 = metrics(val_cm)
        majority_oa = float(val_cm.sum(1)[majority_class] / val_cm.sum())
        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': val_loss,
                        'train_accuracy': train_oa, 'validation_accuracy': val_oa,
                        'train_miou': train_miou, 'validation_miou': val_miou,
                        'training_majority_baseline_validation_accuracy': majority_oa})
        improved = val_miou > best
        best = max(best, val_miou)
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'num_classes': len(train.classes),
                      'class_configuration': train.config, 'epoch': epoch,
                      'best_validation_miou': best, 'history': history,
                      'torch_rng_state': torch.get_rng_state(),
                      'loader_rng_state': generator.get_state(),
                      'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        save_checkpoint(run_dir/'last_model.pt', checkpoint)
        if improved:
            save_checkpoint(run_dir/'best_model.pt', checkpoint)
            np.save(run_dir/'best_validation_confusion.npy', val_cm.numpy())
            with (run_dir/'best_validation_classes.csv').open('w', newline='') as output:
                writer = csv.writer(output)
                writer.writerow(['class_id', 'class_name', 'reference_pixels', 'iou', 'f1'])
                for c in train.classes:
                    i = c['id']
                    writer.writerow([i, c['name'], int(val_cm.sum(1)[i]), iou[i], f1[i]])
        with (run_dir/'history.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f'Epoch {epoch}: train loss {train_loss:.4f} | val loss {val_loss:.4f}'
              f' | val OA {val_oa:.2%} | val mIoU {val_miou:.2%}'
              f' | majority baseline OA {majority_oa:.2%}', flush=True)
        # Save curves every epoch; completed work survives a later interruption.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, metric in zip(axes, ('loss', 'accuracy', 'miou')):
            for split in ('train', 'validation'):
                axis.plot([r['epoch'] for r in history],
                          [r[f'{split}_{metric}'] for r in history], label=split)
            axis.set(xlabel='Epoch', title=metric)
            axis.legend()
            axis.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir/'training_curves.png', dpi=150)
        plt.close(fig)
    print('\nFinished. Test set was not evaluated.', flush=True)


if __name__ == '__main__':
    main()
