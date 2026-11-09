"""Read Sentinel-1 images and matching pixel-level land-cover labels."""

from pathlib import Path
import csv
import json

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class LULCDataset(Dataset):
    """Load one image/label pair at a time, keeping memory usage manageable."""

    def __init__(self, project_dir, split="train", augment=False):
        self.project_dir = Path(project_dir).resolve()
        self.augment = augment

        if split not in ("train", "validation", "test"):
            raise ValueError("Unknown dataset split.")

        if augment and split != "train":
            raise ValueError("Augmentation is only allowed for training.")

        metadata_dir = self.project_dir / "data/metadata"

        with (metadata_dir / f"{split}.csv").open(
            newline="", encoding="utf-8"
        ) as source:
            self.records = list(csv.DictReader(source))

        if not self.records:
            raise ValueError(f"The {split} split is empty.")

        with (metadata_dir / "classes.json").open(
            encoding="utf-8"
        ) as source:
            self.config = json.load(source)

        self.classes = self.config["classes"]
        self.ignore_index = self.config["ignore_index"]

        # Convert original label IDs into the model's zero-based IDs.
        self.mapping = {
            source_id: definition["id"]
            for definition in self.classes
            for source_id in definition["source_ids"]
        }

        self.ignored_source_ids = set(
            self.config["ignored_source_ids"]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = self.project_dir / record["image"]
        label_path = self.project_dir / record["label"]

        with rasterio.open(image_path) as source:
            image = source.read(masked=True).astype(np.float32)
            image_crs = source.crs
            image_transform = source.transform

        with rasterio.open(label_path) as source:
            raw_labels = source.read(
                self.config["label_band"], masked=True
            )
            label_crs = source.crs
            label_transform = source.transform

        if image.shape[0] != 2:
            raise ValueError(f"Expected two radar channels: {image_path}")

        if (
            image.shape[1:] != raw_labels.shape
            or image_crs is None
            or image_crs != label_crs
            or not image_transform.almost_equals(label_transform)
        ):
            raise ValueError(f"Image and label grids differ: {image_path}")

        # Identify pixels with valid values in both radar channels.
        image_values = image.filled(np.nan)
        valid_image = (
            ~np.ma.getmaskarray(image).any(axis=0)
            & np.isfinite(image_values).all(axis=0)
        )

        label_values = raw_labels.filled(0)
        valid_labels = ~np.ma.getmaskarray(raw_labels)

        observed_ids = set(
            np.unique(label_values[valid_labels]).tolist()
        )
        unknown_ids = (
            observed_ids
            - set(self.mapping)
            - self.ignored_source_ids
        )

        if unknown_ids:
            raise ValueError(f"Unknown label IDs: {unknown_ids}")

        # Missing or excluded pixels receive the ignored-target ID.
        target = np.full(
            raw_labels.shape, self.ignore_index, dtype=np.int64
        )

        for source_id, model_id in self.mapping.items():
            valid = (
                valid_image
                & valid_labels
                & (label_values == source_id)
            )
            target[valid] = model_id

        # SEN12MS baseline normalization:
        # clip existing radar values to [-25, 0], then scale to [0, 1].
        # This is NOT a conversion from linear power into dB.
        image_values = np.nan_to_num(
            image_values, nan=-25.0, posinf=0.0, neginf=-25.0
        )
        image_values = (
            np.clip(image_values, -25.0, 0.0) + 25.0
        ) / 25.0
        image_values[:, ~valid_image] = 0.0

        # Apply identical flips to imagery and labels.
        if self.augment:
            if torch.rand(()).item() < 0.5:
                image_values = np.flip(image_values, axis=2)
                target = np.flip(target, axis=1)

            if torch.rand(()).item() < 0.5:
                image_values = np.flip(image_values, axis=1)
                target = np.flip(target, axis=0)

        # Copies remove negative array strides introduced by flipping.
        return {
            "image": torch.from_numpy(image_values.copy()).float(),
            "target": torch.from_numpy(target.copy()).long(),
            "id": image_path.name,
        }
