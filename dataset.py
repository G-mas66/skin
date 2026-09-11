"""Dataset and image transforms for white-light CEA classification."""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transform_functional


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {"image_path", "label", "patient_group", "split_unit"}
MODALITIES = ("M", "MB", "MP", "MR", "MUV")
MODALITY_PATTERN = re.compile(r"^(MUV|MB|MP|MR|M)(\d+)\.jpe?g$", re.IGNORECASE)
SOURCE_ID_PATTERN = re.compile(r"^(?:MUV|MB|MP|MR|M)(\d+)$", re.IGNORECASE)


class ResizePad:
    """Preserve aspect ratio and center-pad to a square canvas."""

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = self.size / max(image.size)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resized = image.resize(
            (width, height), Image.Resampling.BILINEAR, reducing_gap=2.0
        )
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(
            resized,
            ((self.size - width) // 2, (self.size - height) // 2),
        )
        return canvas


class ResizePadRect:
    """Preserve aspect ratio and center-pad to a width × height canvas."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width, self.height = int(width), int(height)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(self.width / image.width, self.height / image.height)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resized = image.resize(
            (width, height), Image.Resampling.BILINEAR, reducing_gap=2.0
        )
        canvas = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        canvas.paste(resized, ((self.width - width) // 2, (self.height - height) // 2))
        return canvas


def build_direct_transform(
    width: int,
    height: int,
    augmentation: dict,
    training: bool,
    color_augmentation: bool,
) -> transforms.Compose:
    steps = [ResizePadRect(width, height)]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(
                    p=float(augmentation["horizontal_flip_p"])
                ),
                transforms.RandomRotation(
                    degrees=float(augmentation["rotation_degrees"]),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=(0, 0, 0),
                ),
            ]
        )
        if color_augmentation:
            steps.append(
                transforms.ColorJitter(
                    brightness=float(augmentation.get("brightness", 0.0)),
                    contrast=float(augmentation.get("contrast", 0.0)),
                    saturation=0.0,
                    hue=0.0,
                )
            )
    steps.extend(
        [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    return transforms.Compose(steps)


def capture_id_from_source_id(value: object) -> int:
    match = SOURCE_ID_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"invalid source_id: {value}")
    return int(match.group(1))


def discover_modality_paths(
    data_root: str | Path, excluded_filenames: list[str] | tuple[str, ...]
) -> dict[str, dict[int, Path]]:
    root = Path(data_root).resolve()
    excluded = {str(name).strip().upper() for name in excluded_filenames}
    lookup = {modality: {} for modality in MODALITIES}
    for path in root.iterdir():
        if not path.is_file() or path.name.upper() in excluded:
            continue
        match = MODALITY_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        modality, capture_id = match.group(1).upper(), int(match.group(2))
        if capture_id in lookup[modality]:
            raise ValueError(
                f"duplicate {modality} files for capture_id {capture_id} after exclusions"
            )
        lookup[modality][capture_id] = path
    return lookup


def common_capture_ids(
    lookup: dict[str, dict[int, Path]], required_modalities: list[str] | tuple[str, ...]
) -> set[int]:
    required = [str(modality).upper() for modality in required_modalities]
    if not required:
        raise ValueError("at least one required modality is needed")
    invalid = sorted(set(required) - set(MODALITIES))
    if invalid:
        raise ValueError(f"unknown modalities: {invalid}")
    available = set(lookup[required[0]])
    for modality in required[1:]:
        available &= set(lookup[modality])
    return available


def local_face_crops(image: Image.Image) -> list[Image.Image]:
    """Return five overlapping fixed regions without writing intermediate images."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    fractions = (
        (0.00, 0.05, 0.58, 0.58),
        (0.42, 0.05, 1.00, 0.58),
        (0.00, 0.30, 0.58, 0.84),
        (0.42, 0.30, 1.00, 0.84),
        (0.20, 0.48, 0.80, 1.00),
    )
    return [
        image.crop(
            (
                round(left * width),
                round(top * height),
                round(right * width),
                round(bottom * height),
            )
        )
        for left, top, right, bottom in fractions
    ]


class DirectCEADataset(Dataset):
    """Read original JPG files directly and keep modality labels capture-aligned."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        data_root: str | Path,
        modality: str,
        required_modalities: list[str] | tuple[str, ...],
        global_size: tuple[int, int],
        local_size: tuple[int, int],
        view_mode: str,
        training: bool,
        augmentation: dict,
        excluded_filenames: list[str] | tuple[str, ...] = (),
    ) -> None:
        frame = (
            load_manifest(manifest)
            if not isinstance(manifest, pd.DataFrame)
            else manifest.reset_index(drop=True).copy()
        )
        self.modality = modality.upper()
        if self.modality not in (*MODALITIES, "ALL"):
            raise ValueError(f"unknown modality: {modality}")
        if view_mode not in {"global", "global_local"}:
            raise ValueError(f"unknown view mode: {view_mode}")
        if self.modality == "ALL" and view_mode != "global":
            raise ValueError("ALL early fusion supports global view only")
        self.view_mode = view_mode
        self.training = training
        self.augmentation = augmentation
        self.lookup = discover_modality_paths(data_root, excluded_filenames)
        available = common_capture_ids(self.lookup, required_modalities)
        frame["capture_id"] = frame["source_id"].map(capture_id_from_source_id)
        self.frame = frame[frame["capture_id"].isin(available)].reset_index(drop=True)
        if self.frame.empty:
            raise RuntimeError("no capture-aligned samples remain after modality filtering")
        if self.modality != "ALL" and not set(self.frame["capture_id"]).issubset(
            self.lookup[self.modality]
        ):
            raise RuntimeError(f"missing {self.modality} image for a selected capture")
        self.global_resize = ResizePadRect(*global_size)
        self.global_transform = build_direct_transform(
            *global_size,
            augmentation,
            training,
            color_augmentation=self.modality == "M",
        )
        self.local_transform = build_direct_transform(
            *local_size,
            augmentation,
            training,
            color_augmentation=self.modality == "M",
        )

    def __len__(self) -> int:
        return len(self.frame)

    def _early_fusion_tensor(self, capture_id: int) -> torch.Tensor:
        images = []
        for modality in MODALITIES:
            with Image.open(self.lookup[modality][capture_id]) as source:
                images.append(ImageOps.exif_transpose(source).convert("RGB"))

        flip = self.training and random.random() < float(
            self.augmentation["horizontal_flip_p"]
        )
        angle = (
            random.uniform(
                -float(self.augmentation["rotation_degrees"]),
                float(self.augmentation["rotation_degrees"]),
            )
            if self.training
            else 0.0
        )
        brightness = (
            random.uniform(
                max(0.0, 1.0 - float(self.augmentation.get("brightness", 0.0))),
                1.0 + float(self.augmentation.get("brightness", 0.0)),
            )
            if self.training
            else 1.0
        )
        contrast = (
            random.uniform(
                max(0.0, 1.0 - float(self.augmentation.get("contrast", 0.0))),
                1.0 + float(self.augmentation.get("contrast", 0.0)),
            )
            if self.training
            else 1.0
        )

        tensors = []
        for modality, image in zip(MODALITIES, images):
            image = self.global_resize(image)
            if flip:
                image = transform_functional.hflip(image)
            if angle:
                image = transform_functional.rotate(
                    image,
                    angle,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=(0, 0, 0),
                )
            if modality == "M":
                image = transform_functional.adjust_brightness(image, brightness)
                image = transform_functional.adjust_contrast(image, contrast)
            tensor = transform_functional.to_tensor(image)
            tensors.append(
                transform_functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
            )
        return torch.cat(tensors, dim=0)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        capture_id = int(row["capture_id"])
        if self.modality == "ALL":
            views = [self._early_fusion_tensor(capture_id)]
        else:
            with Image.open(self.lookup[self.modality][capture_id]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                views = [self.global_transform(image)]
                if self.view_mode == "global_local":
                    views.extend(
                        self.local_transform(crop) for crop in local_face_crops(image)
                    )
        return {
            "views": views,
            "target": int(row["label"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "patient_group": str(row["patient_group"]),
            "capture_id": capture_id,
        }


def direct_collate(batch: list[dict]) -> dict:
    view_counts = {len(item["views"]) for item in batch}
    if len(view_counts) != 1:
        raise ValueError("all direct samples in a batch must have the same view count")
    view_count = next(iter(view_counts))
    return {
        "views": [
            torch.stack([item["views"][view_index] for item in batch])
            for view_index in range(view_count)
        ],
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "sample_weight": torch.tensor(
            [item["sample_weight"] for item in batch], dtype=torch.float32
        ),
        "patient_group": [item["patient_group"] for item in batch],
        "capture_id": [item["capture_id"] for item in batch],
    }


def build_train_transform(size: int, augmentation: dict) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizePad(size),
            transforms.RandomHorizontalFlip(
                p=float(augmentation["horizontal_flip_p"])
            ),
            transforms.RandomRotation(
                degrees=float(augmentation["rotation_degrees"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=(0, 0, 0),
            ),
            transforms.ColorJitter(
                brightness=float(augmentation["brightness"]),
                contrast=float(augmentation["contrast"]),
                saturation=float(augmentation["saturation"]),
                hue=float(augmentation["hue"]),
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizePad(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(path))
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["label"] = frame["label"].astype(int)
    invalid = sorted(set(frame["label"]) - set(range(5)))
    if invalid:
        raise ValueError(f"manifest contains invalid labels: {invalid}")
    if frame["image_path"].duplicated().any():
        raise ValueError("manifest contains duplicate image paths")
    return frame


def validate_split_isolation(frames: dict[str, pd.DataFrame]) -> None:
    names = list(frames)
    for index, left_name in enumerate(names):
        left = frames[left_name]
        for right_name in names[index + 1 :]:
            right = frames[right_name]
            for column in ("patient_group", "split_unit", "image_path"):
                overlap = set(left[column].astype(str)) & set(
                    right[column].astype(str)
                )
                if overlap:
                    raise ValueError(
                        f"{column} leakage between {left_name} and {right_name}"
                    )
            if "sha256" in left and "sha256" in right:
                duplicate_overlap = set(left["sha256"]) & set(right["sha256"])
                if duplicate_overlap:
                    raise ValueError(
                        f"exact duplicate leakage between {left_name} and {right_name}"
                    )


class CEADataset(Dataset):
    """Load cached white-light images and return CEA labels 0-4."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        cache_root: str | Path,
        transform,
    ) -> None:
        self.frame = (
            load_manifest(manifest)
            if not isinstance(manifest, pd.DataFrame)
            else manifest.reset_index(drop=True).copy()
        )
        self.cache_root = Path(cache_root).resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, relative_path: str) -> Path:
        path = (self.cache_root / relative_path).resolve()
        try:
            path.relative_to(self.cache_root)
        except ValueError as exc:
            raise ValueError(f"image path escapes cache root: {relative_path}") from exc
        return path

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = self._image_path(str(row["image_path"]))
        with Image.open(image_path) as image:
            tensor = self.transform(image)
        return {
            "image": tensor,
            "target": int(row["label"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
            "patient_group": str(row["patient_group"]),
            "image_path": str(row["image_path"]),
        }
