#!/usr/bin/env python3
"""Protocol-compliant Step 8 five-class soft-label single-channel benchmark."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


EXCLUDED_IDS = {69, 296, 769, 770}
SEEDS = (42, 3407, 2026)
CHANNELS = ("M", "MB", "MP", "MR", "MUV")
CHANNEL_NAMES = {
    "M": "M (White)",
    "MB": "MB",
    "MP": "MP",
    "MR": "MR",
    "MUV": "MUV",
}
RESOLUTION = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METRICS = ("accuracy", "macro_f1", "mae", "adjacent_error_rate")
PER_CLASS_METRICS = ("precision", "recall", "f1")
SOFT_TRUE_WEIGHT = 0.90
SOFT_ADJACENT_WEIGHT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "smoke", "train", "summarize"), required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/cache_768"))
    parser.add_argument("--label-csv", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/CEA.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", default="42,3407,2026")
    parser.add_argument("--channels", default="M,MB,MP,MR,MUV")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--source-mr69-files",
        default="MR0069.JPG,MR069.JPG",
        help="MR ID=69 source files verified and excluded in the original cache audit",
    )
    return parser.parse_args()


def source_id(value: str) -> int:
    match = __import__("re").fullmatch(r"M(\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid M source id: {value!r}")
    return int(match.group(1))


def load_labels(label_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(label_csv)
    required = {"M_id", "patient_id", "CEA_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"label CSV missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["sample_id"] = frame["M_id"].astype(str).map(source_id)
    frame["patient_group"] = frame["patient_id"].astype(str)
    frame["CEA_class"] = frame["CEA_class"].astype(int)
    if frame["sample_id"].duplicated().any():
        raise ValueError("duplicate M_id in label CSV")
    if not frame["CEA_class"].isin(range(5)).all():
        raise ValueError("CEA_class must be in 0..4")
    return frame[["sample_id", "patient_group", "CEA_class", "M_id"]]


def channel_path(cache_root: Path, channel: str, sample_id: int) -> Path:
    return Path(cache_root) / channel / f"{channel}{sample_id:04d}.JPG"


def soft_label_vector(label: int) -> np.ndarray:
    vector = np.zeros(5, dtype=np.float32)
    vector[label] = SOFT_TRUE_WEIGHT
    adjacent = [value for value in (label - 1, label + 1) if 0 <= value < 5]
    for value in adjacent:
        vector[value] = SOFT_ADJACENT_WEIGHT
    remainder = 1.0 - float(vector.sum())
    if len(adjacent) == 1:
        vector[adjacent[0]] += remainder
    elif len(adjacent) == 2:
        vector[label] += remainder
    if not np.isclose(vector.sum(), 1.0):
        raise ValueError(f"soft label does not sum to one: {vector}")
    return vector


def soft_label_definitions() -> dict:
    return {
        "interior_class": {
            "true_class": SOFT_TRUE_WEIGHT,
            "each_adjacent_class": SOFT_ADJACENT_WEIGHT,
            "non_adjacent_classes": 0.0,
        },
        "boundary_class": {"true_class": 0.90, "only_adjacent_class": 0.10},
        "vectors": {str(label): soft_label_vector(label).tolist() for label in range(5)},
    }


def split_and_audit(cache_root: Path, label_csv: Path, source_mr69_files: str) -> tuple[pd.DataFrame, dict]:
    labels = load_labels(label_csv)
    if len(labels) != 997:
        raise ValueError(f"expected 997 raw labelled M IDs, found {len(labels)}")
    raw_label_count = len(labels)
    excluded_in_raw_labels = sorted(set(labels["sample_id"]) & EXCLUDED_IDS)
    labels = labels.loc[~labels["sample_id"].isin(EXCLUDED_IDS)].copy()
    labels["split"] = np.where(labels["sample_id"] <= 793, "train", "test")

    missing: dict[str, list[str]] = {}
    extra_excluded: dict[str, list[str]] = {}
    channel_counts: dict[str, int] = {}
    for channel in CHANNELS:
        channel_missing = []
        for sample_id in labels["sample_id"]:
            if not channel_path(cache_root, channel, int(sample_id)).is_file():
                channel_missing.append(f"{channel}{int(sample_id):04d}")
        missing[channel] = channel_missing[:10]
        found_excluded = sorted(
            path.name
            for path in Path(cache_root, channel).glob(f"{channel}*.JPG")
            if int(__import__("re").fullmatch(rf"{channel}(\d+)\.JPG", path.name).group(1)) in EXCLUDED_IDS
        )
        extra_excluded[channel] = found_excluded
        channel_counts[channel] = len(list(Path(cache_root, channel).glob(f"{channel}*.JPG")))
    if any(missing.values()):
        raise FileNotFoundError(f"missing channel images: {missing}")
    if any(extra_excluded.values()):
        raise ValueError(f"excluded IDs remain in cache: {extra_excluded}")
    if any(count != 996 for count in channel_counts.values()):
        raise ValueError(f"channel cache counts must all be 996, found {channel_counts}")

    train = labels[labels["split"] == "train"]
    test = labels[labels["split"] == "test"]
    sample_overlap = set(train["sample_id"]) & set(test["sample_id"])
    patient_overlap = set(train["patient_group"]) & set(test["patient_group"])
    if sample_overlap or patient_overlap:
        raise ValueError(f"split leakage: samples={sample_overlap}, patients={patient_overlap}")

    mr69 = sorted(value.strip() for value in source_mr69_files.split(",") if value.strip())
    if mr69 != ["MR0069.JPG", "MR069.JPG"]:
        raise ValueError(f"expected both source MR ID=69 files in audit record, found: {mr69}")

    audit = {
        "passed": True,
        "raw_csv_rows": raw_label_count,
        "raw_unique_labelled_m_ids": raw_label_count,
        "excluded_ids_present_in_raw_labels": excluded_in_raw_labels,
        "train_samples": len(train),
        "test_samples": len(test),
        "excluded_ids": sorted(EXCLUDED_IDS),
        "excluded_ids_absent_from_used_labels": True,
        "mr_id69_files_found": mr69,
        "mr_id69_files_excluded": True,
        "sample_overlap": len(sample_overlap),
        "patient_overlap": len(patient_overlap),
        "task": "five-class",
        "class_mapping": "CEA 0 -> 0; 1 -> 1; 2 -> 2; 3 -> 3; 4 -> 4",
        "num_classes": 5,
        "label_strategy": "adjacent soft label: true=0.90 and each adjacent=0.05; boundary true=0.90 and only adjacent=0.10",
        "soft_label_definition": soft_label_definitions(),
        "input_resolution": RESOLUTION,
        "required_channels": list(CHANNELS),
        "required_channels_present": True,
        "channel_cache_counts": channel_counts,
        "images_and_labels_matched_by_sample_id": True,
        "train_class_distribution": {
            str(key): int(value) for key, value in sorted(Counter(train["CEA_class"]).items())
        },
        "test_class_distribution": {
            str(key): int(value) for key, value in sorted(Counter(test["CEA_class"]).items())
        },
        "hard_label_comparison_status": "pending external Step 5 results",
    }
    return labels, audit


class Letterbox:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = image.resize(new_size, Image.Resampling.BILINEAR)
        left = (self.size - new_size[0]) // 2
        top = (self.size - new_size[1]) // 2
        right = self.size - new_size[0] - left
        bottom = self.size - new_size[1] - top
        return transforms.functional.pad(
            resized, [left, top, right, bottom], fill=0, padding_mode="constant"
        )


class SoftChannelDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, cache_root: Path, channel: str, size: int, train: bool
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.channel = channel
        augmentations = []
        if train:
            augmentations.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(7),
                    transforms.ColorJitter(
                        brightness=0.10, contrast=0.10, saturation=0.0, hue=0.0
                    ),
                ]
            )
        self.transform = transforms.Compose(
            [
                Letterbox(size),
                *augmentations,
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        path = channel_path(self.cache_root, self.channel, int(row["sample_id"]))
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row["CEA_class"])
        return tensor, torch.from_numpy(soft_label_vector(label)), label, str(row["M_id"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def metric_dict(y_true: list[int], y_pred: list[int]) -> dict:
    labels = list(range(5))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mae": float(np.mean(errors)),
        "adjacent_error_rate": float(np.mean(errors == 1)),
        "non_adjacent_error_rate": float(np.mean(errors > 1)),
        **{
            f"precision_class{class_index}": float(precision[class_index])
            for class_index in labels
        },
        **{f"recall_class{class_index}": float(recall[class_index]) for class_index in labels},
        **{f"f1_class{class_index}": float(f1[class_index]) for class_index in labels},
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    records: list[dict] = []
    for images, _, targets, ids in loader:
        probabilities = torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        for sample_id, target, prediction, values in zip(ids, targets.tolist(), predictions, probabilities):
            record = {
                "M_id": sample_id,
                "true_label": int(target),
                "prediction": int(prediction),
            }
            record.update(
                {f"probability_class{class_index}": float(values[class_index]) for class_index in range(5)}
            )
            records.append(record)
    frame = pd.DataFrame(records)
    return metric_dict(frame["true_label"].tolist(), frame["prediction"].tolist()), frame


def smoke_test(args: argparse.Namespace, labels: pd.DataFrame, audit: dict) -> None:
    run_dir = args.output_root / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = labels[labels["split"] == "train"].head(args.batch_size)
    channel_results = []
    for channel in CHANNELS:
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
        dataset = SoftChannelDataset(frame, args.cache_root, channel, RESOLUTION, train=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        images, soft_targets, targets, ids = next(iter(loader))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=not args.no_amp):
            loss = soft_target_cross_entropy(model(images.to(device)), soft_targets.to(device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        finite = bool(loss.isfinite().item())
        target_sums = soft_targets.sum(dim=1)
        passed = finite and bool(torch.allclose(target_sums, torch.ones_like(target_sums)))
        channel_results.append(
            {
                "channel": channel,
                "passed": passed,
                "batch_shape": list(images.shape),
                "ids": list(ids),
                "hard_targets": targets.tolist(),
                "soft_target_sums": target_sums.tolist(),
                "loss": float(loss.item()),
            }
        )
        print(f"smoke {channel}: shape={tuple(images.shape)} loss={loss.item():.6f}", flush=True)
        del model, optimizer, dataset, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "passed": all(item["passed"] for item in channel_results),
        "device": str(device),
        "resolution": RESOLUTION,
        "batch_size": args.batch_size,
        "soft_label_definition": soft_label_definitions(),
        "channels": channel_results,
        "audit_passed": audit["passed"],
    }
    save_json(run_dir / "smoke_result.json", result)
    if not result["passed"]:
        raise RuntimeError("Step 8 smoke test failed")


def train_one(
    args: argparse.Namespace, labels: pd.DataFrame, audit: dict, channel: str, seed: int
) -> Path:
    run_dir = args.output_root / "runs" / f"{channel}_five_soft_res384_seed{seed}"
    if (run_dir / "run_result.json").exists():
        print(f"already complete: {run_dir}", flush=True)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Step 8 training requires CUDA")

    train_dataset = SoftChannelDataset(
        labels[labels["split"] == "train"], args.cache_root, channel, RESOLUTION, train=True
    )
    test_dataset = SoftChannelDataset(
        labels[labels["split"] == "test"], args.cache_root, channel, RESOLUTION, train=False
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.workers > 0,
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    config = {
        "experiment_name": f"step8_channel_{channel}_five_soft_384",
        "benchmark_type": "Five-class Soft-label Single-channel Benchmark",
        "task": "five-class",
        "label_strategy": "adjacent soft label: true=0.90 and each adjacent=0.05; boundary true=0.90 and only adjacent=0.10",
        "soft_label_definition": soft_label_definitions(),
        "seed": seed,
        "train_sample_count": len(train_dataset),
        "test_sample_count": len(test_dataset),
        "input_resolution": RESOLUTION,
        "input_channel": CHANNEL_NAMES[channel],
        "model": "ResNet50",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretrained": "ImageNet ResNet50_Weights.IMAGENET1K_V1",
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": args.epochs,
        "scheduler": None,
        "loss": "SoftTargetCrossEntropy",
        "augmentation": {
            "letterbox": "proportional resize then symmetric zero padding",
            "horizontal_flip_p": 0.5,
            "rotation_degrees": 7,
            "brightness": 0.10,
            "contrast": 0.10,
            "saturation": 0.0,
            "hue": 0.0,
        },
        "mixed_precision": not args.no_amp,
        "early_stopping": False,
        "validation_set": False,
        "hard_label_comparison_status": "pending external Step 5 results",
        "data_audit": audit,
    }
    save_json(run_dir / "config.json", config)

    history: list[dict] = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        epoch_started = time.perf_counter()
        for images, soft_targets, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=not args.no_amp):
                loss = soft_target_cross_entropy(model(images), soft_targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"channel={channel} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"loss={history[-1]['train_loss']:.6f}",
            flush=True,
        )

    torch.save(model.state_dict(), run_dir / "last.pth")
    metrics, predictions = evaluate(model, test_loader, device)
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    metrics.update(
        {
            "experiment_name": config["experiment_name"],
            "channel": channel,
            "input_channel": CHANNEL_NAMES[channel],
            "resolution": RESOLUTION,
            "seed": seed,
            "checkpoint": "last.pth",
            "epochs_completed": args.epochs,
            "training_seconds": time.perf_counter() - started,
        }
    )
    save_json(run_dir / "run_result.json", metrics)
    save_json(
        run_dir / "metadata.json", {**config, "test_used": True, "result_file": "run_result.json"}
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return run_dir


def load_runs(root: Path) -> pd.DataFrame:
    records = []
    for channel in CHANNELS:
        for seed in SEEDS:
            path = root / "runs" / f"{channel}_five_soft_res384_seed{seed}" / "run_result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing formal run result: {path}")
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("channel") != channel or item.get("seed") != seed:
                raise ValueError(f"unexpected Step 8 result: {path}")
            item.pop("confusion_matrix", None)
            item["run_result_path"] = str(path)
            records.append(item)
    frame = pd.DataFrame(records)
    missing = {"channel", "seed", *METRICS} - set(frame.columns)
    if missing:
        raise ValueError(f"run results missing fields: {sorted(missing)}")
    return frame


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    for channel in CHANNELS:
        group = runs[runs["channel"] == channel]
        row: dict[str, object] = {
            "channel": channel,
            "input_channel": CHANNEL_NAMES[channel],
        }
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records)


def benchmark_chart(summary: pd.DataFrame, metric: str, output: Path, majority_accuracy: float) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = np.arange(len(summary))
    scale = 100.0 if metric != "mae" else 1.0
    means = summary[f"{metric}_mean"].to_numpy(float) * scale
    stds = summary[f"{metric}_std"].to_numpy(float) * scale
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2"]
    axis.bar(positions, means, yerr=stds, capsize=5, color=colors, edgecolor="#243B53")
    if metric == "accuracy":
        axis.axvline(-0.5, color="#6B7A90", linestyle="--", alpha=0)
        axis.axhline(majority_accuracy * 100, color="#6B7A90", linestyle="--", linewidth=1.4)
        axis.text(len(summary) - 0.5, majority_accuracy * 100 + 1, f"Majority {majority_accuracy * 100:.2f}%")
    axis.set_xticks(positions, [CHANNEL_NAMES[channel] for channel in summary["channel"]])
    unit = "%" if metric != "mae" else "grade"
    axis.set_ylabel(f"{metric.replace('_', '-')} ({unit})")
    axis.set_xlabel("Input channel")
    axis.set_title(f"Step 8 five-class soft-label benchmark: {metric.replace('_', '-')}")
    axis.grid(axis="y", alpha=0.25)
    upper = 100.0 if metric != "mae" else max(1.0, float(np.max(means + stds)) * 1.25)
    axis.set_ylim(0, upper)
    for position, mean, std in zip(positions, means, stds):
        axis.text(
            position,
            min(upper - 2, mean + std + 2),
            f"{mean:.2f}±{std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def per_class_chart(runs: pd.DataFrame, metric: str, output: Path) -> None:
    width = 0.16
    figure, axis = plt.subplots(figsize=(14, 6.4))
    for channel_index, channel in enumerate(CHANNELS):
        means, stds = [], []
        for class_index in range(5):
            values = runs.loc[runs["channel"] == channel, f"{metric}_class{class_index}"].astype(float)
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        means_array = np.asarray(means) * 100
        stds_array = np.asarray(stds) * 100
        positions = np.arange(5) + (channel_index - 2) * width
        axis.bar(
            positions,
            means_array,
            width=width,
            yerr=stds_array,
            capsize=3,
            label=CHANNEL_NAMES[channel],
        )
        for position, mean, std in zip(positions, means_array, stds_array):
            axis.text(position, min(97, mean + std + 1), f"{mean:.1f}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(np.arange(5), [f"Class {value}" for value in range(5)])
    axis.set_ylim(0, 100)
    axis.set_xlabel("CEA class")
    axis.set_ylabel(f"Per-class {metric} (%)")
    axis.set_title(f"Step 8 five-class soft-label per-class {metric}")
    axis.legend(ncol=3, fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def confusion_chart(runs: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, 5, figsize=(19, 11))
    for channel_index, channel in enumerate(CHANNELS):
        group = runs[runs["channel"] == channel].sort_values("seed")
        for seed_index, (_, row) in enumerate(group.iterrows()):
            axis = axes[seed_index, channel_index]
            matrix = np.asarray(
                json.loads(Path(row["run_result_path"]).read_text(encoding="utf-8"))["confusion_matrix"]
            )
            image = axis.imshow(matrix, cmap="Blues", vmin=0)
            axis.set_title(f"{CHANNEL_NAMES[channel]} / seed {int(row['seed'])}", fontsize=10)
            axis.set_xticks(range(5), [f"pred {value}" for value in range(5)])
            axis.set_yticks(range(5), [f"true {value}" for value in range(5)])
            for y in range(5):
                for x in range(5):
                    axis.text(x, y, str(matrix[y, x]), ha="center", va="center", color="#172033", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Step 8 soft-label confusion matrices (all channels and formal seeds)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def frame_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="metrics", escape=True)


def majority_accuracy(labels: pd.DataFrame) -> float:
    test = labels[labels["split"] == "test"]
    return float(test["CEA_class"].value_counts().iloc[0] / len(test))


def summarize(args: argparse.Namespace, labels: pd.DataFrame, audit: dict) -> None:
    runs = load_runs(args.output_root)
    summary = aggregate(runs)
    ranking = summary.sort_values(
        ["macro_f1_mean", "macro_f1_std"], ascending=[False, True], ignore_index=True
    )[
        [
            "channel",
            "input_channel",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "accuracy_std",
            "mae_mean",
            "mae_std",
            "adjacent_error_rate_mean",
            "adjacent_error_rate_std",
        ]
    ]
    save_json(args.output_root / "soft_channel_ranking.json", ranking.to_dict(orient="records"))
    runs.drop(columns=["run_result_path"]).to_csv(args.output_root / "soft_run_comparison.csv", index=False)
    summary.to_csv(args.output_root / "soft_benchmark_summary.csv", index=False)
    ranking.to_csv(args.output_root / "soft_channel_ranking.csv", index=False)

    baseline = majority_accuracy(labels)
    chart_paths: dict[str, Path] = {}
    for metric in METRICS:
        path = args.output_root / f"{metric}_soft_benchmark.png"
        benchmark_chart(summary, metric, path, baseline)
        chart_paths[metric] = path
    per_class_paths = {
        metric: args.output_root / f"per_class_{metric}_soft_benchmark.png"
        for metric in PER_CLASS_METRICS
    }
    for metric, path in per_class_paths.items():
        per_class_chart(runs, metric, path)
    confusion_path = args.output_root / "soft_confusion_matrices.png"
    confusion_chart(runs, confusion_path)

    conclusion = {
        "stage": "Soft-label training and standalone benchmark complete",
        "ranking_rule": "higher three-seed Macro-F1 mean; tie-break smaller Macro-F1 standard deviation",
        "highest_ranked_channel": ranking.iloc[0]["channel"],
        "hard_label_comparison": "pending external Step 5 results; no stability claim is made yet",
        "next_required_output": "merge Step 5 Hard Label run results, then generate Hard vs Soft comparison charts and conclusion",
    }
    save_json(args.output_root / "step8_soft_stage_conclusion.json", conclusion)

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 8 Five-class Soft-label Single-channel Benchmark</title>
<style>
body {{ font-family: Arial,"Microsoft YaHei",sans-serif; max-width:1280px; margin:32px auto; padding:0 20px; color:#172033; }}
h1,h2 {{ color:#123A63; }} section {{ margin:32px 0; }} img {{ width:100%; height:auto; }}
.metrics {{ border-collapse:collapse; width:100%; font-size:12px; }} .metrics th,.metrics td {{ border:1px solid #D8DEE8; padding:6px; text-align:right; }}
.metrics th {{ background:#EDF3F8; }} pre {{ white-space:pre-wrap; }} .ok {{ color:#137333; font-weight:700; }} .pending {{ color:#A15C00; font-weight:700; }}
</style></head><body>
<h1>Step 8 Five-class Soft-label Single-channel Benchmark</h1>
<p>Soft-label stage only. Input resolution is fixed at 384 x 384. Task: original CEA 0-4 five-class. Soft labels are dynamically generated from original hard labels: interior classes use 0.90 for the true class and 0.05 for each adjacent class; boundary classes use 0.90 and 0.10. ResNet50 ImageNet pretrained; AdamW lr 1e-4, weight decay 1e-4; batch 16; 50 epochs; no validation, scheduler, or early stopping. Seeds: 42, 3407, 2026.</p>
<section><h2>Hard-label comparison status</h2><p class="pending">PENDING: Step 5 Hard Label results are running on another computer. This report does not claim whether Soft Label improves Hard Label.</p></section>
<section><h2>Data integrity</h2><p class="ok">PASS</p><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Soft-label channel ranking</h2>{frame_html(ranking)}</section>
<section><h2>Mean ± standard deviation</h2>{frame_html(summary)}</section>
<section><h2>Raw three-seed results</h2>{frame_html(runs.drop(columns=['run_result_path']))}</section>
<section><h2>Conclusion</h2><pre>{html.escape(json.dumps(conclusion, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Confusion matrices</h2><img src="{image_data_uri(confusion_path)}" alt="Soft-label confusion matrices"></section>
{''.join(f'<section><h2>Per-class {metric}</h2><img src="{image_data_uri(path)}" alt="Per-class {metric}"></section>' for metric, path in per_class_paths.items())}
{''.join(f'<section><h2>{metric.replace("_", "-")}</h2><img src="{image_data_uri(path)}" alt="{metric} soft benchmark"></section>' for metric, path in chart_paths.items())}
</body></html>"""
    (args.output_root / "step8_soft_stage_report.html").write_text(document, encoding="utf-8")
    print(json.dumps(ranking.to_dict(orient="records"), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    channels = tuple(value.strip().upper() for value in args.channels.split(","))
    if seeds != SEEDS or channels != CHANNELS:
        raise ValueError(f"Step 8 requires channels={CHANNELS} and seeds={SEEDS}")
    if args.epochs != 50 or args.batch_size != 16 or RESOLUTION != 384:
        raise ValueError("Step 8 protocol fixes resolution=384, batch_size=16, epochs=50")

    labels, audit = split_and_audit(args.cache_root, args.label_csv, args.source_mr69_files)
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "data_audit.json", audit)
    if args.action == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    elif args.action == "smoke":
        smoke_test(args, labels, audit)
    elif args.action == "train":
        for channel in CHANNELS:
            for seed in SEEDS:
                train_one(args, labels, audit, channel, seed)
    elif args.action == "summarize":
        summarize(args, labels, audit)


if __name__ == "__main__":
    main()
