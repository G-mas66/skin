#!/usr/bin/env python3
"""Protocol-compliant Step 9 five-class soft-label multi-channel benchmark."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosacea_step8.step8_train import (  # noqa: E402
    CHANNELS,
    CHANNEL_NAMES,
    METRICS,
    PER_CLASS_METRICS,
    RESOLUTION,
    SEEDS,
    Letterbox,
    channel_path,
    evaluate,
    frame_html,
    image_data_uri,
    majority_accuracy,
    save_json,
    seed_everything,
    soft_label_definitions,
    soft_label_vector,
    soft_target_cross_entropy,
    split_and_audit,
)


COMBOS = (
    ("M", "MR"),
    ("M", "MB"),
    ("M", "MB", "MR"),
    ("MB", "MR"),
    ("M", "MB", "MP", "MR", "MUV"),
)
METHODS = ("M_baseline", *(("+".join(combo) for combo in COMBOS)))
METHOD_NAMES = {
    "M_baseline": "M (White, Step 8 baseline)",
    "M+MR": "M + MR",
    "M+MB": "M + MB",
    "M+MB+MR": "M + MB + MR",
    "MB+MR": "MB + MR",
    "M+MB+MP+MR+MUV": "All channels",
}
CHART_COLORS = {
    "M_baseline": "#6B7A90",
    "M+MR": "#4C78A8",
    "M+MB": "#F58518",
    "M+MB+MR": "#54A24B",
    "MB+MR": "#B279A2",
    "M+MB+MP+MR+MUV": "#72B7B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "smoke", "train", "summarize"), required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/cache_768"))
    parser.add_argument("--label-csv", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/CEA.csv"))
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("/root/autodl-tmp/rosacea_step8/results"),
        help="Completed Step 8 results used for the Soft-label White baseline",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
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


class SoftMultiChannelDataset(Dataset):
    """Load sample-ID-matched channels with synchronized augmentation and soft labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_root: Path,
        channels: tuple[str, ...],
        size: int,
        train: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.channels = channels
        self.letterbox = Letterbox(size)
        self.train = train

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        images = []
        for channel in self.channels:
            path = channel_path(self.cache_root, channel, int(row["sample_id"]))
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        images = [self.letterbox(image) for image in images]
        if self.train:
            if torch.rand(()) < 0.5:
                images = [TF.hflip(image) for image in images]
            angle = float(torch.empty(()).uniform_(-7.0, 7.0))
            images = [
                TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                for image in images
            ]
            brightness = float(torch.empty(()).uniform_(0.9, 1.1))
            contrast = float(torch.empty(()).uniform_(0.9, 1.1))
            images = [
                TF.adjust_contrast(TF.adjust_brightness(image, brightness), contrast)
                for image in images
            ]

        tensors = []
        for image in images:
            tensor = TF.to_tensor(image)
            tensors.append(
                TF.normalize(
                    tensor,
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                )
            )
        label = int(row["CEA_class"])
        return (
            torch.cat(tensors, dim=0),
            torch.from_numpy(soft_label_vector(label)),
            label,
            str(row["M_id"]),
        )


def build_model(input_channels: int) -> nn.Module:
    if input_channels % 3 or input_channels < 3:
        raise ValueError(f"input_channels must be a multiple of 3, found {input_channels}")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if input_channels != 3:
        original = model.conv1
        replacement = nn.Conv2d(
            input_channels,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=False,
        )
        groups = input_channels // 3
        with torch.no_grad():
            replacement.weight.copy_(original.weight.repeat(1, groups, 1, 1) / groups)
        model.conv1 = replacement
    model.fc = nn.Linear(model.fc.in_features, 5)
    return model


def validate_combinations() -> None:
    required = set(CHANNELS)
    for combo in COMBOS:
        if len(combo) < 2 or len(set(combo)) != len(combo) or set(combo) - required:
            raise ValueError(f"invalid Step 9 channel combination: {combo}")
    if required != set().union(*(set(combo) for combo in COMBOS)):
        raise ValueError("Step 9 must include the all-channel combination")


def method_id(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def run_path(root: Path, method: str, seed: int) -> Path:
    return root / "runs" / f"{method}_five_soft_res384_seed{seed}"


def baseline_result_path(root: Path, seed: int) -> Path:
    return root / "runs" / f"M_five_soft_res384_seed{seed}" / "run_result.json"


def load_baseline(root: Path) -> pd.DataFrame:
    records = []
    for seed in SEEDS:
        path = baseline_result_path(root, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing Step 8 Soft-label White baseline result: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("channel") != "M" or item.get("seed") != seed:
            raise ValueError(f"unexpected Step 8 Soft-label White baseline result: {path}")
        item.pop("channel", None)
        item.pop("confusion_matrix", None)
        item["method_id"] = "M_baseline"
        item["channels"] = ["M"]
        item["input_channel"] = METHOD_NAMES["M_baseline"]
        item["run_result_path"] = str(path)
        records.append(item)
    frame = pd.DataFrame(records)
    missing = {"method_id", "seed", *METRICS} - set(frame.columns)
    if missing:
        raise ValueError(f"Step 8 baseline results missing fields: {sorted(missing)}")
    return frame


def load_multichannel_runs(root: Path) -> pd.DataFrame:
    records = []
    for combo in COMBOS:
        method = method_id(combo)
        for seed in SEEDS:
            path = run_path(root, method, seed) / "run_result.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing formal Step 9 result: {path}")
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("method_id") != method or item.get("seed") != seed:
                raise ValueError(f"unexpected Step 9 result: {path}")
            item.pop("confusion_matrix", None)
            item["run_result_path"] = str(path)
            records.append(item)
    frame = pd.DataFrame(records)
    missing = {"method_id", "seed", *METRICS} - set(frame.columns)
    if missing:
        raise ValueError(f"Step 9 results missing fields: {sorted(missing)}")
    return frame


def smoke_test(args: argparse.Namespace, labels: pd.DataFrame, audit: dict) -> None:
    run_dir = args.output_root / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = labels[labels["split"] == "train"].head(args.batch_size)
    results = []
    for combo in COMBOS:
        method = method_id(combo)
        model = build_model(len(combo) * 3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
        dataset = SoftMultiChannelDataset(frame, args.cache_root, combo, RESOLUTION, train=True)
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
        results.append(
            {
                "method_id": method,
                "channels": list(combo),
                "passed": passed,
                "batch_shape": list(images.shape),
                "ids": list(ids),
                "hard_targets": targets.tolist(),
                "soft_target_sums": target_sums.tolist(),
                "loss": float(loss.item()),
            }
        )
        print(f"smoke {method}: shape={tuple(images.shape)} loss={loss.item():.6f}", flush=True)
        del model, optimizer, dataset, loader
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "passed": all(item["passed"] for item in results),
        "device": str(device),
        "resolution": RESOLUTION,
        "batch_size": args.batch_size,
        "soft_label_definition": soft_label_definitions(),
        "methods": results,
        "audit_passed": audit["passed"],
    }
    save_json(run_dir / "smoke_result.json", result)
    if not result["passed"]:
        raise RuntimeError("Step 9 smoke test failed")


def train_one(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    audit: dict,
    combo: tuple[str, ...],
    seed: int,
) -> Path:
    method = method_id(combo)
    directory = run_path(args.output_root, method, seed)
    if (directory / "run_result.json").exists():
        print(f"already complete: {directory}", flush=True)
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("formal Step 9 training requires CUDA")

    train_dataset = SoftMultiChannelDataset(
        labels[labels["split"] == "train"], args.cache_root, combo, RESOLUTION, train=True
    )
    test_dataset = SoftMultiChannelDataset(
        labels[labels["split"] == "test"], args.cache_root, combo, RESOLUTION, train=False
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

    input_channels = len(combo) * 3
    model = build_model(input_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    config = {
        "experiment_name": f"step9_channels_{method}_five_soft_384",
        "benchmark_type": "Five-class Soft-label Multi-channel Benchmark",
        "task": "five-class",
        "label_strategy": "adjacent soft label: true=0.90 and each adjacent=0.05; boundary true=0.90 and only adjacent=0.10",
        "soft_label_definition": soft_label_definitions(),
        "seed": seed,
        "train_sample_count": len(train_dataset),
        "test_sample_count": len(test_dataset),
        "input_resolution": RESOLUTION,
        "input_channel": METHOD_NAMES[method],
        "input_channel_order": list(combo),
        "input_channels": input_channels,
        "channel_fusion": "concatenate each channel's RGB tensor along the channel dimension",
        "model": "ResNet50",
        "architecture_note": "only conv1 input-channel dimension adapts to concatenated RGB groups; no additional layer, width, or depth change",
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "pretrained": "ImageNet ResNet50_Weights.IMAGENET1K_V1; conv1 groups tiled and divided by group count",
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": args.epochs,
        "scheduler": None,
        "loss": "SoftTargetCrossEntropy",
        "augmentation": {
            "letterbox": "proportional resize then symmetric zero padding",
            "synchronized_across_channels": True,
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
        "hard_label_comparison_status": "pending external Step 6 results",
        "data_audit": audit,
    }
    save_json(directory / "config.json", config)

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
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(
            f"method={method} seed={seed} epoch={epoch:03d}/{args.epochs} "
            f"loss={history[-1]['train_loss']:.6f}",
            flush=True,
        )

    torch.save(model.state_dict(), directory / "last.pth")
    metrics, predictions = evaluate(model, test_loader, device)
    predictions.to_csv(directory / "test_predictions.csv", index=False)
    metrics.update(
        {
            "experiment_name": config["experiment_name"],
            "method_id": method,
            "channels": list(combo),
            "input_channel": METHOD_NAMES[method],
            "input_channels": input_channels,
            "resolution": RESOLUTION,
            "seed": seed,
            "checkpoint": "last.pth",
            "epochs_completed": args.epochs,
            "training_seconds": time.perf_counter() - started,
        }
    )
    save_json(directory / "run_result.json", metrics)
    save_json(directory / "metadata.json", {**config, "test_used": True, "result_file": "run_result.json"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return directory


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    records = []
    white_macro = float(runs.loc[runs["method_id"] == "M_baseline", "macro_f1"].astype(float).mean())
    white_accuracy = float(runs.loc[runs["method_id"] == "M_baseline", "accuracy"].astype(float).mean())
    for method in METHODS:
        group = runs[runs["method_id"] == method]
        row: dict[str, object] = {
            "method_id": method,
            "input_channel": METHOD_NAMES[method],
        }
        for metric in METRICS:
            values = group[metric].dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row["macro_f1_delta_vs_white_mean"] = float(row["macro_f1_mean"]) - white_macro
        row["accuracy_delta_vs_white_mean"] = float(row["accuracy_mean"]) - white_accuracy
        records.append(row)
    return pd.DataFrame(records)


def benchmark_chart(
    summary: pd.DataFrame,
    metric: str,
    output: Path,
    majority_accuracy_value: float,
) -> None:
    figure, axis = plt.subplots(figsize=(14, 6.2))
    positions = np.arange(len(summary))
    scale = 1.0 if metric == "mae" else 100.0
    means = summary[f"{metric}_mean"].to_numpy(float) * scale
    stds = summary[f"{metric}_std"].to_numpy(float) * scale
    colors = [CHART_COLORS[method] for method in summary["method_id"]]
    axis.bar(positions, means, yerr=stds, capsize=5, color=colors, edgecolor="#243B53")
    if metric == "accuracy":
        axis.axhline(majority_accuracy_value * 100, color="#6B7A90", linestyle="--", linewidth=1.4)
        axis.text(len(summary) - 0.55, majority_accuracy_value * 100 + 1, f"Majority {majority_accuracy_value * 100:.2f}%")
    axis.set_xticks(positions, [METHOD_NAMES[method] for method in summary["method_id"]], rotation=20, ha="right")
    unit = "grade" if metric == "mae" else "%"
    axis.set_ylabel(f"{metric.replace('_', '-')} ({unit})")
    axis.set_xlabel("Input channel combination")
    axis.set_title(f"Step 9 five-class soft-label multi-channel benchmark: {metric.replace('_', '-')}")
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
            fontsize=8,
        )
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def per_class_chart(runs: pd.DataFrame, metric: str, output: Path) -> None:
    width = 0.13
    figure, axis = plt.subplots(figsize=(15, 6.6))
    for method_index, method in enumerate(METHODS):
        means, stds = [], []
        for class_index in range(5):
            values = runs.loc[runs["method_id"] == method, f"{metric}_class{class_index}"].astype(float)
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        means_array = np.asarray(means) * 100
        stds_array = np.asarray(stds) * 100
        positions = np.arange(5) + (method_index - (len(METHODS) - 1) / 2) * width
        axis.bar(
            positions,
            means_array,
            width=width,
            yerr=stds_array,
            capsize=3,
            label=METHOD_NAMES[method],
            color=CHART_COLORS[method],
        )
        for position, mean, std in zip(positions, means_array, stds_array):
            axis.text(position, min(97, mean + std + 1), f"{mean:.1f}", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(np.arange(5), [f"Class {value}" for value in range(5)])
    axis.set_ylim(0, 100)
    axis.set_xlabel("CEA class")
    axis.set_ylabel(f"Per-class {metric} (%)")
    axis.set_title(f"Step 9 five-class soft-label per-class {metric}")
    axis.legend(ncol=3, fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def confusion_chart(runs: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(3, len(METHODS), figsize=(24, 10))
    for method_index, method in enumerate(METHODS):
        group = runs[runs["method_id"] == method].sort_values("seed")
        for seed_index, (_, row) in enumerate(group.iterrows()):
            axis = axes[seed_index, method_index]
            matrix = np.asarray(
                json.loads(Path(row["run_result_path"]).read_text(encoding="utf-8"))["confusion_matrix"]
            )
            image = axis.imshow(matrix, cmap="Blues", vmin=0)
            axis.set_title(f"{METHOD_NAMES[method]} / seed {int(row['seed'])}", fontsize=9)
            axis.set_xticks(range(5), [f"pred {value}" for value in range(5)])
            axis.set_yticks(range(5), [f"true {value}" for value in range(5)])
            for y in range(5):
                for x in range(5):
                    axis.text(x, y, str(matrix[y, x]), ha="center", va="center", color="#172033", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Step 9 soft-label confusion matrices (baseline, combinations, and formal seeds)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def load_step8_single_channel_context(root: Path) -> dict:
    path = root / "soft_channel_ranking.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Step 8 single-channel ranking: {path}")
    ranking = json.loads(path.read_text(encoding="utf-8"))
    if len(ranking) != len(CHANNELS):
        raise ValueError(f"unexpected Step 8 ranking length: {len(ranking)}")
    best = ranking[0]
    return {
        "ranking_rule": "higher three-seed Macro-F1 mean; tie-break smaller Macro-F1 standard deviation",
        "best_single_channel": best["channel"],
        "best_single_channel_macro_f1_mean": float(best["macro_f1_mean"]),
        "best_single_channel_macro_f1_std": float(best["macro_f1_std"]),
    }


def summarize(args: argparse.Namespace, labels: pd.DataFrame, audit: dict) -> None:
    baseline = load_baseline(args.baseline_root)
    multichannel = load_multichannel_runs(args.output_root)
    runs = pd.concat([baseline, multichannel], ignore_index=True)
    summary = aggregate(runs)
    ranking = summary.sort_values(
        ["macro_f1_mean", "macro_f1_std"], ascending=[False, True], ignore_index=True
    )[
        [
            "method_id",
            "input_channel",
            "macro_f1_mean",
            "macro_f1_std",
            "accuracy_mean",
            "accuracy_std",
            "mae_mean",
            "mae_std",
            "adjacent_error_rate_mean",
            "adjacent_error_rate_std",
            "macro_f1_delta_vs_white_mean",
            "accuracy_delta_vs_white_mean",
        ]
    ]
    save_json(args.output_root / "soft_method_ranking.json", ranking.to_dict(orient="records"))
    runs.drop(columns=["run_result_path"]).to_csv(args.output_root / "soft_run_comparison.csv", index=False)
    summary.to_csv(args.output_root / "soft_benchmark_summary.csv", index=False)
    ranking.to_csv(args.output_root / "soft_method_ranking.csv", index=False)

    baseline_accuracy = majority_accuracy(labels)
    chart_paths: dict[str, Path] = {}
    for metric in METRICS:
        path = args.output_root / f"{metric}_soft_benchmark.png"
        benchmark_chart(summary, metric, path, baseline_accuracy)
        chart_paths[metric] = path
    per_class_paths = {
        metric: args.output_root / f"per_class_{metric}_soft_benchmark.png"
        for metric in PER_CLASS_METRICS
    }
    for metric, path in per_class_paths.items():
        per_class_chart(runs, metric, path)
    confusion_path = args.output_root / "soft_confusion_matrices.png"
    confusion_chart(runs, confusion_path)

    improvements = summary[summary["method_id"] != "M_baseline"]
    worthwhile = improvements[improvements["macro_f1_delta_vs_white_mean"] > 0]
    step8_context = load_step8_single_channel_context(args.baseline_root)
    conclusion = {
        "stage": "Soft-label multi-channel training and standalone benchmark complete",
        "ranking_rule": "higher three-seed Macro-F1 mean; tie-break smaller Macro-F1 standard deviation",
        "highest_ranked_method": ranking.iloc[0]["method_id"],
        "highest_ranked_input": ranking.iloc[0]["input_channel"],
        "multi_channel_methods_beating_white_macro_f1_mean": worthwhile["method_id"].tolist(),
        "any_multi_channel_beats_white_macro_f1_mean": bool(len(worthwhile)),
        "step8_best_single_channel_context": step8_context,
        "hard_label_comparison": "pending external Step 6 results; no Soft-versus-Hard improvement claim is made",
        "final_input_and_label_choice": "not determined until corresponding Hard Label multi-channel results are supplied",
    }
    save_json(args.output_root / "step9_soft_stage_conclusion.json", conclusion)

    predeclared = {
        "combination_source": "the repository's predeclared Step 3 multi-channel combinations, carried forward because no external Step 6 combination list was supplied",
        "combinations": [
            {"method_id": method_id(combo), "channels": list(combo)}
            for combo in COMBOS
        ],
        "all_channel_combination_included": True,
        "white_baseline": "Step 8 M five-class soft-label runs reused without retraining",
    }
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 9 Five-class Soft-label Multi-channel Benchmark</title>
<style>
body {{ font-family: Arial,"Microsoft YaHei",sans-serif; max-width:1280px; margin:32px auto; padding:0 20px; color:#172033; }}
h1,h2 {{ color:#123A63; }} section {{ margin:32px 0; }} img {{ width:100%; height:auto; }}
.metrics {{ border-collapse:collapse; width:100%; font-size:12px; }} .metrics th,.metrics td {{ border:1px solid #D8DEE8; padding:6px; text-align:right; }}
.metrics th {{ background:#EDF3F8; }} pre {{ white-space:pre-wrap; }} .ok {{ color:#137333; font-weight:700; }} .pending {{ color:#A15C00; font-weight:700; }}
</style></head><body>
<h1>Step 9 Five-class Soft-label Multi-channel Benchmark</h1>
<p>Input resolution is fixed at 384 x 384. Task: original CEA 0-4 five-class. Soft labels are generated dynamically from hard labels: interior classes use 0.90 for the true class and 0.05 for each adjacent class; boundary classes use 0.90 and 0.10. ResNet50 ImageNet pretrained; AdamW lr 1e-4, weight decay 1e-4; batch 16; 50 epochs; no validation, scheduler, or early stopping. Seeds: 42, 3407, 2026. Step 8 M (White) is the Soft-label baseline and is not retrained.</p>
<section><h2>Hard-label comparison status</h2><p class="pending">PENDING: the corresponding external Step 6 Hard Label multi-channel results were not supplied. This report does not claim whether Soft Label improves Hard Label and does not make a final input-plus-label choice.</p></section>
<section><h2>Data integrity</h2><p class="ok">PASS</p><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Predeclared combinations</h2><pre>{html.escape(json.dumps(predeclared, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Soft-label method ranking</h2>{frame_html(ranking)}</section>
<section><h2>Mean ± standard deviation</h2>{frame_html(summary)}</section>
<section><h2>Raw three-seed results</h2>{frame_html(runs.drop(columns=['run_result_path']))}</section>
<section><h2>Conclusion</h2><pre>{html.escape(json.dumps(conclusion, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Confusion matrices</h2><img src="{image_data_uri(confusion_path)}" alt="Soft-label confusion matrices"></section>
{''.join(f'<section><h2>Per-class {metric}</h2><img src="{image_data_uri(path)}" alt="Per-class {metric}"></section>' for metric, path in per_class_paths.items())}
{''.join(f'<section><h2>{metric.replace("_", "-")}</h2><img src="{image_data_uri(path)}" alt="{metric} soft benchmark"></section>' for metric, path in chart_paths.items())}
</body></html>"""
    (args.output_root / "step9_soft_stage_report.html").write_text(document, encoding="utf-8")
    print(json.dumps(ranking.to_dict(orient="records"), ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    validate_combinations()
    args = parse_args()
    if args.epochs != 50 or args.batch_size != 16 or RESOLUTION != 384:
        raise ValueError("Step 9 protocol fixes resolution=384, batch_size=16, epochs=50")
    labels, base_audit = split_and_audit(args.cache_root, args.label_csv, args.source_mr69_files)
    audit = {
        **base_audit,
        "benchmark_type": "Five-class Soft-label Multi-channel Benchmark",
        "predeclared_channel_combinations": [list(combo) for combo in COMBOS],
        "multi_channel_matching": "all images matched by sample_id; synchronized geometric and photometric augmentation",
        "hard_label_comparison_status": "pending external Step 6 results",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    save_json(args.output_root / "data_audit.json", audit)
    if args.action == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
    elif args.action == "smoke":
        smoke_test(args, labels, audit)
    elif args.action == "train":
        for combo in COMBOS:
            for seed in SEEDS:
                train_one(args, labels, audit, combo, seed)
    elif args.action == "summarize":
        summarize(args, labels, audit)


if __name__ == "__main__":
    main()
