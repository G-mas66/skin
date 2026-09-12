#!/usr/bin/env python3
"""Protocol-controlled Step 10 five-class soft-label DeepLIFT and ablation analysis."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
from pathlib import Path
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import DeepLift
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import Bottleneck


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rosacea_step8.step8_train import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESOLUTION,
    SEEDS,
    metric_dict,
    save_json,
    soft_label_definitions,
    split_and_audit,
)
from rosacea_step9.step9_train import (  # noqa: E402
    COMBOS,
    METHOD_NAMES,
    SoftMultiChannelDataset,
    build_model,
    method_id,
    run_path,
)


APPROVAL_TOKEN = "USER_CONFIRMED_STEP9_TEST_RANKING_SELECTION"
REPRESENTATIVE_METHOD = "M+MB"
PERTURBATION = "replace one channel's RGB group with normalized black pixels"
CLASS_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "smoke", "run", "report", "all"), required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/cache_768"))
    parser.add_argument("--label-csv", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/CEA.csv"))
    parser.add_argument("--step9-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step9/results"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--representative-method", default=REPRESENTATIVE_METHOD)
    parser.add_argument(
        "--selection-approval",
        help=(
            "Required formal-run marker. For the user-confirmed Step 9 Test-ranking rule use: "
            "USER_CONFIRMED_STEP9_TEST_RANKING_SELECTION"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument(
        "--source-mr69-files",
        default="MR0069.JPG,MR069.JPG",
        help="MR ID=69 source files verified and excluded in the original cache audit",
    )
    return parser.parse_args()


def parse_combo(method: str) -> tuple[str, ...]:
    for combo in COMBOS:
        if method_id(combo) == method:
            return combo
    valid = [method_id(combo) for combo in COMBOS]
    raise ValueError(f"unknown Step 9 multi-channel method {method!r}; valid methods: {valid}")


def normalized_black_baseline(input_channels: int, device: torch.device) -> torch.Tensor:
    rgb_baseline = torch.tensor(
        [(0.0 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        dtype=torch.float32,
        device=device,
    )
    return rgb_baseline.repeat(input_channels // 3)


def bottleneck_forward(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    return self.relu3(out)


def load_model(checkpoint: Path, channels: tuple[str, ...], device: torch.device) -> nn.Module:
    model = build_model(len(channels) * 3)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    for module in model.modules():
        if isinstance(module, Bottleneck):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)
            module.forward = MethodType(bottleneck_forward, module)
    return model


def validate_formal_selection(args: argparse.Namespace, channels: tuple[str, ...]) -> dict:
    if args.representative_method != REPRESENTATIVE_METHOD:
        raise ValueError(
            f"the current user confirmation covers only representative method {REPRESENTATIVE_METHOD}"
        )
    if args.selection_approval != APPROVAL_TOKEN:
        raise ValueError(
            "formal Step 10 execution requires --selection-approval "
            "USER_CONFIRMED_STEP9_TEST_RANKING_SELECTION after explicit user confirmation"
        )
    if args.batch_size != 16 or RESOLUTION != 384:
        raise ValueError("Step 10 explanation settings require batch_size=16 and resolution=384")

    ranking_path = args.step9_root / "soft_method_ranking.json"
    conclusion_path = args.step9_root / "step9_soft_stage_conclusion.json"
    if not ranking_path.is_file() or not conclusion_path.is_file():
        raise FileNotFoundError("Step 9 ranking or conclusion is missing")
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
    if conclusion.get("highest_ranked_method") != args.representative_method:
        raise ValueError("representative method does not match the Step 9 conclusion")
    if ranking[0].get("method_id") != args.representative_method:
        raise ValueError("representative method is not highest in the Step 9 Macro-F1 ranking")

    for seed in SEEDS:
        directory = run_path(args.step9_root, args.representative_method, seed)
        config_path = directory / "config.json"
        result_path = directory / "run_result.json"
        checkpoint = directory / "last.pth"
        if not config_path.is_file() or not result_path.is_file() or not checkpoint.is_file():
            raise FileNotFoundError(f"incomplete representative Step 9 run: {directory}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            config.get("task") != "five-class"
            or config.get("epochs") != 50
            or config.get("input_resolution") != RESOLUTION
            or config.get("input_channel_order") != list(channels)
            or result.get("checkpoint") != "last.pth"
            or result.get("epochs_completed") != 50
            or result.get("seed") != seed
        ):
            raise ValueError(f"representative Step 9 run violates the required configuration: {directory}")

    top = ranking[0]
    return {
        "representative_method": args.representative_method,
        "representative_input": METHOD_NAMES[args.representative_method],
        "selection_rule": "highest Step 9 three-seed Macro-F1 mean; tie-break smaller std",
        "approval": APPROVAL_TOKEN,
        "user_confirmation": "current Step 10 request explicitly asked to base the run on Step 8 and Step 9 results",
        "protocol_deviation": True,
        "reason": "Test ranking was used only to select the explanation target, not to train or tune a model",
        "selected_macro_f1_mean": float(top["macro_f1_mean"]),
        "selected_macro_f1_std": float(top["macro_f1_std"]),
        "checkpoints": "all three Step 9 epoch-50 last.pth files",
    }


def perturbed_frame(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    channels: tuple[str, ...],
    ablated_channel: str | None,
) -> tuple[dict, pd.DataFrame]:
    baseline = normalized_black_baseline(len(channels) * 3, device)
    records: list[dict] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, _, targets, ids in loader:
            images = images.to(device, non_blocking=True)
            if ablated_channel is not None:
                group = channels.index(ablated_channel)
                images[:, group * 3 : (group + 1) * 3] = baseline[
                    None, group * 3 : (group + 1) * 3, None, None
                ]
            logits = model(images)
            probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
            predictions = probability.argmax(axis=1)
            probabilities.extend(probability)
            for sample_id, target, prediction, probability_row in zip(
                ids, targets.numpy(), predictions, probability
            ):
                record = {
                    "M_id": str(sample_id),
                    "true_label": int(target),
                    "prediction": int(prediction),
                }
                record.update(
                    {
                        f"probability_class{class_index}": float(probability_row[class_index])
                        for class_index in range(CLASS_COUNT)
                    }
                )
                records.append(record)
    frame = pd.DataFrame(records)
    metrics = metric_dict(frame["true_label"].tolist(), frame["prediction"].tolist())
    return metrics, frame


def run_ablations(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    channels: tuple[str, ...],
    audit: dict,
    seed: int,
    sample_limit: int | None = None,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run_path(args.step9_root, args.representative_method, seed) / "last.pth"
    model = load_model(checkpoint, channels, device)
    frame = labels[labels["split"] == "test"]
    if sample_limit is not None:
        frame = frame.head(sample_limit)
    dataset = SoftMultiChannelDataset(frame, args.cache_root, channels, RESOLUTION, train=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    records: list[dict] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    for channel in (None, *channels):
        condition = "original" if channel is None else f"ablate_{channel}"
        metrics, prediction = perturbed_frame(model, loader, device, channels, channel)
        prediction_frames[condition] = prediction
        metrics.update(
            {
                "seed": seed,
                "condition": condition,
                "ablated_channel": channel,
                "checkpoint": str(checkpoint),
                "perturbation": PERTURBATION,
                "test_sample_count": len(prediction),
                "data_audit_passed": audit["passed"],
            }
        )
        records.append(metrics)
        print(
            f"seed={seed} condition={condition} macro_f1={metrics['macro_f1']:.6f} "
            f"mae={metrics['mae']:.6f}",
            flush=True,
        )
    return records, prediction_frames


def run_deeplift(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    channels: tuple[str, ...],
    seed: int,
    sample_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run_path(args.step9_root, args.representative_method, seed) / "last.pth"
    model = load_model(checkpoint, channels, device)
    frame = labels[labels["split"] == "test"].reset_index(drop=True)
    if sample_limit is not None:
        frame = frame.head(sample_limit)
    dataset = SoftMultiChannelDataset(frame, args.cache_root, channels, RESOLUTION, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    deep_lift = DeepLift(model)
    baseline_rgb = normalized_black_baseline(3, device)
    records: list[dict] = []
    spatial_sums = {
        channel: np.zeros((RESOLUTION, RESOLUTION), dtype=np.float64) for channel in channels
    }
    sample_count = 0

    model.eval()
    for images, _, targets, ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        baselines = baseline_rgb.repeat(len(channels))[None, :, None, None].expand_as(images)
        attributions = deep_lift.attribute(
            images, baselines=baselines, target=predictions
        ).detach().cpu()
        absolute = attributions.abs()
        channel_sums = {
            channel: absolute[:, index * 3 : (index + 1) * 3]
            .sum(dim=(1, 2, 3))
            .detach()
            .cpu()
            .numpy()
            for index, channel in enumerate(channels)
        }
        for row_index, (sample_id, target) in enumerate(zip(ids, targets.numpy())):
            values = {channel: float(channel_sums[channel][row_index]) for channel in channels}
            total = sum(values.values()) + 1e-12
            records.append(
                {
                    "seed": seed,
                    "test_index": sample_count,
                    "M_id": str(sample_id),
                    "true_label": int(target),
                    "predicted_label": int(predictions[row_index].item()),
                    **{
                        f"deep_lift_abs_sum_{channel}": value
                        for channel, value in values.items()
                    },
                    **{
                        f"deep_lift_share_{channel}": value / total
                        for channel, value in values.items()
                    },
                }
            )
            for index, channel in enumerate(channels):
                spatial_sums[channel] += (
                    absolute[row_index, index * 3 : (index + 1) * 3].mean(dim=0).numpy()
                )
            sample_count += 1
    raw = pd.DataFrame(records)
    spatial = {channel: value / max(sample_count, 1) for channel, value in spatial_sums.items()}

    center = RESOLUTION // 4
    region_records = []
    for channel in channels:
        map_value = spatial[channel]
        center_sum = float(
            map_value[center : RESOLUTION - center, center : RESOLUTION - center].sum()
        )
        total_sum = float(map_value.sum()) + 1e-12
        region_records.append(
            {
                "seed": seed,
                "channel": channel,
                "center_share": center_sum / total_sum,
                "periphery_share": (total_sum - center_sum) / total_sum,
            }
        )
    return raw, spatial, pd.DataFrame(region_records)


def summarize(values: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records = []
    for group_value, group in values.groupby(group_column, sort=False):
        row: dict[str, object] = {group_column: group_value, "seed_count": len(group)}
        for metric in values.columns:
            if metric in (group_column, "seed"):
                continue
            numbers = pd.to_numeric(group[metric], errors="coerce").dropna()
            if numbers.empty:
                continue
            row[f"{metric}_mean"] = float(numbers.mean())
            row[f"{metric}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records)


def add_performance_deltas(values: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    for channel in channels:
        baseline = values[values["condition"] == "original"].set_index("seed")
        ablated = values[values["condition"] == f"ablate_{channel}"].set_index("seed")
        for metric in ("accuracy", "macro_f1", "mae", "adjacent_error_rate"):
            differences = ablated[metric].to_numpy() - baseline.loc[ablated.index, metric].to_numpy()
            values.loc[values["condition"] == f"ablate_{channel}", f"delta_{metric}"] = differences
    return values


def save_prediction_files(
    args: argparse.Namespace, seed: int, prediction_frames: dict[str, pd.DataFrame]
) -> None:
    directory = args.output_root / "runs" / f"seed{seed}" / "perturbation_predictions"
    directory.mkdir(parents=True, exist_ok=True)
    for condition, frame in prediction_frames.items():
        frame.to_csv(directory / f"{condition}.csv", index=False)


def metric_chart(
    frame: pd.DataFrame,
    metric: str,
    label_column: str,
    output: Path,
) -> None:
    percentage = metric != "mae"
    scale = 100.0 if percentage else 1.0
    unit = "%" if percentage else "grade"
    means = frame[f"{metric}_mean"].to_numpy(float) * scale
    stds = frame[f"{metric}_std"].to_numpy(float) * scale
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(11, 5.8))
    axis.bar(positions, means, yerr=stds, capsize=5, color="#4C78A8", edgecolor="#243B53")
    axis.set_xticks(positions, frame[label_column], rotation=15, ha="right")
    axis.set_ylabel(f"{metric.replace('_', '-')} ({unit})")
    axis.set_xlabel("Input condition")
    axis.set_title(f"Step 10 soft-label channel ablation: {metric.replace('_', '-')}")
    axis.grid(axis="y", alpha=0.25)
    upper = max(100.0 if percentage else 1.0, float(np.max(means + stds)) * 1.18)
    axis.set_ylim(0, upper)
    for position, mean, std in zip(positions, means, stds):
        axis.text(position, mean + std + upper * 0.015, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def signed_delta_chart(
    frame: pd.DataFrame,
    metric: str,
    label_column: str,
    output: Path,
) -> None:
    percentage = metric != "mae"
    scale = 100.0 if percentage else 1.0
    unit = "percentage points" if percentage else "grade"
    means = frame[f"delta_{metric}_mean"].to_numpy(float) * scale
    stds = frame[f"delta_{metric}_std"].to_numpy(float) * scale
    positions = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.bar(positions, means, yerr=stds, capsize=5, color="#B279A2", edgecolor="#243B53")
    axis.axhline(0, color="#5A6B7F", linestyle="--", linewidth=1)
    axis.set_xticks(positions, frame[label_column])
    axis.set_xlabel("Ablated channel")
    axis.set_ylabel(f"{metric.replace('_', '-')} change ({unit})")
    axis.set_title(f"Step 10 soft-label ablation change: {metric.replace('_', '-')}")
    axis.grid(axis="y", alpha=0.25)
    spread = float(np.max(np.abs(np.vstack((means - stds, means + stds)))))
    bound = max(5.0 if percentage else 0.2, spread * 1.2)
    axis.set_ylim(-bound, bound)
    for position, mean, std in zip(positions, means, stds):
        label_y = max(mean + std, 0) + bound * 0.035 if mean >= 0 else min(mean - std, 0) - bound * 0.035
        alignment = "bottom" if mean >= 0 else "top"
        prefix = "+" if percentage and mean >= 0 else ""
        axis.text(position, label_y, f"{prefix}{mean:.2f}±{std:.2f}", ha="center", va=alignment, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def deep_lift_chart(summary: pd.DataFrame, channels: tuple[str, ...], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    means = [float(summary.loc[summary["channel"] == channel, "share_mean"].iloc[0]) * 100 for channel in channels]
    stds = [float(summary.loc[summary["channel"] == channel, "share_std"].iloc[0]) * 100 for channel in channels]
    positions = np.arange(len(channels))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2"][: len(channels)]
    axis.bar(positions, means, yerr=stds, capsize=5, color=colors, edgecolor="#243B53")
    axis.set_xticks(positions, channels)
    axis.set_xlabel("Input channel")
    axis.set_ylabel("Mean DeepLIFT share (%)")
    axis.set_title("Step 10 soft-label DeepLIFT channel importance")
    axis.set_ylim(0, max(70.0, float(np.max(np.asarray(means) + np.asarray(stds))) * 1.2))
    for position, mean, std in zip(positions, means, stds):
        axis.text(position, mean + std + 1, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def deep_lift_class_chart(
    class_summary: pd.DataFrame,
    channels: tuple[str, ...],
    grouping: str,
    output: Path,
) -> None:
    frame = class_summary[class_summary["grouping"] == grouping].copy()
    classes = sorted(frame["class_label"].unique())
    width = 0.36
    positions = np.arange(len(classes))
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for channel_index, channel in enumerate(channels):
        means, stds = [], []
        for class_label in classes:
            rows = frame[(frame["class_label"] == class_label) & (frame["channel"] == channel)]
            means.append(float(rows["share_mean"].iloc[0]) * 100)
            stds.append(float(rows["share_std"].iloc[0]) * 100)
        offsets = positions + (channel_index - (len(channels) - 1) / 2) * width
        axis.bar(offsets, means, width=width, yerr=stds, capsize=4, label=channel)
        for position, mean, std in zip(offsets, means, stds):
            axis.text(position, mean + std + 1, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(positions, [f"Class {value}" for value in classes])
    axis.set_xlabel(f"{grouping} CEA severity class")
    axis.set_ylabel("Mean DeepLIFT share (%)")
    axis.set_title(f"Step 10 DeepLIFT share by {grouping} severity class")
    axis.set_ylim(0, max(80.0, float(np.max(np.asarray(means) + np.asarray(stds))) * 1.2))
    axis.legend(title="Channel")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def spatial_chart(maps: dict[str, np.ndarray], output: Path) -> None:
    figure, axes = plt.subplots(1, len(maps), figsize=(5.2 * len(maps), 4.8))
    for axis, (channel, map_value) in zip(np.atleast_1d(axes), maps.items()):
        image = axis.imshow(map_value, cmap="magma", vmin=0)
        axis.set_title(f"{channel} aggregate map")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Step 10 aggregate DeepLIFT spatial importance (no patient images shown)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def frame_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="metrics", escape=True)


def class_importance(raw: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    records = []
    for grouping, source_column in (("true", "true_label"), ("predicted", "predicted_label")):
        for class_label, group in raw.groupby(source_column, sort=True):
            for channel in channels:
                values = group[f"deep_lift_share_{channel}"].astype(float)
                records.append(
                    {
                        "grouping": grouping,
                        "class_label": int(class_label),
                        "channel": channel,
                        "sample_count": len(values),
                        "share_mean": float(values.mean()),
                        "share_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
                )
    return pd.DataFrame(records)


def create_report(
    args: argparse.Namespace,
    channels: tuple[str, ...],
    audit: dict,
    selection: dict,
) -> Path:
    perturbation_summary = pd.read_csv(args.output_root / "perturbation_summary.csv")
    importance_summary = pd.read_csv(args.output_root / "deep_lift_channel_summary.csv")
    class_summary = pd.read_csv(args.output_root / "deep_lift_class_summary.csv")
    region_summary = pd.read_csv(args.output_root / "deep_lift_region_summary.csv")
    conclusion = json.loads((args.output_root / "step10_conclusion.json").read_text(encoding="utf-8"))

    labels = {
        "original": "Original input",
        **{f"ablate_{channel}": f"Ablate {channel}" for channel in channels},
    }
    perturbation_summary["condition_label"] = perturbation_summary["condition"].map(labels)
    changes = perturbation_summary[perturbation_summary["condition"] != "original"].copy()
    changes["change_label"] = changes["condition"].str.removeprefix("ablate_")

    chart_specs = {
        "macro_f1": (perturbation_summary, "perturbation_macro_f1.png"),
        "accuracy": (perturbation_summary, "perturbation_accuracy.png"),
        "mae": (perturbation_summary, "perturbation_mae.png"),
        "adjacent_error_rate": (perturbation_summary, "perturbation_adjacent_error_rate.png"),
    }
    for metric, (frame, filename) in chart_specs.items():
        metric_chart(frame, metric, "condition_label", args.output_root / filename)
    for metric in chart_specs:
        signed_delta_chart(changes, metric, "change_label", args.output_root / f"perturbation_{metric}_delta.png")

    deep_lift_chart(importance_summary, channels, args.output_root / "deep_lift_channel_importance.png")
    deep_lift_class_chart(class_summary, channels, "true", args.output_root / "deep_lift_true_class_importance.png")
    deep_lift_class_chart(class_summary, channels, "predicted", args.output_root / "deep_lift_predicted_class_importance.png")
    maps: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        array = np.load(args.output_root / "runs" / f"seed{seed}" / "deep_lift_spatial.npy")
        for channel_index, channel in enumerate(channels):
            maps.setdefault(channel, []).append(array[channel_index])
    maps = {channel: np.mean(values, axis=0) for channel, values in maps.items()}
    spatial_chart(maps, args.output_root / "deep_lift_spatial_importance.png")

    configuration = {
        "task": "five-class",
        "task_labels": "CEA 0 -> 0; 1 -> 1; 2 -> 2; 3 -> 3; 4 -> 4",
        "label_strategy": "adjacent soft labels from Step 9",
        "soft_label_definition": soft_label_definitions(),
        "representative_channels": list(channels),
        "resolution": RESOLUTION,
        "seeds": list(SEEDS),
        "checkpoint": "Step 9 epoch-50 last.pth",
        "deeplift_library": "captum.attr.DeepLift",
        "deeplift_target": "model-predicted class",
        "deeplift_baseline": "each RGB group replaced by normalized black pixels",
        "channel_importance_statistic": "sum absolute input attribution, then within-sample share across channels",
        "perturbation": PERTURBATION,
        "split": "Step 9 fixed Test split; no retraining and no split change",
    }
    hard_comparison = {
        "status": "pending external Step 7 five-class Hard Label explanation results",
        "step4_reference": "Step 4 is the binary Hard-label explanation and is not a direct five-class comparator",
        "claim_made": False,
    }
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 10 Five-class Soft-label DeepLIFT and Channel Ablation</title>
<style>
body {{ font-family: Arial,"Microsoft YaHei",sans-serif; max-width:1280px; margin:32px auto; padding:0 20px; color:#172033; }}
h1,h2 {{ color:#123A63; }} section {{ margin:32px 0; }} img {{ width:100%; height:auto; }}
.metrics {{ border-collapse:collapse; width:100%; font-size:12px; }} .metrics th,.metrics td {{ border:1px solid #D8DEE8; padding:6px; text-align:right; }}
.metrics th {{ background:#EDF3F8; }} pre {{ white-space:pre-wrap; }} .ok {{ color:#137333; font-weight:bold; }} .warn {{ color:#A61C1C; font-weight:bold; }} .pending {{ color:#A15C00; font-weight:bold; }}
</style></head><body>
<h1>Step 10 Five-class Soft-label DeepLIFT and Channel Ablation</h1>
<p>Explanation-only analysis of the representative Step 9 Soft-label multi-channel model. No parameters were retrained, no checkpoint was selected by Step 10 results, and the fixed Test split was unchanged.</p>
<section><h2>Representative-model selection</h2><p class="warn">Recorded protocol deviation</p><pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Configuration</h2><pre>{html.escape(json.dumps(configuration, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Data integrity</h2><p class="ok">PASS</p><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Channel perturbation performance</h2>{frame_html(perturbation_summary)}{''.join(f'<img src="{image_data_uri(args.output_root / filename)}" alt="{metric} ablation chart">' for metric, (_, filename) in chart_specs.items())}{''.join(f'<img src="{image_data_uri(args.output_root / f"perturbation_{metric}_delta.png")}" alt="{metric} delta chart">' for metric in chart_specs)}</section>
<section><h2>DeepLIFT channel importance</h2>{frame_html(importance_summary)}<img src="{image_data_uri(args.output_root / 'deep_lift_channel_importance.png')}" alt="DeepLIFT channel importance"></section>
<section><h2>DeepLIFT by true severity</h2>{frame_html(class_summary[class_summary['grouping'] == 'true'])}<img src="{image_data_uri(args.output_root / 'deep_lift_true_class_importance.png')}" alt="DeepLIFT true-class importance"></section>
<section><h2>DeepLIFT by predicted severity</h2>{frame_html(class_summary[class_summary['grouping'] == 'predicted'])}<img src="{image_data_uri(args.output_root / 'deep_lift_predicted_class_importance.png')}" alt="DeepLIFT predicted-class importance"></section>
<section><h2>DeepLIFT spatial regions</h2>{frame_html(region_summary)}<img src="{image_data_uri(args.output_root / 'deep_lift_spatial_importance.png')}" alt="Aggregate DeepLIFT spatial importance"></section>
<section><h2>Hard-label comparison status</h2><p class="pending">PENDING</p><pre>{html.escape(json.dumps(hard_comparison, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Conclusion</h2><pre>{html.escape(json.dumps(conclusion, ensure_ascii=False, indent=2))}</pre></section>
</body></html>"""
    report_path = args.output_root / "step10_report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def formal_run(args: argparse.Namespace, labels: pd.DataFrame, channels: tuple[str, ...], audit: dict) -> dict:
    all_perturbation: list[dict] = []
    all_deep_lift: list[pd.DataFrame] = []
    all_regions: list[pd.DataFrame] = []
    for seed in SEEDS:
        perturbation_records, prediction_frames = run_ablations(
            args, labels, channels, audit, seed
        )
        all_perturbation.extend(perturbation_records)
        save_prediction_files(args, seed, prediction_frames)
        deep_lift_raw, spatial, regions = run_deeplift(args, labels, channels, seed)
        all_deep_lift.append(deep_lift_raw)
        all_regions.append(regions)
        seed_dir = args.output_root / "runs" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.save(seed_dir / "deep_lift_spatial.npy", np.stack([spatial[channel] for channel in channels]))
        deep_lift_raw.to_csv(seed_dir / "deep_lift_samples.csv", index=False)

    perturbation_values = add_performance_deltas(pd.DataFrame(all_perturbation), channels)
    perturbation_values.to_csv(args.output_root / "perturbation_runs.csv", index=False)
    perturbation_summary = summarize(perturbation_values, "condition")
    perturbation_summary.to_csv(args.output_root / "perturbation_summary.csv", index=False)

    deep_lift_values = pd.concat(all_deep_lift, ignore_index=True)
    deep_lift_values.to_csv(args.output_root / "deep_lift_samples.csv", index=False)
    class_summary = class_importance(deep_lift_values, channels)
    class_summary.to_csv(args.output_root / "deep_lift_class_summary.csv", index=False)
    importance_records = []
    for channel in channels:
        shares = deep_lift_values[f"deep_lift_share_{channel}"].astype(float)
        importance_records.append(
            {
                "channel": channel,
                "share_mean": float(shares.mean()),
                "share_std": float(shares.std(ddof=1)),
            }
        )
    importance_summary = pd.DataFrame(importance_records)
    importance_summary.to_csv(args.output_root / "deep_lift_channel_summary.csv", index=False)
    region_values = pd.concat(all_regions, ignore_index=True)
    region_values.to_csv(args.output_root / "deep_lift_region_runs.csv", index=False)
    region_summary = summarize(
        region_values.rename(columns={"center_share": "center", "periphery_share": "periphery"}),
        "channel",
    )
    region_summary.to_csv(args.output_root / "deep_lift_region_summary.csv", index=False)

    ablation_rows = perturbation_summary[perturbation_summary["condition"] != "original"].copy()
    original_macro = float(
        perturbation_summary.loc[
            perturbation_summary["condition"] == "original", "macro_f1_mean"
        ].iloc[0]
    )
    ablation_rows["macro_f1_mean_decrease"] = original_macro - ablation_rows["macro_f1_mean"]
    deep_lift_rank = importance_summary.sort_values(
        ["share_mean", "share_std"], ascending=[False, True]
    )["channel"].tolist()
    ablation_rank = (
        ablation_rows.sort_values("macro_f1_mean_decrease", ascending=False)["condition"]
        .str.removeprefix("ablate_")
        .tolist()
    )
    dominant_by_true_class = {}
    true_summary = class_summary[class_summary["grouping"] == "true"]
    for class_label, group in true_summary.groupby("class_label"):
        best = group.sort_values(["share_mean", "share_std"], ascending=[False, True]).iloc[0]
        dominant_by_true_class[str(int(class_label))] = str(best["channel"])
    return {
        "deeplift_importance_rule": "higher mean within-sample absolute-attribution share",
        "ablation_importance_rule": "larger Macro-F1 mean decrease after channel ablation",
        "deeplift_channel_ranking": deep_lift_rank,
        "ablation_macro_f1_channel_ranking": ablation_rank,
        "dominant_deep_lift_channel_by_true_class": dominant_by_true_class,
        "representative_method": args.representative_method,
        "protocol_deviation_recorded": True,
        "hard_label_comparison": "pending external Step 7 five-class Hard Label explanation results",
    }


def main() -> None:
    args = parse_args()
    channels = parse_combo(args.representative_method)
    labels, base_audit = split_and_audit(args.cache_root, args.label_csv, args.source_mr69_files)
    selection = validate_formal_selection(args, channels)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        **base_audit,
        "benchmark_type": "Five-class Soft-label DeepLIFT and Channel Perturbation",
        "representative_method": args.representative_method,
        "channel_order": list(channels),
        "perturbation": PERTURBATION,
        "explanation_only": True,
    }
    save_json(args.output_root / "data_audit.json", audit)
    save_json(args.output_root / "representative_selection.json", selection)

    if args.action == "audit":
        print(json.dumps({"audit": audit, "selection": selection}, ensure_ascii=False, indent=2))
        return
    if args.action in ("smoke", "all"):
        perturbation_records, _ = run_ablations(
            args,
            labels,
            channels,
            audit,
            seed=SEEDS[0],
            sample_limit=args.smoke_samples,
        )
        deep_lift_raw, _, _ = run_deeplift(
            args, labels, channels, seed=SEEDS[0], sample_limit=args.smoke_samples
        )
        numeric_columns = [column for column in deep_lift_raw.columns if column.startswith("deep_lift_")]
        smoke = {
            "passed": bool(
                audit["passed"]
                and np.isfinite(deep_lift_raw[numeric_columns].to_numpy(float)).all()
                and all(np.isfinite(record["macro_f1"]) for record in perturbation_records)
            ),
            "sample_count": args.smoke_samples,
            "perturbation_metrics": perturbation_records,
            "deeplift_columns": deep_lift_raw.columns.tolist(),
        }
        save_json(args.output_root / "smoke_result.json", smoke)
        if not smoke["passed"]:
            raise RuntimeError("Step 10 smoke test failed")
        if args.action == "smoke":
            return

    if args.action in ("run", "all"):
        started = time.perf_counter()
        conclusion = formal_run(args, labels, channels, audit)
        conclusion["analysis_seconds"] = time.perf_counter() - started
        save_json(args.output_root / "step10_conclusion.json", conclusion)
        if args.action == "run":
            return
    if args.action in ("report", "all"):
        report_path = create_report(args, channels, audit, selection)
        print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
