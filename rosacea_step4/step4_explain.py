#!/usr/bin/env python3
"""Protocol-controlled Step 4 binary DeepLIFT and channel-ablation analysis."""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
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
from rosacea_step2.step2_train import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    METRICS,
    RESOLUTION,
    SEEDS,
    evaluate,
    metric_dict,
    save_json,
    split_and_audit,
)
from rosacea_step3.step3_train import (  # noqa: E402
    COMBOS,
    MultiChannelDataset,
    build_model,
    method_id,
    run_path,
)


APPROVAL_TOKEN = "USER_CONFIRMED_TEST_RANKING_DEVIATION"
PERTURBATION = "replace one channel's RGB group with normalized black pixels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("audit", "smoke", "run", "report", "all"), required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/cache_768"))
    parser.add_argument("--label-csv", type=Path, default=Path("/root/autodl-tmp/rosacea_step2/CEA.csv"))
    parser.add_argument("--step3-root", type=Path, default=Path("/root/autodl-tmp/rosacea_step3/results"))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--representative-method", default="M+MB+MR")
    parser.add_argument(
        "--selection-approval",
        help=(
            "Required formal-run approval marker. For the Test-ranking rule use: "
            "USER_CONFIRMED_TEST_RANKING_DEVIATION"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--smoke-samples", type=int, default=8)
    parser.add_argument(
        "--source-mr69-files",
        default="MR0069.JPG,MR069.JPG",
        help="MR ID=69 files verified and excluded in the original source audit",
    )
    return parser.parse_args()


def parse_combo(method: str) -> tuple[str, ...]:
    for combo in COMBOS:
        if method_id(combo) == method:
            return combo
    valid = [method_id(combo) for combo in COMBOS]
    raise ValueError(f"unknown Step 3 multi-channel method {method!r}; valid methods: {valid}")


def normalized_black_baseline(input_channels: int, device: torch.device) -> torch.Tensor:
    rgb_baseline = torch.tensor(
        [(0.0 - mean) / std for mean, std in zip(IMAGENET_MEAN, IMAGENET_STD)],
        dtype=torch.float32,
        device=device,
    )
    return rgb_baseline.repeat(input_channels // 3)


def load_model(checkpoint: Path, channels: tuple[str, ...], device: torch.device) -> nn.Module:
    model = build_model(len(channels) * 3)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    # Captum needs forward hooks that can be invalidated by in-place activations.
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    # torchvision Bottleneck reuses one ReLU instance for two forward calls;
    # DeepLIFT requires each nonlinear module instance to occur once.
    for module in model.modules():
        if isinstance(module, Bottleneck):
            module.relu1 = nn.ReLU(inplace=False)
            module.relu2 = nn.ReLU(inplace=False)
            module.relu3 = nn.ReLU(inplace=False)
            module.forward = MethodType(bottleneck_forward, module)
    return model


def bottleneck_forward(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    identity = x
    out = self.relu1(self.bn1(self.conv1(x)))
    out = self.relu2(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
        identity = self.downsample(x)
    out += identity
    return self.relu3(out)


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
        for images, targets, _ in loader:
            images = images.to(device, non_blocking=True)
            if ablated_channel is not None:
                group = channels.index(ablated_channel)
                images[:, group * 3 : (group + 1) * 3] = baseline[None, group * 3 : (group + 1) * 3, None, None]
            logits = model(images)
            probability = torch.softmax(logits, dim=1).detach().cpu().numpy()
            predictions = probability.argmax(axis=1)
            probabilities.extend(probability)
            for target, prediction, probability_row in zip(targets.numpy(), predictions, probability):
                records.append(
                    {
                        "true_label": int(target),
                        "prediction": int(prediction),
                        "probability_class0": float(probability_row[0]),
                        "probability_class1": float(probability_row[1]),
                    }
                )
    frame = pd.DataFrame(records)
    metrics = metric_dict(frame["true_label"].tolist(), frame["prediction"].tolist(), np.asarray(probabilities))
    return metrics, frame


def run_ablations(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    channels: tuple[str, ...],
    audit: dict,
    seed: int,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run_path(args.step3_root, args.representative_method, seed) / "last.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing representative checkpoint: {checkpoint}")
    model = load_model(checkpoint, channels, device)
    test_dataset = MultiChannelDataset(
        labels[labels["split"] == "test"], args.cache_root, channels, RESOLUTION, train=False
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    records: list[dict] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    for channel in (None, *channels):
        condition = "original" if channel is None else f"ablate_{channel}"
        metrics, frame = perturbed_frame(model, loader, device, channels, channel)
        prediction_frames[condition] = frame
        metrics.update(
            {
                "seed": seed,
                "condition": condition,
                "ablated_channel": channel,
                "checkpoint": str(checkpoint),
                "perturbation": PERTURBATION,
                "test_sample_count": len(frame),
            }
        )
        records.append(metrics)
        print(f"seed={seed} condition={condition} macro_f1={metrics['macro_f1']:.6f}", flush=True)
    metadata = {
        "representative_method": args.representative_method,
        "channels": list(channels),
        "seed": seed,
        "checkpoint": str(checkpoint),
        "data_audit_passed": audit["passed"],
    }
    return records, prediction_frames


def run_deeplift(
    args: argparse.Namespace,
    labels: pd.DataFrame,
    channels: tuple[str, ...],
    seed: int,
    sample_limit: int | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = run_path(args.step3_root, args.representative_method, seed) / "last.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing representative checkpoint: {checkpoint}")
    model = load_model(checkpoint, channels, device)
    frame = labels[labels["split"] == "test"].reset_index(drop=True)
    if sample_limit is not None:
        frame = frame.head(sample_limit)
    dataset = MultiChannelDataset(frame, args.cache_root, channels, RESOLUTION, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    deep_lift = DeepLift(model)
    baseline_rgb = normalized_black_baseline(3, device)
    records: list[dict] = []
    spatial_sums = {channel: np.zeros((RESOLUTION, RESOLUTION), dtype=np.float64) for channel in channels}
    sample_count = 0

    model.eval()
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            predictions = model(images).argmax(dim=1)
        baselines = baseline_rgb.repeat(len(channels))[None, :, None, None].expand_as(images)
        attributions = deep_lift.attribute(images, baselines=baselines, target=predictions).detach().cpu()
        absolute = attributions.abs()
        channel_sums = {
            channel: absolute[:, index * 3 : (index + 1) * 3].sum(dim=(1, 2, 3)).detach().cpu().numpy()
            for index, channel in enumerate(channels)
        }
        for row_index, target in enumerate(targets.numpy()):
            values = {channel: float(channel_sums[channel][row_index]) for channel in channels}
            total = sum(values.values()) + 1e-12
            records.append(
                {
                    "seed": seed,
                    "test_index": sample_count,
                    "true_label": int(target),
                    "predicted_label": int(predictions[row_index].item()),
                    **{f"deep_lift_abs_sum_{channel}": value for channel, value in values.items()},
                    **{f"deep_lift_share_{channel}": value / total for channel, value in values.items()},
                }
            )
            for index, channel in enumerate(channels):
                spatial_sums[channel] += absolute[row_index, index * 3 : (index + 1) * 3].mean(dim=0).numpy()
            sample_count += 1
    raw = pd.DataFrame(records)
    spatial = {channel: value / max(sample_count, 1) for channel, value in spatial_sums.items()}

    center = RESOLUTION // 4
    region_records = []
    for channel in channels:
        map_value = spatial[channel]
        center_sum = float(map_value[center : RESOLUTION - center, center : RESOLUTION - center].sum())
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
            row[f"{metric}_mean"] = float(numbers.mean()) if len(numbers) else np.nan
            row[f"{metric}_std"] = float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0
        records.append(row)
    return pd.DataFrame(records)


def add_performance_deltas(values: pd.DataFrame, channels: tuple[str, ...]) -> pd.DataFrame:
    for channel in channels:
        baseline = values[values["condition"] == "original"].set_index("seed")
        ablated = values[values["condition"] == f"ablate_{channel}"].set_index("seed")
        for metric in ("accuracy", "macro_f1", "mae", "sensitivity", "specificity", "auc"):
            differences = ablated[metric].to_numpy() - baseline.loc[ablated.index, metric].to_numpy()
            values.loc[values["condition"] == f"ablate_{channel}", f"delta_{metric}"] = differences
    return values


def save_prediction_files(
    args: argparse.Namespace,
    seed: int,
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    directory = args.output_root / "runs" / f"seed{seed}" / "perturbation_predictions"
    directory.mkdir(parents=True, exist_ok=True)
    for condition, frame in prediction_frames.items():
        frame.to_csv(directory / f"{condition}.csv", index=False)


def error_bar_chart(
    frame: pd.DataFrame,
    value_column: str,
    std_column: str,
    label_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.8))
    means = frame[value_column].to_numpy(float) * 100
    stds = frame[std_column].to_numpy(float) * 100
    positions = np.arange(len(frame))
    axis.bar(positions, means, yerr=stds, capsize=5, color="#4C78A8", edgecolor="#243B53")
    axis.set_xticks(positions, frame[label_column], rotation=15, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(0, max(100.0, float(np.max(means + stds)) * 1.15))
    for position, mean, std in zip(positions, means, stds):
        axis.text(position, mean + std + 1.5, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def signed_delta_chart(
    frame: pd.DataFrame,
    value_column: str,
    std_column: str,
    label_column: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    means = frame[value_column].to_numpy(float) * 100
    stds = frame[std_column].to_numpy(float) * 100
    positions = np.arange(len(frame))
    axis.bar(positions, means, yerr=stds, capsize=5, color="#B279A2", edgecolor="#243B53")
    axis.axhline(0, color="#5A6B7F", linestyle="--", linewidth=1)
    axis.set_xticks(positions, frame[label_column])
    axis.set_xlabel("Ablated channel")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    spread = float(np.max(np.abs(np.vstack((means - stds, means + stds)))))
    axis.set_ylim(-max(5.0, spread * 1.2), max(5.0, spread * 1.2))
    for position, mean, std in zip(positions, means, stds):
        label_pad = spread * 0.025
        if mean >= 0:
            label_y = max(mean + std, 0) + label_pad
            va = "bottom"
        else:
            label_y = min(mean - std, 0) - label_pad
            va = "top"
        axis.text(position, label_y, f"{mean:+.2f}±{std:.2f}", ha="center", va=va, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def deep_lift_chart(summary: pd.DataFrame, channels: tuple[str, ...], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    means = [float(summary.loc[summary["channel"] == channel, "share_mean"].iloc[0]) * 100 for channel in channels]
    stds = [float(summary.loc[summary["channel"] == channel, "share_std"].iloc[0]) * 100 for channel in channels]
    positions = np.arange(len(channels))
    axis.bar(positions, means, yerr=stds, capsize=5, color=["#4C78A8", "#F58518", "#54A24B"], edgecolor="#243B53")
    axis.set_xticks(positions, channels)
    axis.set_xlabel("Input channel")
    axis.set_ylabel("Mean DeepLIFT share (%)")
    axis.set_title("DeepLIFT channel importance for predicted class")
    axis.set_ylim(0, max(60.0, float(np.max(np.asarray(means) + np.asarray(stds))) * 1.2))
    for position, mean, std in zip(positions, means, stds):
        axis.text(position, mean + std + 1, f"{mean:.2f}±{std:.2f}", ha="center", va="bottom", fontsize=9)
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
    figure.suptitle("Aggregate DeepLIFT spatial importance (no patient images shown)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def frame_html(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="metrics", escape=True)


def create_report(
    args: argparse.Namespace,
    channels: tuple[str, ...],
    audit: dict,
    perturbation_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
) -> Path:
    perturbation_chart = args.output_root / "perturbation_macro_f1.png"
    accuracy_chart = args.output_root / "perturbation_accuracy.png"
    macro_delta_chart = args.output_root / "perturbation_macro_f1_delta.png"
    accuracy_delta_chart = args.output_root / "perturbation_accuracy_delta.png"
    importance_chart = args.output_root / "deep_lift_channel_importance.png"
    spatial_path = args.output_root / "deep_lift_spatial_importance.png"
    labels = {
        "original": "Original input",
        **{f"ablate_{channel}": f"Ablate {channel}" for channel in channels},
    }
    perturbation_summary["condition_label"] = perturbation_summary["condition"].map(labels)
    error_bar_chart(
        perturbation_summary,
        "macro_f1_mean",
        "macro_f1_std",
        "condition_label",
        "Channel ablation: binary Macro-F1",
        "Macro-F1 (%)",
        perturbation_chart,
    )
    error_bar_chart(
        perturbation_summary,
        "accuracy_mean",
        "accuracy_std",
        "condition_label",
        "Channel ablation: binary Accuracy",
        "Accuracy (%)",
        accuracy_chart,
    )
    changes = perturbation_summary[perturbation_summary["condition"] != "original"].copy()
    changes["change_label"] = changes["condition"].str.removeprefix("ablate_")
    signed_delta_chart(
        changes,
        "delta_macro_f1_mean",
        "delta_macro_f1_std",
        "change_label",
        "Channel ablation: Macro-F1 change",
        "Macro-F1 change (percentage points)",
        macro_delta_chart,
    )
    signed_delta_chart(
        changes,
        "delta_accuracy_mean",
        "delta_accuracy_std",
        "change_label",
        "Channel ablation: Accuracy change",
        "Accuracy change (percentage points)",
        accuracy_delta_chart,
    )

    raw = pd.read_csv(args.output_root / "deep_lift_samples.csv")
    importance_rows = []
    for channel in channels:
        values = raw[f"deep_lift_share_{channel}"].astype(float)
        importance_rows.append(
            {
                "channel": channel,
                "share_mean": float(values.mean()),
                "share_std": float(values.std(ddof=1)),
            }
        )
    importance_summary = pd.DataFrame(importance_rows)
    deep_lift_chart(importance_summary, channels, importance_chart)
    maps: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        array = np.load(args.output_root / "runs" / f"seed{seed}" / "deep_lift_spatial.npy")
        for channel_index, channel in enumerate(channels):
            maps.setdefault(channel, []).append(array[channel_index])
    maps = {channel: np.mean(values, axis=0) for channel, values in maps.items()}
    spatial_chart(maps, spatial_path)

    report_path = args.output_root / "step4_report.html"
    selection = {
        "representative_method": args.representative_method,
        "selection_rule": "highest Step 3 three-seed Macro-F1 mean; tie-break smaller std",
        "approval": APPROVAL_TOKEN,
        "protocol_deviation": True,
        "reason": "explicit user confirmation after Step 3 report; Test ranking was used only to select the explanation target",
    }
    configuration = {
        "task": "binary",
        "task_labels": "CEA 0,1,2 -> class 0; CEA 3,4 -> class 1",
        "representative_channels": list(channels),
        "resolution": RESOLUTION,
        "seeds": list(SEEDS),
        "checkpoint": "Step 3 epoch-50 last.pth",
        "deeplift_library": "captum.attr.DeepLift",
        "deeplift_target": "model-predicted class",
        "deeplift_baseline": "each RGB group replaced by normalized black pixels",
        "channel_importance_statistic": "sum absolute input attribution, then within-sample share across channels",
        "perturbation": PERTURBATION,
        "split": "Step 3 fixed Test split; no retraining and no split change",
    }
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 4 Binary DeepLIFT and Channel Ablation</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1280px; margin: 32px auto; padding: 0 20px; color:#172033; }}
h1,h2 {{ color:#123A63; }} section {{ margin:32px 0; }} img {{ width:100%; height:auto; }}
.metrics {{ border-collapse:collapse; width:100%; font-size:12px; }} .metrics th,.metrics td {{ border:1px solid #D8DEE8; padding:6px; text-align:right; }}
.metrics th {{ background:#EDF3F8; }} pre {{ white-space:pre-wrap; }} .ok {{ color:#137333; font-weight:bold; }} .warn {{ color:#A61C1C; font-weight:bold; }}
</style></head><body>
<h1>Step 4 Binary DeepLIFT and Channel Ablation</h1>
<p>Explanation-only analysis of the representative Step 3 multi-channel model. No parameters were retrained, no checkpoint was selected by Step 4 results, and the fixed Test split was unchanged.</p>
<section><h2>Representative-model selection</h2><p class="warn">Recorded protocol deviation</p><pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Configuration</h2><pre>{html.escape(json.dumps(configuration, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Data integrity</h2><p class="ok">PASS</p><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Channel perturbation performance</h2>{frame_html(perturbation_summary)}<img src="{image_data_uri(perturbation_chart)}" alt="Macro-F1 ablation chart"><img src="{image_data_uri(accuracy_chart)}" alt="Accuracy ablation chart"><img src="{image_data_uri(macro_delta_chart)}" alt="Macro-F1 change chart"><img src="{image_data_uri(accuracy_delta_chart)}" alt="Accuracy change chart"></section>
<section><h2>DeepLIFT channel importance</h2>{frame_html(importance_summary)}<img src="{image_data_uri(importance_chart)}" alt="DeepLIFT channel importance chart"></section>
<section><h2>DeepLIFT spatial regions</h2>{frame_html(region_summary)}<img src="{image_data_uri(spatial_path)}" alt="Aggregate DeepLIFT spatial importance"></section>
</body></html>"""
    report_path.write_text(document, encoding="utf-8")
    return report_path


def validate_formal_selection(args: argparse.Namespace) -> None:
    if args.representative_method != "M+MB+MR":
        raise ValueError("The preregistered approval in this script covers only representative method M+MB+MR")
    if args.selection_approval != APPROVAL_TOKEN:
        raise ValueError(
            "formal Step 4 execution requires --selection-approval "
            "USER_CONFIRMED_TEST_RANKING_DEVIATION after explicit user confirmation"
        )
    if args.batch_size != 16 or RESOLUTION != 384:
        raise ValueError("Step 4 explanation settings require batch_size=16 and the Step 3 resolution=384")


def main() -> None:
    args = parse_args()
    channels = parse_combo(args.representative_method)
    labels, audit = split_and_audit(args.cache_root, args.label_csv, args.source_mr69_files)
    args.output_root.mkdir(parents=True, exist_ok=True)
    audit = {
        **audit,
        "benchmark_type": "Binary DeepLIFT and Channel Perturbation",
        "representative_method": args.representative_method,
        "channel_order": list(channels),
        "perturbation": PERTURBATION,
    }
    save_json(args.output_root / "data_audit.json", audit)
    if args.action == "audit":
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    if args.action in ("smoke", "all"):
        validate_formal_selection(args)
        perturbation_records, _ = run_ablations(
            args, labels[labels["split"] == "test"].head(args.smoke_samples), channels, audit, seed=SEEDS[0]
        )
        deep_lift_raw, _, _ = run_deeplift(
            args, labels, channels, seed=SEEDS[0], sample_limit=args.smoke_samples
        )
        smoke = {
            "passed": bool(audit["passed"] and np.isfinite(deep_lift_raw.iloc[:, 3:].to_numpy(float)).all()),
            "sample_count": args.smoke_samples,
            "perturbation_metrics": perturbation_records,
            "deeplift_columns": deep_lift_raw.columns.tolist(),
        }
        save_json(args.output_root / "smoke_result.json", smoke)
        if not smoke["passed"]:
            raise RuntimeError("Step 4 smoke test failed")
        if args.action == "smoke":
            return

    if args.action in ("run", "all"):
        validate_formal_selection(args)
        all_perturbation: list[dict] = []
        all_deep_lift: list[pd.DataFrame] = []
        all_regions: list[pd.DataFrame] = []
        for seed in SEEDS:
            perturbation_records, prediction_frames = run_ablations(args, labels, channels, audit, seed)
            all_perturbation.extend(perturbation_records)
            save_prediction_files(args, seed, prediction_frames)
            deep_lift_raw, spatial, regions = run_deeplift(args, labels, channels, seed)
            all_deep_lift.append(deep_lift_raw)
            all_regions.append(regions)
            seed_dir = args.output_root / "runs" / f"seed{seed}"
            np.save(seed_dir / "deep_lift_spatial.npy", np.stack([spatial[channel] for channel in channels]))
            deep_lift_raw.to_csv(seed_dir / "deep_lift_samples.csv", index=False)
        perturbation_values = add_performance_deltas(pd.DataFrame(all_perturbation), channels)
        perturbation_values.to_csv(args.output_root / "perturbation_runs.csv", index=False)
        perturbation_summary = summarize(perturbation_values, "condition")
        perturbation_summary.to_csv(args.output_root / "perturbation_summary.csv", index=False)
        deep_lift_values = pd.concat(all_deep_lift, ignore_index=True)
        deep_lift_values.to_csv(args.output_root / "deep_lift_samples.csv", index=False)
        region_values = pd.concat(all_regions, ignore_index=True)
        region_values.to_csv(args.output_root / "deep_lift_region_runs.csv", index=False)
        region_summary = summarize(region_values.rename(columns={"center_share": "center", "periphery_share": "periphery"}), "channel")
        region_summary.to_csv(args.output_root / "deep_lift_region_summary.csv", index=False)
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
        ablation_rows = perturbation_summary[perturbation_summary["condition"] != "original"].copy()
        ablation_rows["macro_f1_mean_decrease"] = (
            float(perturbation_summary.loc[perturbation_summary["condition"] == "original", "macro_f1_mean"].iloc[0])
            - ablation_rows["macro_f1_mean"]
        )
        deep_lift_rank = importance_summary.sort_values(["share_mean", "share_std"], ascending=[False, True])[
            "channel"
        ].tolist()
        ablation_rank = ablation_rows.sort_values("macro_f1_mean_decrease", ascending=False)[
            "condition"
        ].str.removeprefix("ablate_").tolist()
        conclusion = {
            "deeplift_importance_rule": "higher mean within-sample absolute-attribution share",
            "ablation_importance_rule": "larger Macro-F1 mean decrease after channel ablation",
            "deeplift_channel_ranking": deep_lift_rank,
            "ablation_macro_f1_channel_ranking": ablation_rank,
            "representative_method": args.representative_method,
            "protocol_deviation_recorded": True,
        }
        save_json(args.output_root / "step4_conclusion.json", conclusion)
        if args.action == "run":
            return

    if args.action in ("report", "all"):
        perturbation_summary = pd.read_csv(args.output_root / "perturbation_summary.csv")
        perturbation_values = add_performance_deltas(
            pd.read_csv(args.output_root / "perturbation_runs.csv"), channels
        )
        perturbation_values.to_csv(args.output_root / "perturbation_runs.csv", index=False)
        perturbation_summary = summarize(perturbation_values, "condition")
        perturbation_summary.to_csv(args.output_root / "perturbation_summary.csv", index=False)
        importance_summary = pd.read_csv(args.output_root / "deep_lift_channel_summary.csv")
        deep_lift_values = pd.read_csv(args.output_root / "deep_lift_samples.csv")
        importance_summary = pd.DataFrame(
            {
                "channel": channels,
                "share_mean": [
                    float(deep_lift_values[f"deep_lift_share_{channel}"].astype(float).mean())
                    for channel in channels
                ],
                "share_std": [
                    float(deep_lift_values[f"deep_lift_share_{channel}"].astype(float).std(ddof=1))
                    for channel in channels
                ],
            }
        )
        importance_summary.to_csv(args.output_root / "deep_lift_channel_summary.csv", index=False)
        region_values = pd.read_csv(args.output_root / "deep_lift_region_runs.csv")
        region_summary = summarize(
            region_values.rename(columns={"center_share": "center", "periphery_share": "periphery"}),
            "channel",
        )
        region_summary.to_csv(args.output_root / "deep_lift_region_summary.csv", index=False)
        report_path = create_report(
            args, channels, audit, perturbation_summary, importance_summary, region_summary
        )
        print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
