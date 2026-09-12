"""Run the five-class single-modality adjacent Soft Label benchmark."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import yaml
from matplotlib.patches import Patch
from torch import nn

from dataset import MODALITIES
from model import build_classifier, count_parameters
from train import (
    LABELS,
    load_protocol_step5_frames,
    make_protocol_step5_loader,
    run_epoch_direct,
    validate_protocol_output_dir,
)

SEEDS = (42, 3407, 2026)
INPUT_SIZE = 384
SOFT_LABEL_RULE = {
    "true_class": 0.9,
    "adjacent_class": 0.05,
    "boundary_adjacent_class": 0.1,
}


class AdjacentSoftLabelCrossEntropy(nn.Module):
    """Cross entropy against a dynamically generated adjacent soft target."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        targets = torch.zeros_like(logits)
        targets.scatter_(1, labels.unsqueeze(1), SOFT_LABEL_RULE["true_class"])
        left = labels > 0
        right = labels < len(LABELS) - 1
        if left.any():
            left_labels = labels[left]
            targets[left, left_labels - 1] = torch.where(
                left_labels == len(LABELS) - 1,
                logits.new_tensor(SOFT_LABEL_RULE["boundary_adjacent_class"]),
                logits.new_tensor(SOFT_LABEL_RULE["adjacent_class"]),
            )
        if right.any():
            right_labels = labels[right]
            targets[right, right_labels + 1] = torch.where(
                right_labels == 0,
                logits.new_tensor(SOFT_LABEL_RULE["boundary_adjacent_class"]),
                logits.new_tensor(SOFT_LABEL_RULE["adjacent_class"]),
            )
        return -(targets * functional.log_softmax(logits, dim=1)).sum(dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--step5-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_step_args(args: argparse.Namespace, modality: str) -> argparse.Namespace:
    return argparse.Namespace(
        modality=modality,
        data_root=args.data_root,
        cache_root=args.cache_root,
    )


def validate_config(config: dict) -> None:
    if int(config["data"]["input_size"]) != INPUT_SIZE:
        raise ValueError("Step 8 requires input_size=384")
    step_config = config["protocol_training"]["step5"]
    expected = {
        "epochs": 50,
        "seeds": [42, 3407, 2026],
        "batch_size": 16,
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
    }
    for key, value in expected.items():
        if step_config.get(key) != value:
            raise ValueError(f"Step 8 requires Step 5 setting {key}={value}")


def validate_step5_reference(step5_root: Path, modality: str, seed: int) -> None:
    metadata_path = step5_root / f"{modality}_seed{seed}" / "metadata.json"
    metrics_path = step5_root / f"{modality}_seed{seed}" / "test_metrics.json"
    if not metadata_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"missing Step 5 Hard Label reference: {modality} seed {seed}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_step": 5,
        "task": "five_class",
        "label_strategy": "hard_label",
        "modality": modality,
        "seed": seed,
        "input_resolution": "384x384",
        "epochs_completed": 50,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"invalid Step 5 reference {modality} seed {seed}: {key}")


def integrity_check(args: argparse.Namespace, config: dict) -> dict:
    train_frame, test_frame = load_protocol_step5_frames(args.manifest_dir)
    if len(train_frame) != 789 or len(test_frame) != 207:
        raise ValueError(f"unexpected fixed split sizes: train={len(train_frame)}, test={len(test_frame)}")
    overlap = set(train_frame["patient_group"]) & set(test_frame["patient_group"])
    if overlap:
        raise ValueError(f"patient leakage between fixed splits: {sorted(overlap)}")
    counts = {}
    step_args = make_step_args(args, MODALITIES[0])
    for modality in MODALITIES:
        step_args.modality = modality
        train_loader, train_dataset = make_protocol_step5_loader(train_frame, step_args, config, False, 42)
        test_loader, test_dataset = make_protocol_step5_loader(test_frame, step_args, config, False, 42)
        del train_loader, test_loader
        if len(train_dataset) != 789 or len(test_dataset) != 207:
            raise ValueError(f"{modality} does not preserve the fixed Train/Test sample counts")
        counts[modality] = {"train_images": len(train_dataset), "test_images": len(test_dataset)}
    return {
        "excluded_ids": [69, 296, 769, 770],
        "train_images": len(train_frame),
        "test_images": len(test_frame),
        "train_patient_groups": int(train_frame["patient_group"].nunique()),
        "test_patient_groups": int(test_frame["patient_group"].nunique()),
        "train_test_patient_overlap": 0,
        "train_class_counts": {str(key): int(value) for key, value in train_frame["label"].value_counts().sort_index().items()},
        "test_class_counts": {str(key): int(value) for key, value in test_frame["label"].value_counts().sort_index().items()},
        "modality_counts": counts,
    }


def save_protocol8_checkpoint(path: Path, model: nn.Module, epoch: int, seed: int, modality: str) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "modality": modality,
            "model_name": "resnet50",
            "protocol_step": 8,
            "task": "five_class",
            "label_strategy": "adjacent_soft_label",
            "soft_label_rule": SOFT_LABEL_RULE,
        },
        path,
    )


def train_and_test(args: argparse.Namespace, config: dict, modality: str, seed: int, root: Path) -> None:
    step_args = make_step_args(args, modality)
    train_frame, test_frame = load_protocol_step5_frames(args.manifest_dir)
    train_loader, train_dataset = make_protocol_step5_loader(train_frame, step_args, config, True, seed)
    test_loader, test_dataset = make_protocol_step5_loader(test_frame, step_args, config, False, seed)
    run_dir = validate_protocol_output_dir(root / f"{modality}_seed{seed}", 8)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("formal Step 8 requires CUDA")
    model = build_classifier(
        "resnet50", 5, INPUT_SIZE, True, float(config["training"]["dropout"]), 3
    ).to(device)
    criterion = AdjacentSoftLabelCrossEntropy()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    epochs = 1 if args.smoke_test else 50
    max_batches = 2 if args.smoke_test else None
    metadata = {
        "protocol_step": 8,
        "task": "five_class",
        "label_strategy": "adjacent_soft_label",
        "soft_label_rule": SOFT_LABEL_RULE,
        "modality": modality,
        "seed": seed,
        "model_name": "resnet50",
        "model_parameter_count": count_parameters(model),
        "pretrained": True,
        "input_size": INPUT_SIZE,
        "input_resolution": "384x384",
        "batch_size": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.0001,
        "epochs_planned": epochs,
        "scheduler": None,
        "early_stopping": False,
        "loss": "AdjacentSoftLabelCrossEntropy",
        "augmentation": config["augmentation"],
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "train_patient_groups": int(train_dataset.frame["patient_group"].nunique()),
        "excluded_ids": [69, 296, 769, 770],
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "smoke_test": args.smoke_test,
        "test_used": False,
        "input_cache": {"mode": "resized_jpeg_once", "root": args.cache_root, "resolution": "384x384"},
    }
    save_json(run_dir / "metadata.json", metadata)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        metrics, _, _, _, _ = run_epoch_direct(
            model, train_loader, criterion, device, optimizer, scaler, "finetune", 1.0, 1, max_batches
        )
        history.append({"epoch": epoch, **metrics})
        save_protocol8_checkpoint(run_dir / "last.pth", model, epoch, seed, modality)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"{modality} seed={seed} epoch={epoch} train_f1={metrics['macro_f1']:.4f}", flush=True)
    metadata.update({"epochs_completed": epochs, "training_seconds": time.perf_counter() - started, "checkpoint": "last.pth"})
    with torch.no_grad():
        metrics, targets, predictions, capture_ids, probabilities = run_epoch_direct(
            model, test_loader, nn.CrossEntropyLoss(reduction="none"), device, None, None, None, 0.0
        )
    save_json(run_dir / "test_metrics.json", metrics)
    prediction = pd.DataFrame({"capture_id": capture_ids, "true_label": targets, "prediction": predictions})
    for label in LABELS:
        prediction[f"probability_{label}"] = [row[label] for row in probabilities]
    prediction.to_csv(run_dir / "test_predictions.csv", index=False)
    metadata.update({"test_used": True, "epochs_completed": epochs})
    save_json(run_dir / "metadata.json", metadata)
    del model
    torch.cuda.empty_cache()


def figure_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def save_figure(figure, path: Path) -> str:
    figure.savefig(path, format="png", dpi=160, bbox_inches="tight")
    uri = figure_uri(figure)
    return uri


def parse_confusion_matrix(value):
    return json.loads(value) if isinstance(value, str) else value


def write_report(root: Path, step5_root: Path, metadata: dict) -> None:
    rows = []
    for modality in MODALITIES:
        for seed in SEEDS:
            hard_dir = step5_root / f"{modality}_seed{seed}"
            soft_dir = root / f"{modality}_seed{seed}"
            hard = json.loads((hard_dir / "test_metrics.json").read_text(encoding="utf-8"))
            soft = json.loads((soft_dir / "test_metrics.json").read_text(encoding="utf-8"))
            for strategy, metrics in (("Hard Label", hard), ("Soft Label", soft)):
                rows.append({
                    "modality": modality,
                    "seed": seed,
                    "strategy": strategy,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "mae": metrics["mae"],
                    "confusion_matrix": metrics["confusion_matrix"],
                })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "run_comparison.csv", index=False)
    summary_rows = []
    for (modality, strategy), group in comparison.groupby(["modality", "strategy"], sort=False):
        row = {"modality": modality, "strategy": strategy, "runs": len(group)}
        for metric in ("accuracy", "macro_f1", "mae"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "benchmark_summary.csv", index=False)
    strategies = ("Hard Label", "Soft Label")
    palette = {"M": "#0072B2", "MB": "#E69F00", "MP": "#009E73", "MR": "#D55E00", "MUV": "#CC79A7"}
    hatches = {"Hard Label": "", "Soft Label": "///"}
    plot_uris = {}
    for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("mae", "MAE")):
        figure, axis = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(MODALITIES))
        width = 0.36
        for index, strategy in enumerate(strategies):
            subset = summary[summary["strategy"] == strategy].set_index("modality").reindex(MODALITIES)
            bars = axis.bar(positions + (index - 0.5) * width, subset[f"{metric}_mean"], width, yerr=subset[f"{metric}_std"], capsize=4, color=[palette[modality] for modality in MODALITIES], hatch=hatches[strategy], edgecolor="#263238")
            for bar, mean, std in zip(bars, subset[f"{metric}_mean"], subset[f"{metric}_std"]):
                axis.annotate(f"{mean:.3f} ± {std:.3f}", (bar.get_x() + bar.get_width() / 2, mean + std), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, rotation=90)
        axis.set_title(f"Step 8 Hard Label vs Soft Label: {title}")
        axis.set_xticks(positions, MODALITIES)
        axis.set_ylabel(title)
        modality_handles = [Patch(facecolor=palette[modality], edgecolor="#263238", label=modality) for modality in MODALITIES]
        strategy_handles = [Patch(facecolor="white", edgecolor="#263238", hatch=hatches[strategy], label=strategy) for strategy in strategies]
        modality_legend = axis.legend(handles=modality_handles, title="Modality", loc="upper left", fontsize=8)
        axis.add_artist(modality_legend)
        axis.legend(handles=strategy_handles, title="Label strategy", loc="upper right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        plot_uris[metric] = save_figure(figure, root / f"step8_{metric}.png")
    figure, axes = plt.subplots(2, len(MODALITIES), figsize=(16, 7), squeeze=False)
    for row_index, strategy in enumerate(strategies):
        for column_index, modality in enumerate(MODALITIES):
            matrices = comparison[(comparison["strategy"] == strategy) & (comparison["modality"] == modality)]["confusion_matrix"].map(parse_confusion_matrix)
            matrix = sum((np.asarray(value, dtype=float) for value in matrices), np.zeros((5, 5)))
            normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
            axis = axes[row_index, column_index]
            axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for row in range(5):
                for column in range(5):
                    axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", fontsize=7)
            axis.set_title(f"{strategy}\n{modality}")
            axis.set_xticks(range(5))
            axis.set_yticks(range(5))
    figure.suptitle("Step 8 normalized test confusion matrices")
    figure.tight_layout()
    confusion_uri = save_figure(figure, root / "step8_confusion_matrices.png")
    document = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>rosacea_step8_report</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1600px;margin:32px auto;padding:0 20px;color:#172033}}section{{margin:30px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dee8;padding:6px;text-align:right}}th{{background:#edf3f8}}pre{{background:#f5f7fa;padding:12px;overflow:auto}}</style></head><body><h1>CEA Step 8 五分类 Soft Label 单通道 Benchmark</h1><p>Soft Label 规则：真实类 0.9；中间类别的相邻类各 0.05；边界类为 0.9/0.1。除 Label Strategy 外，其余条件与 Step 5 Hard Label 保持一致。Test 结果仅用于最终报告，不用于训练、Epoch 或模型选择。</p><section><h2>实际配置与数据完整性</h2><pre>{json.dumps(metadata, ensure_ascii=False, indent=2)}</pre></section><section><h2>Accuracy</h2><img src=\"{plot_uris['accuracy']}\"></section><section><h2>Macro-F1</h2><img src=\"{plot_uris['macro_f1']}\"></section><section><h2>MAE</h2><img src=\"{plot_uris['mae']}\"></section><section><h2>归一化测试集混淆矩阵</h2><img src=\"{confusion_uri}\"></section><section><h2>三 Seed 汇总</h2>{summary.to_html(index=False)}</section><section><h2>逐 Seed 结果</h2>{comparison.drop(columns=['confusion_matrix']).to_html(index=False)}</section></body></html>"""
    (root / "rosacea_step8_report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    step5_root = Path(args.step5_root).resolve()
    if root.name != "rosacea_step8":
        raise ValueError("Step 8 output must be skin/rosacea_step8")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    integrity = integrity_check(args, config)
    for modality in MODALITIES:
        for seed in SEEDS:
            validate_step5_reference(step5_root, modality, seed)
    root.mkdir(parents=True, exist_ok=True)
    modalities = MODALITIES
    seeds = (42,) if args.smoke_test else SEEDS
    for modality in modalities:
        for seed in seeds:
            train_and_test(args, config, modality, seed, root)
    metadata = {
        "protocol_step": 8,
        "task": "five_class",
        "label_strategy": "adjacent_soft_label",
        "soft_label_rule": SOFT_LABEL_RULE,
        "modalities": list(MODALITIES),
        "seeds": list(seeds),
        "input_resolution": "384x384",
        "step5_hard_label_reference": str(step5_root),
        "integrity": integrity,
        "smoke_test": args.smoke_test,
    }
    save_json(root / "analysis_metadata.json", metadata)
    if not args.smoke_test:
        write_report(root, step5_root, metadata)


if __name__ == "__main__":
    main()
