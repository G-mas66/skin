"""Run the five-class multi-channel adjacent Soft Label benchmark."""

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
from torch.utils.data import DataLoader

from dataset import DirectCEADataset, MODALITIES, direct_collate
from model import build_classifier, count_parameters
from train import (
    LABELS,
    load_protocol_step5_frames,
    run_epoch_direct,
    seed_everything,
    seed_worker,
    validate_protocol_output_dir,
)

SEEDS = (42, 3407, 2026)
INPUT_SIZE = 384
SOFT_LABEL_RULE = {"true_class": 0.9, "adjacent_class": 0.05, "boundary_adjacent_class": 0.1}
COMBINATIONS = (
    ("m_mr", "M + MR", ("M", "MR")),
    ("m_mb", "M + MB", ("M", "MB")),
    ("m_mr_mb", "M + MR + MB", ("M", "MR", "MB")),
    ("mr_mb", "MR + MB", ("MR", "MB")),
    ("all_channels", "All channels", MODALITIES),
)
PALETTE = {"m_mr": "#0072B2", "m_mb": "#E69F00", "m_mr_mb": "#009E73", "mr_mb": "#D55E00", "all_channels": "#CC79A7"}


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
    parser.add_argument("--step6-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_config(config: dict) -> None:
    if int(config["data"]["input_size"]) != INPUT_SIZE:
        raise ValueError("Step 9 requires input_size=384")
    expected = {"epochs": 50, "seeds": [42, 3407, 2026], "batch_size": 16, "learning_rate": 0.0001, "weight_decay": 0.0001}
    for key, value in expected.items():
        if config["protocol_training"]["step6"].get(key) != value:
            raise ValueError(f"Step 9 requires Step 6 setting {key}={value}")


def make_dataset(frame: pd.DataFrame, args: argparse.Namespace, config: dict, training: bool, modalities: tuple[str, ...]) -> DirectCEADataset:
    return DirectCEADataset(
        frame,
        args.data_root,
        "ALL",
        MODALITIES,
        (INPUT_SIZE, INPUT_SIZE),
        (INPUT_SIZE, INPUT_SIZE),
        "global",
        training,
        config["augmentation"],
        config["data"].get("excluded_filenames", []),
        color_augmentation=False,
        cache_root=args.cache_root,
        fusion_modalities=modalities,
    )


def make_loader(dataset: DirectCEADataset, config: dict, training: bool, seed: int) -> DataLoader:
    settings = config["protocol_training"]["step6"]
    generator = torch.Generator()
    generator.manual_seed(seed)
    workers = int(settings["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=direct_collate,
        drop_last=False,
    )


def validate_step6_reference(step6_root: Path, key: str, name: str, modalities: tuple[str, ...], seed: int) -> None:
    run = step6_root / f"{key}_seed{seed}"
    metadata_path, metrics_path = run / "metadata.json", run / "test_metrics.json"
    if not metadata_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"missing Step 6 Hard Label reference: {name} seed {seed}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "protocol_step": 6,
        "experiment": name,
        "seed": seed,
        "task": "five_class",
        "label_strategy": "hard_label",
        "epochs_completed": 50,
        "input_resolution": "384x384",
        "fusion_modalities": list(modalities),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(f"invalid Step 6 reference {name} seed {seed}: {field}")


def integrity_check(args: argparse.Namespace, config: dict) -> dict:
    train_frame, test_frame = load_protocol_step5_frames(args.manifest_dir)
    if len(train_frame) != 789 or len(test_frame) != 207:
        raise ValueError(f"unexpected fixed split sizes: train={len(train_frame)}, test={len(test_frame)}")
    if set(train_frame["patient_group"]) & set(test_frame["patient_group"]):
        raise ValueError("patient leakage between fixed splits")
    modality_counts = {}
    for key, name, modalities in COMBINATIONS:
        train_dataset = make_dataset(train_frame, args, config, False, modalities)
        test_dataset = make_dataset(test_frame, args, config, False, modalities)
        if len(train_dataset) != 789 or len(test_dataset) != 207:
            raise ValueError(f"{name} does not preserve the fixed Train/Test sample counts")
        modality_counts[key] = {"name": name, "modalities": list(modalities), "train_images": len(train_dataset), "test_images": len(test_dataset)}
    return {
        "excluded_ids": [69, 296, 769, 770],
        "train_images": len(train_frame),
        "test_images": len(test_frame),
        "train_patient_groups": int(train_frame["patient_group"].nunique()),
        "test_patient_groups": int(test_frame["patient_group"].nunique()),
        "train_test_patient_overlap": 0,
        "train_class_counts": {str(k): int(v) for k, v in train_frame["label"].value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in test_frame["label"].value_counts().sort_index().items()},
        "combination_counts": modality_counts,
    }


def save_checkpoint(path: Path, model: nn.Module, epoch: int, seed: int, key: str, modalities: tuple[str, ...]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "experiment": key,
            "fusion_modalities": list(modalities),
            "model_name": "resnet50",
            "protocol_step": 9,
            "task": "five_class",
            "label_strategy": "adjacent_soft_label",
            "soft_label_rule": SOFT_LABEL_RULE,
        },
        path,
    )


def train_and_test(args: argparse.Namespace, config: dict, key: str, name: str, modalities: tuple[str, ...], seed: int, root: Path) -> None:
    seed_everything(seed)
    train_frame, test_frame = load_protocol_step5_frames(args.manifest_dir)
    train_dataset = make_dataset(train_frame, args, config, True, modalities)
    test_dataset = make_dataset(test_frame, args, config, False, modalities)
    train_loader = make_loader(train_dataset, config, True, seed)
    test_loader = make_loader(test_dataset, config, False, seed)
    run_dir = validate_protocol_output_dir(root / f"{key}_seed{seed}", 9)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("formal Step 9 requires CUDA")
    model = build_classifier("resnet50", 5, INPUT_SIZE, True, float(config["training"]["dropout"]), 3 * len(modalities)).to(device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    criterion = AdjacentSoftLabelCrossEntropy()
    epochs = 1 if args.smoke_test else 50
    max_batches = 2 if args.smoke_test else None
    metadata = {
        "protocol_step": 9,
        "benchmark": "multi_channel_soft_label",
        "task": "five_class",
        "label_strategy": "adjacent_soft_label",
        "soft_label_rule": SOFT_LABEL_RULE,
        "experiment": name,
        "experiment_key": key,
        "fusion_modalities": list(modalities),
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
        metrics, _, _, _, _ = run_epoch_direct(model, train_loader, criterion, device, optimizer, scaler, "finetune", 1.0, 1, max_batches)
        history.append({"epoch": epoch, **metrics})
        save_checkpoint(run_dir / "last.pth", model, epoch, seed, key, modalities)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"{key} seed={seed} epoch={epoch} train_f1={metrics['macro_f1']:.4f}", flush=True)
    metadata.update({"epochs_completed": epochs, "training_seconds": time.perf_counter() - started, "checkpoint": "last.pth"})
    with torch.no_grad():
        metrics, targets, predictions, capture_ids, probabilities = run_epoch_direct(model, test_loader, nn.CrossEntropyLoss(reduction="none"), device, None, None, None, 0.0)
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
    return figure_uri(figure)


def write_report(root: Path, step6_root: Path, metadata: dict) -> None:
    rows = []
    for key, name, modalities in COMBINATIONS:
        for seed in SEEDS:
            hard = json.loads((step6_root / f"{key}_seed{seed}" / "test_metrics.json").read_text(encoding="utf-8"))
            soft = json.loads((root / f"{key}_seed{seed}" / "test_metrics.json").read_text(encoding="utf-8"))
            for strategy, metrics in (("Hard Label", hard), ("Soft Label", soft)):
                rows.append({"experiment_key": key, "experiment": name, "modalities": " + ".join(modalities), "seed": seed, "strategy": strategy, "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"], "mae": metrics["mae"], "confusion_matrix": metrics["confusion_matrix"]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / "run_comparison.csv", index=False)
    summary_rows = []
    for (key, name, strategy), group in comparison.groupby(["experiment_key", "experiment", "strategy"], sort=False):
        row = {"experiment_key": key, "experiment": name, "strategy": strategy, "runs": len(group)}
        for metric in ("accuracy", "macro_f1", "mae"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(root / "benchmark_summary.csv", index=False)
    strategies = ("Hard Label", "Soft Label")
    hatches = {"Hard Label": "", "Soft Label": "///"}
    plot_uris = {}
    positions = np.arange(len(COMBINATIONS))
    for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("mae", "MAE")):
        figure, axis = plt.subplots(figsize=(15, 6))
        width = 0.34
        for index, strategy in enumerate(strategies):
            subset = summary[summary["strategy"] == strategy].set_index("experiment_key").reindex([item[0] for item in COMBINATIONS])
            bars = axis.bar(positions + (index - 0.5) * width, subset[f"{metric}_mean"], width, yerr=subset[f"{metric}_std"], capsize=4, color=[PALETTE[key] for key, _, _ in COMBINATIONS], hatch=hatches[strategy], edgecolor="#263238")
            for bar, mean, std in zip(bars, subset[f"{metric}_mean"], subset[f"{metric}_std"]):
                axis.annotate(f"{mean:.3f} ± {std:.3f}", (bar.get_x() + bar.get_width() / 2, mean + std), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8, rotation=90)
        axis.set_title(f"Step 9 Hard Label vs Soft Label: {title}")
        axis.set_xticks(positions, [name for _, name, _ in COMBINATIONS], rotation=20, ha="right")
        axis.set_ylabel(title)
        modality_handles = [Patch(facecolor=PALETTE[key], edgecolor="#263238", label=name) for key, name, _ in COMBINATIONS]
        strategy_handles = [Patch(facecolor="white", edgecolor="#263238", hatch=hatches[strategy], label=strategy) for strategy in strategies]
        modality_legend = axis.legend(handles=modality_handles, title="Input combination", loc="upper left", fontsize=8)
        axis.add_artist(modality_legend)
        axis.legend(handles=strategy_handles, title="Label strategy", loc="upper right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        plot_uris[metric] = save_figure(figure, root / f"step9_{metric}.png")
    figure, axes = plt.subplots(2, len(COMBINATIONS), figsize=(18, 7), squeeze=False)
    for row_index, strategy in enumerate(strategies):
        for column_index, (key, name, _) in enumerate(COMBINATIONS):
            matrices = comparison[(comparison["strategy"] == strategy) & (comparison["experiment_key"] == key)]["confusion_matrix"]
            matrix = sum((np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=float) for value in matrices), np.zeros((5, 5)))
            normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
            axis = axes[row_index, column_index]
            axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            for row in range(5):
                for column in range(5):
                    axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", fontsize=7)
            axis.set_title(f"{strategy}\n{name}", fontsize=8)
            axis.set_xticks(range(5))
            axis.set_yticks(range(5))
    figure.suptitle("Step 9 normalized test confusion matrices")
    figure.tight_layout()
    confusion_uri = save_figure(figure, root / "step9_confusion_matrices.png")
    document = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>rosacea_step9_report</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1600px;margin:32px auto;padding:0 20px;color:#172033}}section{{margin:30px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dee8;padding:6px;text-align:right}}th{{background:#edf3f8}}pre{{background:#f5f7fa;padding:12px;overflow:auto}}</style></head><body><h1>CEA Step 9 五分类 Soft Label 多通道 Benchmark</h1><p>本次实验重复 Step 6 预先定义的五组多通道输入组合。Soft Label 规则：真实类 0.9；中间类别的相邻类各 0.05；边界类为 0.9/0.1。除 Label Strategy 外，其余条件与对应 Step 6 Hard Label 实验保持一致；Test 结果仅用于最终报告，不用于训练、Epoch 或模型选择。</p><section><h2>实际配置与数据完整性</h2><pre>{json.dumps(metadata, ensure_ascii=False, indent=2)}</pre></section><section><h2>Accuracy</h2><img src=\"{plot_uris['accuracy']}\"></section><section><h2>Macro-F1</h2><img src=\"{plot_uris['macro_f1']}\"></section><section><h2>MAE</h2><img src=\"{plot_uris['mae']}\"></section><section><h2>归一化测试集混淆矩阵</h2><img src=\"{confusion_uri}\"></section><section><h2>三 Seed 汇总</h2>{summary.to_html(index=False)}</section><section><h2>逐 Seed 结果</h2>{comparison.drop(columns=['confusion_matrix']).to_html(index=False)}</section></body></html>"""
    (root / "rosacea_step9_report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    step6_root = Path(args.step6_root).resolve()
    if root.name != "rosacea_step9":
        raise ValueError("Step 9 output must be skin/rosacea_step9")
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    integrity = integrity_check(args, config)
    for key, name, modalities in COMBINATIONS:
        for seed in SEEDS:
            validate_step6_reference(step6_root, key, name, modalities, seed)
    root.mkdir(parents=True, exist_ok=True)
    seeds = (42,) if args.smoke_test else SEEDS
    for key, name, modalities in COMBINATIONS:
        for seed in seeds:
            train_and_test(args, config, key, name, modalities, seed, root)
    metadata = {"protocol_step": 9, "benchmark": "multi_channel_soft_label", "task": "five_class", "label_strategy": "adjacent_soft_label", "soft_label_rule": SOFT_LABEL_RULE, "combinations": [{"key": key, "name": name, "modalities": list(modalities)} for key, name, modalities in COMBINATIONS], "seeds": list(seeds), "input_resolution": "384x384", "step6_hard_label_reference": str(step6_root), "integrity": integrity, "smoke_test": args.smoke_test}
    save_json(root / "analysis_metadata.json", metadata)
    if not args.smoke_test:
        write_report(root, step6_root, metadata)


if __name__ == "__main__":
    main()
