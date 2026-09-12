"""Run the pre-registered Step 7 DeepLIFT and channel-ablation analysis."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from captum.attr import DeepLift
from torch.utils.data import DataLoader

from dataset import DirectCEADataset, MODALITIES, direct_collate
from model import build_classifier
from train import LABELS, classification_metrics, load_protocol_step5_frames, seed_everything

SEEDS = (42, 3407, 2026)
EXPERIMENTS = (
    ("m_mb", "M + MB", ("M", "MB")),
    ("m_mr_mb", "M + MR + MB", ("M", "MR", "MB")),
    ("all_channels", "All channels", MODALITIES),
)
PALETTE = {"M": "#0072B2", "MB": "#E69F00", "MP": "#009E73", "MR": "#D55E00", "MUV": "#CC79A7"}


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


def validate_step6(step6_root: Path, experiment_key: str, experiment_name: str, modalities: tuple[str, ...]) -> None:
    for seed in SEEDS:
        run = step6_root / f"{experiment_key}_seed{seed}"
        metadata_path, checkpoint = run / "metadata.json", run / "last.pth"
        if not metadata_path.exists() or not checkpoint.exists():
            raise FileNotFoundError(f"missing fixed Step 6 {experiment_name} run for seed {seed}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "protocol_step": 6,
            "experiment": experiment_name,
            "seed": seed,
            "task": "five_class",
            "label_strategy": "hard_label",
            "epochs_completed": 50,
            "input_resolution": "384x384",
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Step 6 checkpoint violates pre-registered requirement: {key}")
        if metadata.get("fusion_modalities") != list(modalities):
            raise ValueError(f"Step 6 checkpoint modalities do not match {experiment_name}")


def make_loader(args: argparse.Namespace, config: dict, frame: pd.DataFrame, modalities: tuple[str, ...]) -> tuple[DataLoader, DirectCEADataset]:
    dataset = DirectCEADataset(
        frame,
        args.data_root,
        "ALL",
        MODALITIES,
        (384, 384),
        (384, 384),
        "global",
        False,
        config["augmentation"],
        config["data"].get("excluded_filenames", []),
        color_augmentation=False,
        cache_root=args.cache_root,
        fusion_modalities=modalities,
    )
    batch_size = 2 if args.smoke_test else 4
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=direct_collate), dataset


def evaluate(model, loader, device, modalities: tuple[str, ...], ablate: str | None = None, max_batches: int | None = None) -> dict:
    targets, predictions, groups = [], [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            images = batch["views"][0].to(device, non_blocking=True)
            if ablate:
                start = modalities.index(ablate) * 3
                images = images.clone()
                images[:, start:start + 3] = 0
            predictions.extend(model(images).argmax(dim=1).cpu().tolist())
            targets.extend(batch["target"].tolist())
            groups.extend(batch["patient_group"])
    return classification_metrics(targets, predictions, groups)


def prepare_deeplift_model(model):
    traced = torch.fx.symbolic_trace(model)
    calls: dict[str, int] = {}
    replacement_index = 0
    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue
        module = traced.get_submodule(node.target)
        if type(module) is not torch.nn.ReLU:
            continue
        call_number = calls.get(node.target, 0)
        calls[node.target] = call_number + 1
        module.inplace = False
        if call_number:
            target = f"_step7_relu_{replacement_index}"
            replacement_index += 1
            traced.add_module(target, copy.deepcopy(module))
            traced.get_submodule(target).inplace = False
            node.target = target
    traced.graph.lint()
    traced.recompile()
    return traced


def analyze_seed(model, loader, device, modalities: tuple[str, ...], max_batches: int | None) -> tuple[list[dict], dict]:
    explainer = DeepLift(model)
    total = np.zeros(len(modalities), dtype=float)
    counts = np.zeros(5, dtype=int)
    per_label = np.zeros((5, len(modalities)), dtype=float)
    seen = 0
    model.eval()
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images = batch["views"][0].to(device, non_blocking=True)
        with torch.no_grad():
            target = model(images).argmax(dim=1)
        attribution = explainer.attribute(images, baselines=torch.zeros_like(images), target=target).abs()
        values = attribution.mean(dim=(2, 3)).reshape(-1, len(modalities), 3).mean(dim=2).detach().cpu().numpy()
        labels = batch["target"].numpy()
        total += values.sum(axis=0)
        seen += len(values)
        for label in LABELS:
            mask = labels == label
            if mask.any():
                per_label[label] += values[mask].sum(axis=0)
                counts[label] += int(mask.sum())
    rows = [{"scope": "all", "true_label": "all", "modality": modality, "importance": float(total[i] / seen)} for i, modality in enumerate(modalities)]
    for label in LABELS:
        for index, modality in enumerate(modalities):
            rows.append({"scope": "true_label", "true_label": label, "modality": modality, "importance": float(per_label[label, index] / counts[label]) if counts[label] else np.nan})
    return rows, {"attribution_samples": seen, "attribution_target": "predicted_class", "grouping": "true_label"}


def save_figure(figure, path: Path) -> str:
    figure.savefig(path, format="png", dpi=160, bbox_inches="tight")
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    plt.close(figure)
    return "data:image/png;base64," + encoded


def write_report(root: Path, metadata: dict) -> None:
    importance = pd.read_csv(root / "channel_attribution.csv")
    perturb = pd.read_csv(root / "channel_perturbation.csv")
    importance["true_label"] = importance["true_label"].astype(str)
    label_names = [str(label) for label in LABELS]
    experiment_order = [item[0] for item in EXPERIMENTS]
    experiment_names = {item[0]: item[1] for item in EXPERIMENTS}
    experiment_modalities = {item[0]: item[2] for item in EXPERIMENTS}
    overall = importance[importance["scope"] == "all"].groupby(["experiment", "modality"])["importance"].agg(["mean", "std"]).reset_index()
    figure, axis = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(experiment_order))
    width = 0.14
    for modality_index, modality in enumerate(MODALITIES):
        for experiment_index, experiment_key in enumerate(experiment_order):
            if modality not in experiment_modalities[experiment_key]:
                continue
            row = overall[(overall["experiment"] == experiment_key) & (overall["modality"] == modality)].iloc[0]
            position = positions[experiment_index] + (modality_index - (len(MODALITIES) - 1) / 2) * width
            bar = axis.bar(position, row["mean"], width, yerr=row["std"], capsize=4, color=PALETTE[modality], label=modality if experiment_index == 0 else None)[0]
            axis.annotate(f"{row['mean']:.2e} ± {row['std']:.2e}", (bar.get_x() + bar.get_width() / 2, row["mean"] + row["std"]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7, rotation=90)
    axis.set_xticks(positions, [experiment_names[item] for item in experiment_order])
    axis.set_title("Step 7 DeepLIFT absolute attribution by input modality")
    axis.set_ylabel("Mean absolute attribution")
    axis.legend(title="Modality")
    axis.grid(axis="y", alpha=0.2)
    importance_uri = save_figure(figure, root / "deeplift_overall.png")
    class_figure, class_axes = plt.subplots(1, len(EXPERIMENTS), figsize=(13, 5), squeeze=False)
    for axis, (experiment_key, experiment_name, modalities) in zip(class_axes[0], EXPERIMENTS):
        matrix = importance[(importance["scope"] == "true_label") & (importance["experiment"] == experiment_key)].groupby(["true_label", "modality"])["importance"].mean().unstack().reindex(index=label_names, columns=modalities)
        image = axis.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap="YlOrRd")
        for row_index in range(5):
            for column_index in range(len(modalities)):
                value = matrix.iloc[row_index, column_index]
                if pd.notna(value):
                    axis.text(column_index, row_index, f"{value:.2e}", ha="center", va="center", fontsize=8)
        axis.set_xticks(range(len(modalities)), modalities)
        axis.set_yticks(range(5), [f"Class {label}" for label in LABELS])
        axis.set_title(experiment_name)
    class_figure.suptitle("DeepLIFT attribution grouped by true severity class")
    class_figure.colorbar(image, ax=class_axes.ravel().tolist(), label="Mean absolute attribution")
    class_uri = save_figure(class_figure, root / "deeplift_by_true_class.png")
    deltas = perturb[perturb["condition"] != "baseline"].groupby(["experiment", "condition"])[["macro_f1_drop", "accuracy_drop", "mae_increase"]].agg(["mean", "std"]).reset_index()
    figure, axes = plt.subplots(len(EXPERIMENTS), 3, figsize=(15, 12), squeeze=False)
    for row_index, (experiment_key, experiment_name, modalities) in enumerate(EXPERIMENTS):
        subset = deltas[deltas["experiment"] == experiment_key]
        for axis, metric, title in zip(axes[row_index], ("macro_f1_drop", "accuracy_drop", "mae_increase"), ("Macro-F1 drop", "Accuracy drop", "MAE increase")):
            means, stds, labels = [], [], []
            for modality in modalities:
                condition = f"ablate_{modality}"
                item = subset[subset["condition"] == condition].iloc[0]
                means.append(item[(metric, "mean")])
                stds.append(item[(metric, "std")])
                labels.append(modality)
            bars = axis.bar(labels, means, yerr=stds, capsize=4, color=[PALETTE[item] for item in labels])
            for bar, mean, std in zip(bars, means, stds):
                axis.annotate(f"{mean:.3f} ± {std:.3f}", (bar.get_x() + bar.get_width() / 2, mean + std), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
            axis.set_title(f"{experiment_name}: {title}")
            axis.grid(axis="y", alpha=0.2)
    perturb_uri = save_figure(figure, root / "channel_perturbation.png")
    summary = overall.copy()
    summary["experiment"] = summary["experiment"].map(experiment_names)
    perturb_summary = deltas.copy()
    perturb_summary["experiment"] = perturb_summary["experiment"].map(experiment_names)
    document = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><title>rosacea_step7_report</title><style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1500px;margin:32px auto;padding:0 20px;color:#172033}}section{{margin:30px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #d8dee8;padding:6px;text-align:right}}th{{background:#edf3f8}}pre{{background:#f5f7fa;padding:12px;overflow:auto}}</style></head><body><h1>CEA Step 7 五分类 DeepLIFT + 通道扰动</h1><p>本次分析包含 M + MB、M + MR + MB 和全通道三个经用户确认的 Step 6 代表模型；每组使用三个固定 seed 的 Epoch-50 checkpoint，未使用 Test 指标选择模型。DeepLIFT target 为模型预测类别，归因按真实 CEA class 分组。扰动：将该模型中一个模态对应的归一化后 RGB 三通道置零（ImageNet-mean baseline），其余 Test 流程不变。</p><section><h2>实际配置与完整性记录</h2><pre>{json.dumps(metadata, ensure_ascii=False, indent=2)}</pre></section><section><h2>通道总体重要性</h2><img src=\"data:image/png;base64,{importance_uri.split(',', 1)[1]}\"></section><section><h2>按严重程度分组的重要性</h2><img src=\"data:image/png;base64,{class_uri.split(',', 1)[1]}\"></section><section><h2>通道扰动的性能变化</h2><img src=\"data:image/png;base64,{perturb_uri.split(',', 1)[1]}\"></section><section><h2>DeepLIFT 汇总</h2>{summary.to_html(index=False)}</section><section><h2>扰动汇总</h2>{perturb_summary.to_html(index=False)}</section></body></html>"""
    (root / "rosacea_step7_report.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root, step6_root = Path(args.run_root).resolve(), Path(args.step6_root).resolve()
    if root.name != "rosacea_step7":
        raise ValueError("Step 7 output must be skin/rosacea_step7")
    root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_frame, test_frame = load_protocol_step5_frames(args.manifest_dir)
    if len(train_frame) != 789 or len(test_frame) != 207:
        raise ValueError(f"unexpected fixed split sizes: train={len(train_frame)}, test={len(test_frame)}")
    integrity = {"excluded_ids": [69, 296, 769, 770], "train_images": len(train_frame), "test_images": len(test_frame), "train_patient_groups": int(train_frame["patient_group"].nunique()), "test_patient_groups": int(test_frame["patient_group"].nunique()), "train_test_patient_overlap": int(len(set(train_frame["patient_group"]) & set(test_frame["patient_group"]))), "train_class_counts": {str(key): int(value) for key, value in train_frame["label"].value_counts().sort_index().items()}, "test_class_counts": {str(key): int(value) for key, value in test_frame["label"].value_counts().sort_index().items()}}
    if integrity["train_test_patient_overlap"] != 0:
        raise ValueError("patient leakage between fixed train and test manifests")
    for _, _, modalities in EXPERIMENTS:
        train_dataset = DirectCEADataset(train_frame, args.data_root, "ALL", MODALITIES, (384, 384), (384, 384), "global", False, config["augmentation"], config["data"].get("excluded_filenames", []), color_augmentation=False, fusion_modalities=modalities)
        test_dataset = DirectCEADataset(test_frame, args.data_root, "ALL", MODALITIES, (384, 384), (384, 384), "global", False, config["augmentation"], config["data"].get("excluded_filenames", []), color_augmentation=False, fusion_modalities=modalities)
        if len(train_dataset) != 789 or len(test_dataset) != 207:
            raise ValueError(f"modality availability changed fixed split for {modalities}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Step 7 requires CUDA")
    shutil.copy2(args.config, root / "config_snapshot.yaml")
    all_importance, all_perturb = [], []
    max_batches = 2 if args.smoke_test else None
    for experiment_key, experiment_name, modalities in EXPERIMENTS:
        validate_step6(step6_root, experiment_key, experiment_name, modalities)
        loader, dataset = make_loader(args, config, test_frame, modalities)
        for seed in SEEDS:
            seed_everything(seed)
            run_dir = root / f"{experiment_key}_seed{seed}"
            run_dir.mkdir(exist_ok=True)
            checkpoint = torch.load(step6_root / f"{experiment_key}_seed{seed}" / "last.pth", map_location=device, weights_only=False)
            model = build_classifier("resnet50", 5, 384, False, float(config["training"]["dropout"]), 3 * len(modalities)).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            model = prepare_deeplift_model(model).to(device)
            model.eval()
            importance, analysis = analyze_seed(model, loader, device, modalities, max_batches)
            for row in importance:
                row.update({"experiment": experiment_key, "experiment_name": experiment_name, "seed": seed})
            pd.DataFrame(importance).to_csv(run_dir / "channel_attribution.csv", index=False)
            all_importance.extend(importance)
            baseline = evaluate(model, loader, device, modalities, max_batches=max_batches)
            rows = [{"experiment": experiment_key, "experiment_name": experiment_name, "seed": seed, "condition": "baseline", **baseline, "macro_f1_drop": 0.0, "accuracy_drop": 0.0, "mae_increase": 0.0}]
            for modality in modalities:
                metrics = evaluate(model, loader, device, modalities, modality, max_batches)
                rows.append({"experiment": experiment_key, "experiment_name": experiment_name, "seed": seed, "condition": f"ablate_{modality}", **metrics, "macro_f1_drop": baseline["macro_f1"] - metrics["macro_f1"], "accuracy_drop": baseline["accuracy"] - metrics["accuracy"], "mae_increase": metrics["mae"] - baseline["mae"]})
            pd.DataFrame(rows).to_csv(run_dir / "channel_perturbation.csv", index=False)
            all_perturb.extend(rows)
            save_json(run_dir / "metadata.json", {"protocol_step": 7, "seed": seed, "representative_model": experiment_name, "fusion_modalities": list(modalities), "analysis": analysis, "test_images": len(dataset), "smoke_test": args.smoke_test})
            del model
            torch.cuda.empty_cache()
    pd.DataFrame(all_importance).to_csv(root / "channel_attribution.csv", index=False)
    pd.DataFrame(all_perturb).to_csv(root / "channel_perturbation.csv", index=False)
    save_json(root / "channel_attribution.json", {"rows": all_importance})
    save_json(root / "channel_perturbation.json", {"rows": all_perturb})
    metadata = {"protocol_step": 7, "task": "five_class", "representative_models": [{"key": key, "name": name, "fusion_modalities": list(modalities)} for key, name, modalities in EXPERIMENTS], "seeds": list(SEEDS), "input_resolution": "384x384", "test_images": 207, "deep_lift_target": "predicted_class", "perturbation": "zero normalized RGB triplet", "integrity": integrity, "smoke_test": args.smoke_test}
    save_json(root / "analysis_metadata.json", metadata)
    if not args.smoke_test:
        write_report(root, metadata)


if __name__ == "__main__":
    main()
