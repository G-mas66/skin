"""Create plots, aggregate metrics, and a self-contained HTML benchmark report."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--audit-dir")
    parser.add_argument("--direct-root")
    parser.add_argument("--output")
    parser.add_argument("--include-smoke-tests", action="store_true")
    return parser.parse_args()


def load_runs(root: Path, include_smoke_tests: bool) -> list[dict]:
    runs = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("smoke_test") and not include_smoke_tests:
            continue
        history_path, test_path = run_dir / "history.csv", run_dir / "test_metrics.json"
        if not history_path.exists() or not test_path.exists():
            continue
        run = {"run_dir": str(run_dir), "metadata": metadata,
               "history": pd.read_csv(history_path),
               "test": json.loads(test_path.read_text(encoding="utf-8"))}
        runs.append(run)
    return runs


def summary_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metadata, metrics = run["metadata"], run["test"]
        rows.append({"model": metadata.get("model_name", "unknown"), "seed": metadata.get("seed"),
                     "best_epoch": metadata.get("best_epoch"), "epochs_completed": metadata.get("epochs_completed"),
                     "parameter_count": metadata.get("parameter_count"), "training_seconds": metadata.get("training_seconds"),
                     "accuracy": metrics["accuracy"], "balanced_accuracy": metrics["balanced_accuracy"],
                     "macro_f1": metrics["macro_f1"], "mae": metrics["mae"],
                     "quadratic_weighted_kappa": metrics["quadratic_weighted_kappa"], "run_dir": run["run_dir"]})
    return pd.DataFrame(rows)


def aggregate_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "mae", "quadratic_weighted_kappa"]
    rows = []
    for model, group in summary.groupby("model", sort=True):
        row = {"model": model, "runs": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def save_class_distribution(preprocess: dict, path: Path) -> None:
    counts = {str(label): 0 for label in range(5)}
    for split in preprocess["split_counts"]:
        for label in range(5):
            counts[str(label)] += split[f"label_{label}_count"]
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(list(counts), list(counts.values()), color="#3d7ea6")
    axis.set_xlabel("CEA class")
    axis.set_ylabel("Images")
    axis.set_title("Usable image distribution")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_training_curves(runs: list[dict], path: Path) -> None:
    if not runs:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    for run in runs:
        history = run["history"]
        label = f"{run['metadata'].get('model_name')} s{run['metadata'].get('seed')}"
        axes[0, 0].plot(history["epoch"], history["train_loss"], alpha=.35)
        axes[0, 0].plot(history["epoch"], history["val_loss"], label=label)
        axes[0, 1].plot(history["epoch"], history["train_accuracy"], alpha=.35)
        axes[0, 1].plot(history["epoch"], history["val_accuracy"], label=label)
        axes[1, 0].plot(history["epoch"], history["train_macro_f1"], alpha=.35)
        axes[1, 0].plot(history["epoch"], history["val_macro_f1"], label=label)
        axes[1, 1].plot(history["epoch"], history["val_mae"], label=label)
    titles = [("Loss", "Loss"), ("Accuracy", "Score"), ("Macro-F1", "Score"), ("Validation MAE", "MAE")]
    for axis, (title, ylabel) in zip(axes.flat, titles):
        axis.set_title(title); axis.set_xlabel("Epoch"); axis.set_ylabel(ylabel); axis.grid(alpha=.2)
    axes[1, 1].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_confusions(runs: list[dict], path: Path, normalized: bool) -> None:
    groups = {}
    for run in runs:
        groups.setdefault(run["metadata"].get("model_name", "unknown"), []).append(run["test"])
    if not groups:
        return
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.5), squeeze=False)
    for axis, (model, metrics_list) in zip(axes[0], sorted(groups.items())):
        matrix = sum(np.asarray(item["confusion_matrix"], dtype=float) for item in metrics_list)
        if normalized:
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else None)
        for row in range(5):
            for column in range(5):
                value = f"{matrix[row, column]:.2f}" if normalized else f"{int(matrix[row, column])}"
                axis.text(column, row, value, ha="center", va="center", fontsize=8)
        axis.set_title(model); axis.set_xlabel("Predicted"); axis.set_ylabel("True")
        axis.set_xticks(range(5)); axis.set_yticks(range(5)); fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
    fig.suptitle("Normalized confusion matrix" if normalized else "Confusion matrix")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def save_per_class(runs: list[dict], path: Path) -> None:
    groups = {}
    for run in runs:
        groups.setdefault(run["metadata"].get("model_name", "unknown"), []).append(run["test"])
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, 4 * len(groups)), squeeze=False)
    x = np.arange(5)
    for axis, (model, metrics_list) in zip(axes[:, 0], sorted(groups.items())):
        for metric, offset in (("precision", -.25), ("recall", 0), ("f1", .25)):
            values = np.mean([[item[metric] for item in metrics["per_class"]] for metrics in metrics_list], axis=0)
            axis.bar(x + offset, values, .24, label=metric)
        axis.set_title(model); axis.set_ylim(0, 1); axis.set_xticks(x); axis.set_xlabel("CEA class"); axis.set_ylabel("Score"); axis.legend(); axis.grid(axis="y", alpha=.2)
    fig.suptitle("Per-class test metrics"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def frame_html(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "<p>Not available.</p>"
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        if column not in {"seed", "best_epoch", "epochs_completed", "parameter_count", "runs"}:
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    return display.to_html(index=False, border=0, classes="metrics", escape=True)


def figure_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=170, bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def write_audit_report(audit_dir: Path, output: Path) -> None:
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    classes = pd.read_csv(audit_dir / "class_summary.csv")
    modalities = pd.read_csv(audit_dir / "modality_summary.csv")
    pairing = pd.read_csv(audit_dir / "pairing_summary.csv")
    patients = pd.read_csv(audit_dir / "patient_group_summary.csv")

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.arange(len(classes))
    axes[0, 0].bar(x - 0.25, classes["captures"], 0.25, label="captures")
    axes[0, 0].bar(x, classes["patient_groups"], 0.25, label="patient groups")
    axes[0, 0].bar(x + 0.25, classes["images"], 0.25, label="all modality images")
    axes[0, 0].set_xticks(x, classes["label"])
    axes[0, 0].set_title("Class distribution")
    axes[0, 0].set_xlabel("CEA class")
    axes[0, 0].legend(fontsize=8)

    width = 0.36
    mx = np.arange(len(modalities))
    axes[0, 1].bar(mx - width / 2, modalities["raw_images"], width, label="raw")
    axes[0, 1].bar(
        mx + width / 2, modalities["eligible_images"], width, label="eligible"
    )
    axes[0, 1].set_xticks(mx, modalities["modality"])
    axes[0, 1].set_title("Modality coverage")
    axes[0, 1].legend()

    axes[1, 0].barh(pairing["modality_set"], pairing["capture_count"], color="#3d7ea6")
    axes[1, 0].set_title("Capture pairing completeness")
    axes[1, 0].set_xlabel("Captures")

    bins = np.arange(0.5, patients["class_count"].max() + 1.5, 1)
    axes[1, 1].hist(patients["class_count"], bins=bins, rwidth=0.8, color="#8c5e58")
    axes[1, 1].set_xticks(range(1, int(patients["class_count"].max()) + 1))
    axes[1, 1].set_title("CEA classes per patient group")
    axes[1, 1].set_xlabel("Distinct classes")
    axes[1, 1].set_ylabel("Patient groups")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    chart_uri = figure_data_uri(figure)

    checks = pd.DataFrame(
        [{"check": key, "passed": value} for key, value in summary["checks"].items()]
    )
    status_class = "ok" if summary["passed"] else "bad"
    status_text = "PASS" if summary["passed"] else "REVIEW REQUIRED"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEA 数据审计报告</title><style>
body {{ font-family: Arial,"Microsoft YaHei",sans-serif; max-width:1280px; margin:32px auto; padding:0 20px; color:#172033; }}
h1,h2 {{ color:#123a63; }} section {{ margin:30px 0; }} img {{ width:100%; height:auto; }}
.metrics {{ border-collapse:collapse; width:100%; font-size:13px; }} .metrics th,.metrics td {{ border:1px solid #d8dee8; padding:7px; text-align:right; }}
.metrics th {{ background:#edf3f8; }} .ok {{ color:#137333; font-weight:700; }} .bad {{ color:#b3261e; font-weight:700; }}
code,pre {{ background:#f5f7fa; padding:10px; overflow:auto; }}
</style></head><body>
<h1>CEA 0–4 多模态数据审计</h1>
<p>原始 JPG 只读；未创建图片缓存。标注 capture：{summary['label_checks']['labelled_captures']}；可用 capture：{summary['eligible_captures']}；患者组：{summary['patient_groups']}；可用图片：{summary['eligible_images']}。</p>
<p class="{status_class}">综合检查：{status_text}</p>
<section><h2>关键检查</h2>{frame_html(checks)}</section>
<section><h2>审计可视化</h2><img src="{chart_uri}" alt="CEA data audit charts"></section>
<section><h2>类别统计</h2>{frame_html(classes)}</section>
<section><h2>模态覆盖</h2>{frame_html(modalities)}</section>
<section><h2>配对完整性</h2>{frame_html(pairing)}</section>
<section><h2>患者组分布摘要</h2>{frame_html(pd.DataFrame([summary['patient_group_distribution']]))}</section>
<section><h2>完整聚合审计记录</h2><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre></section>
</body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def load_direct_runs(root: Path) -> list[dict]:
    runs = []
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("smoke_test"):
            continue
        run_dir = metadata_path.parent
        validation_path = run_dir / "validation_metrics.json"
        if not validation_path.exists():
            continue
        runs.append(
            {
                "run_dir": run_dir,
                "run": run_dir.relative_to(root).as_posix(),
                "metadata": metadata,
                "validation": json.loads(validation_path.read_text(encoding="utf-8")),
                "test": json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
                if (run_dir / "test_metrics.json").exists()
                else None,
                "history": pd.read_csv(run_dir / "history.csv")
                if (run_dir / "history.csv").exists()
                else None,
            }
        )
    return runs


def direct_experiment_name(run: dict) -> str:
    metadata = run["metadata"]
    if metadata.get("model_name") == "late_fusion":
        return "+".join(metadata.get("modalities", [])) + " late fusion"
    modality = metadata.get("modality", "unknown")
    if modality == "M":
        return "White-only"
    if modality == "ALL":
        return "All modalities | 15-channel early fusion"
    view = metadata.get("view_mode", "unknown")
    sampling = metadata.get("sampling_mode", "unknown")
    return f"{modality} | {view} | {sampling}"


def direct_comparison_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        metadata = run["metadata"]
        row = {
            "run": run["run"],
            "experiment": direct_experiment_name(run),
            "seed": metadata.get("seed"),
            "modality": metadata.get("modality", "+".join(metadata.get("modalities", []))),
            "view_mode": metadata.get("view_mode"),
            "sampling_mode": metadata.get("sampling_mode"),
            "input_long_edge": metadata.get("input_long_edge"),
            "batch_size": metadata.get("micro_batch_size"),
            "best_epoch": metadata.get("best_epoch"),
            "epochs_completed": metadata.get("epochs_completed"),
            "training_seconds": metadata.get("training_seconds"),
        }
        for split in ("validation", "test"):
            metrics = run[split]
            for metric in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mae",
                "quadratic_weighted_kappa",
            ):
                row[f"{split}_{metric}"] = metrics.get(metric) if metrics else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def direct_aggregate_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_columns = [
        column
        for column in comparison.columns
        if column.startswith("validation_") or column.startswith("test_")
    ]
    for experiment, group in comparison.groupby("experiment", sort=True):
        row = {"experiment": experiment, "runs": len(group)}
        for column in metric_columns:
            valid = group[column].dropna()
            row[f"{column}_mean"] = valid.mean() if len(valid) else np.nan
            row[f"{column}_std"] = valid.std(ddof=0) if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def direct_overview_figure(runs: list[dict], comparison: pd.DataFrame):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    experiments = list(dict.fromkeys(comparison["experiment"]))
    positions = np.arange(len(experiments))
    validation = comparison.groupby("experiment")["validation_macro_f1"].agg(["mean", "std"]).reindex(experiments)
    axes[0, 0].bar(positions, validation["mean"], yerr=validation["std"].fillna(0), color="#3d7ea6")
    axes[0, 0].set_title("Validation Macro-F1")
    axes[0, 0].set_xticks(positions, experiments, rotation=35, ha="right")

    tested = comparison.dropna(subset=["test_macro_f1"])
    test_grouped = tested.groupby("experiment")["test_macro_f1"].agg(["mean", "std"])
    test_names = list(test_grouped.index)
    axes[0, 1].bar(
        np.arange(len(test_names)),
        test_grouped["mean"],
        yerr=test_grouped["std"].fillna(0),
        color="#7d9d68",
    )
    axes[0, 1].set_title("Final test Macro-F1")
    axes[0, 1].set_xticks(np.arange(len(test_names)), test_names, rotation=35, ha="right")

    tested_runs = [run for run in runs if run["test"]]
    recall_groups = {}
    for run in tested_runs:
        recall_groups.setdefault(direct_experiment_name(run), []).append(
            [item["recall"] for item in run["test"]["per_class"]]
        )
    x = np.arange(5)
    width = 0.8 / max(len(recall_groups), 1)
    for index, (name, values) in enumerate(sorted(recall_groups.items())):
        axes[1, 0].bar(
            x - 0.4 + width / 2 + index * width,
            np.mean(values, axis=0),
            width,
            label=name,
        )
    axes[1, 0].set_title("Mean per-class test recall")
    axes[1, 0].set_xticks(x, range(5))
    axes[1, 0].legend(fontsize=7)

    histories = [run for run in runs if run["history"] is not None]
    for run in histories:
        history = run["history"]
        axes[1, 1].plot(
            history["epoch"],
            history["val_macro_f1"],
            alpha=0.65,
            label=f"{direct_experiment_name(run)} s{run['metadata'].get('seed')}",
        )
    axes[1, 1].set_title("Validation Macro-F1 curves")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend(fontsize=6, ncol=2)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_ylim(bottom=0)
    figure.tight_layout()
    return figure


def direct_confusion_figure(runs: list[dict]):
    groups = {}
    for run in runs:
        if run["test"]:
            groups.setdefault(direct_experiment_name(run), []).append(
                np.asarray(run["test"]["confusion_matrix"], dtype=float)
            )
    figure, axes = plt.subplots(1, max(len(groups), 1), figsize=(5 * max(len(groups), 1), 4.5), squeeze=False)
    for axis, (name, matrices) in zip(axes[0], sorted(groups.items())):
        matrix = sum(matrices)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center", fontsize=8)
        axis.set_title(name)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(5))
        axis.set_yticks(range(5))
    figure.suptitle("Normalized test confusion matrices pooled across seeds")
    figure.tight_layout()
    return figure


def write_direct_report(root: Path, output: Path, audit_dir: Path | None) -> None:
    runs = load_direct_runs(root)
    if not runs:
        raise RuntimeError("no completed direct-read runs found")
    comparison = direct_comparison_frame(runs)
    aggregate = direct_aggregate_frame(comparison)
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output.parent / "run_comparison.csv", index=False)
    aggregate.to_csv(output.parent / "benchmark_summary.csv", index=False)
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else {}
    first_metadata = runs[0]["metadata"]
    input_width = first_metadata.get("input_width", "unknown")
    input_height = first_metadata.get("input_height", "unknown")
    batch_size = first_metadata.get("micro_batch_size", "unknown")
    minimum_epochs = selection.get(
        "minimum_epochs", first_metadata.get("minimum_epochs", "not recorded")
    )
    maximum_epochs = selection.get(
        "maximum_epochs", first_metadata.get("maximum_epochs", "not recorded")
    )
    audit = {}
    if audit_dir and (audit_dir / "audit_summary.json").exists():
        audit = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    overview_uri = figure_data_uri(direct_overview_figure(runs, comparison))
    confusion_uri = figure_data_uri(direct_confusion_figure(runs))
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEA 0–4 原图直读实验报告</title><style>body {{ font-family:Arial,"Microsoft YaHei",sans-serif; max-width:1500px; margin:32px auto; padding:0 20px; color:#172033; }} h1,h2 {{ color:#123a63; }} section {{ margin:30px 0; }} img {{ width:100%; height:auto; }} .metrics {{ border-collapse:collapse; width:100%; font-size:11px; }} .metrics th,.metrics td {{ border:1px solid #d8dee8; padding:6px; text-align:right; }} .metrics th {{ background:#edf3f8; }} pre {{ background:#f5f7fa; padding:12px; overflow:auto; }}</style></head><body>
<h1>CEA 0–4 白光与多通道对比实验</h1>
<p>输入 {input_width}×{input_height}，batch size {batch_size}，最少 {minimum_epochs} epochs、最多 {maximum_epochs} epochs，ResNet18 ImageNet pretrained。所有实验复用同一患者级 split；模型选择只使用 validation，test 在预注册实验全部完成后评估。</p>
<section><h2>最终选择</h2><pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>总体可视化</h2><img src="{overview_uri}" alt="overview"></section>
<section><h2>测试集归一化混淆矩阵</h2><img src="{confusion_uri}" alt="confusion matrices"></section>
<section><h2>各实验均值 ± 标准差</h2>{frame_html(aggregate)}</section>
<section><h2>逐 run 指标</h2>{frame_html(comparison)}</section>
<section><h2>数据审计摘要</h2><pre>{html.escape(json.dumps(audit, ensure_ascii=False, indent=2))}</pre></section>
</body></html>"""
    output.write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.direct_root:
        root = Path(args.direct_root).resolve()
        output = Path(args.output).resolve() if args.output else root / "experiment_report.html"
        audit_dir = Path(args.audit_dir).resolve() if args.audit_dir else None
        write_direct_report(root, output, audit_dir)
        print(output)
        return
    if args.audit_dir:
        output = (
            Path(args.output).resolve()
            if args.output
            else Path(args.audit_dir).resolve() / "experiment_report.html"
        )
        write_audit_report(Path(args.audit_dir).resolve(), output)
        print(output)
        return
    if not args.benchmark_root or not args.manifest_dir or not args.output:
        raise ValueError(
            "benchmark mode requires --benchmark-root, --manifest-dir and --output"
        )
    benchmark_root, manifest_dir, output = Path(args.benchmark_root).resolve(), Path(args.manifest_dir).resolve(), Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    assets = output.parent / f"{output.stem}_assets"; assets.mkdir(parents=True, exist_ok=True)
    runs = load_runs(benchmark_root, args.include_smoke_tests)
    preprocess = json.loads((manifest_dir / "preprocess_summary.json").read_text(encoding="utf-8"))
    summary, aggregate = summary_frame(runs), aggregate_frame(summary_frame(runs))
    summary.to_csv(output.parent / "run_comparison.csv", index=False); aggregate.to_csv(output.parent / "benchmark_summary.csv", index=False)
    plot_paths = {}
    for title, function, filename in [("Class distribution", save_class_distribution, "class_distribution.png")]:
        path = assets / filename; function(preprocess, path); plot_paths[title] = path
    path = assets / "training_curves.png"; save_training_curves(runs, path); plot_paths["Training curves"] = path
    for title, normalized, filename in [("Confusion matrix", False, "confusion_matrix.png"), ("Normalized confusion matrix", True, "confusion_matrix_normalized.png")]:
        path = assets / filename; save_confusions(runs, path, normalized)
        if path.exists(): plot_paths[title] = path
    path = assets / "per_class_metrics.png"; save_per_class(runs, path)
    if path.exists(): plot_paths["Per-class metrics"] = path
    leakage = preprocess["leakage_checks"]
    images = "".join(f'<section><h2>{html.escape(title)}</h2><img src="{image_data_uri(path)}" alt="{html.escape(title)}"></section>' for title, path in plot_paths.items())
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CEA 0-4 multi-model benchmark</title><style>body {{ font-family: Arial,sans-serif; max-width: 1280px; margin: 32px auto; padding: 0 20px; color:#172033; }} h1,h2 {{ color:#123a63; }} section {{ margin:32px 0; }} img {{ width:100%; height:auto; }} .metrics {{ border-collapse:collapse; width:100%; font-size:12px; overflow-wrap:anywhere; }} .metrics th,.metrics td {{ border:1px solid #d8dee8; padding:7px; text-align:right; }} .metrics th {{ background:#edf3f8; }} .ok {{ color:#137333; font-weight:700; }}</style></head><body><h1>CEA 0-4 White-light Multi-model Benchmark</h1><p>Usable images: {preprocess['usable_images']}; patient groups: {preprocess['patient_groups']}; cache long edge: {preprocess['cache_long_edge']}; model input: 512×512; evaluated runs: {len(runs)}.</p><p class="ok">Leakage checks passed: {html.escape(str(leakage['passed']))}</p><section><h2>Mean ± standard deviation across seeds</h2>{frame_html(aggregate)}</section><section><h2>Per-run test metrics</h2>{frame_html(summary)}</section>{images}<section><h2>Split, leakage audit, and experiment metadata</h2><pre>{html.escape(json.dumps(preprocess, ensure_ascii=False, indent=2))}</pre></section></body></html>"""
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
