"""Train, validate, or finally test the white-light CEA classifiers."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset import (
    CEADataset,
    DirectCEADataset,
    build_eval_transform,
    build_train_transform,
    direct_collate,
    load_manifest,
    validate_split_isolation,
)
from model import build_classifier, count_parameters

LABELS = list(range(5))
SPLIT_NAMES = ("train", "validation", "test")
MODEL_NAMES = ("resnet18", "vit_small", "mamba_vision_t", "unetpp_encoder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("train", "test", "evaluate", "memory-smoke", "fuse"),
        default="train",
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--model", choices=MODEL_NAMES, default=None)
    parser.add_argument("--loss-mode", choices=("ce", "weighted_ce"), default="ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument(
        "--modality", choices=("M", "MB", "MP", "MR", "MUV", "ALL"), default="M"
    )
    parser.add_argument("--required-modalities", default="M,MB,MP,MR,MUV")
    parser.add_argument("--input-long-edge", default="1024")
    parser.add_argument("--view-mode", choices=("global", "global_local"), default="global")
    parser.add_argument("--sampling-mode", choices=("patient", "patient_class"), default="patient")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--input-runs")
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["project"]["class_labels"] != LABELS:
        raise ValueError("config class_labels must be [0, 1, 2, 3, 4]")
    return config


def validate_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).resolve()
    if run_dir == SCRIPT_DIR or SCRIPT_DIR in run_dir.parents:
        raise ValueError("run-dir must be outside the skin source directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_free_space(path: Path, minimum_gb: float) -> None:
    free_gb = shutil.disk_usage(path).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(
            f"free space guard stopped the run: {free_gb:.2f} GB < {minimum_gb:.2f} GB"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_patient_weights(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = result["patient_group"].value_counts()
    result["sample_weight"] = result["patient_group"].map(lambda group: 1.0 / float(counts[group]))
    result["sample_weight"] *= len(result) / result["sample_weight"].sum()
    return result


def make_loader(frame: pd.DataFrame, cache_root: Path, transform, batch_size: int, num_workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = CEADataset(frame, cache_root, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
                      worker_init_fn=seed_worker, generator=generator, drop_last=False)


def parse_modalities(value: str) -> list[str]:
    modalities = [item.strip().upper() for item in value.split(",") if item.strip()]
    valid = {"M", "MB", "MP", "MR", "MUV"}
    invalid = sorted(set(modalities) - valid)
    if not modalities or invalid:
        raise ValueError(f"invalid required modalities: {invalid or modalities}")
    return modalities


def input_dimensions(value: str | int, config: dict) -> tuple[int, int]:
    text = str(value).strip().lower()
    if text == "native":
        return int(config["data"]["native_width"]), int(
            config["data"]["native_height"]
        )
    long_edge = int(text)
    if long_edge <= 0:
        raise ValueError("input long edge must be positive or 'native'")
    width = max(32, round(long_edge * 3 / 4 / 32) * 32)
    return width, long_edge


def input_channels_for_modality(modality: str) -> int:
    return 15 if modality.upper() == "ALL" else 3


def should_stop_early(
    epoch: int, epochs_without_improvement: int, minimum_epochs: int, patience: int
) -> bool:
    return epoch >= minimum_epochs and epochs_without_improvement >= patience


def sampling_weights(frame: pd.DataFrame, mode: str, max_class_multiplier: float) -> np.ndarray:
    patient_counts = frame["patient_group"].value_counts()
    weights = frame["patient_group"].map(
        lambda group: 1.0 / float(patient_counts[group])
    ).to_numpy(dtype=np.float64).copy()
    if mode == "patient_class":
        class_counts = frame["label"].value_counts().reindex(LABELS, fill_value=0)
        if (class_counts == 0).any():
            raise ValueError("patient_class sampling requires every class")
        largest = float(class_counts.max())
        multipliers = np.sqrt(largest / class_counts.astype(float)).clip(
            upper=max_class_multiplier
        )
        weights *= frame["label"].map(multipliers).to_numpy(dtype=np.float64)
    weights *= len(weights) / weights.sum()
    return weights


def make_direct_loader(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    config: dict,
    training: bool,
    seed: int,
) -> tuple[DataLoader, DirectCEADataset]:
    global_size = input_dimensions(args.input_long_edge, config)
    local_long_edge = min(
        int(config["data"]["local_crop_long_edge"]), global_size[1]
    )
    local_size = input_dimensions(local_long_edge, config)
    dataset = DirectCEADataset(
        frame,
        args.data_root,
        args.modality,
        parse_modalities(args.required_modalities),
        global_size,
        local_size,
        args.view_mode,
        training,
        config["augmentation"],
        config["data"].get("excluded_filenames", []),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    if training:
        weights = sampling_weights(
            dataset.frame,
            args.sampling_mode,
            float(config["direct_training"]["class_sampling_max_multiplier"]),
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    num_workers = int(config["training"]["num_workers"])
    loader = DataLoader(
        dataset,
        batch_size=int(config["direct_training"]["micro_batch_size"]),
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=direct_collate,
        drop_last=False,
    )
    return loader, dataset


def classification_metrics(
    targets: list[int], predictions: list[int], patient_groups: list[str] | None = None
) -> dict:
    matrix = confusion_matrix(targets, predictions, labels=LABELS)
    precision, recall, per_class_f1, support = precision_recall_fscore_support(targets, predictions, labels=LABELS, zero_division=0)
    kappa = cohen_kappa_score(targets, predictions, labels=LABELS, weights="quadratic")
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, labels=LABELS, average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(np.asarray(targets) - np.asarray(predictions)))),
        "quadratic_weighted_kappa": float(np.nan_to_num(kappa, nan=0.0)),
        "per_class": [{"label": label, "precision": float(precision[label]), "recall": float(recall[label]), "f1": float(per_class_f1[label]), "support": int(support[label])} for label in LABELS],
        "confusion_matrix": matrix.astype(int).tolist(),
    }
    if patient_groups is not None:
        if len(patient_groups) != len(targets):
            raise ValueError("patient_groups must align with targets")
        counts = pd.Series(patient_groups).value_counts()
        sample_weights = np.asarray([1.0 / counts[group] for group in patient_groups])
        sample_weights *= len(sample_weights) / sample_weights.sum()
        patient_kappa = cohen_kappa_score(
            targets,
            predictions,
            labels=LABELS,
            weights="quadratic",
            sample_weight=sample_weights,
        )
        result["patient_group_macro"] = {
            "accuracy": float(
                accuracy_score(targets, predictions, sample_weight=sample_weights)
            ),
            "macro_f1": float(
                f1_score(
                    targets,
                    predictions,
                    labels=LABELS,
                    average="macro",
                    sample_weight=sample_weights,
                    zero_division=0,
                )
            ),
            "mae": float(
                np.average(
                    np.abs(np.asarray(targets) - np.asarray(predictions)),
                    weights=sample_weights,
                )
            ),
            "quadratic_weighted_kappa": float(
                np.nan_to_num(patient_kappa, nan=0.0)
            ),
        }
    return result


def class_weights(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    counts = frame["label"].value_counts().reindex(LABELS, fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError("weighted CE requires every class in the training split")
    return torch.tensor(len(frame) / (len(LABELS) * counts.astype(np.float64)), dtype=torch.float32, device=device)


def run_epoch(model, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer, scaler,
              stage: str | None, gradient_clip_norm: float, accumulation_steps: int = 1,
              max_batches: int | None = None) -> tuple[dict, list[int], list[int], list[str]]:
    training = optimizer is not None
    if training:
        model.set_training_stage(stage or "warmup")
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
    total_loss = 0.0
    total_items = 0
    targets, predictions, image_paths = [], [], []
    active_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    for batch_index, batch in enumerate(loader):
        if batch_index >= active_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["target"].to(device, non_blocking=True)
        sample_weights = batch["sample_weight"].to(device, non_blocking=True)
        amp_enabled = device.type == "cuda" and scaler is not None
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            per_item_loss = criterion(logits, labels)
            loss = (per_item_loss * sample_weights).mean() if training else per_item_loss.mean()
        if training:
            scaled_loss = loss / accumulation_steps
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == active_batches
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
        image_paths.extend(list(batch["image_path"]))
    metrics = classification_metrics(targets, predictions)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, targets, predictions, image_paths


def run_epoch_direct(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer,
    scaler,
    stage: str | None,
    gradient_clip_norm: float,
    accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> tuple[dict, list[int], list[int], list[int], list[list[float]]]:
    training = optimizer is not None
    if training:
        model.set_training_stage(stage or "warmup")
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()
    total_loss = 0.0
    total_items = 0
    targets: list[int] = []
    predictions: list[int] = []
    capture_ids: list[int] = []
    patient_groups: list[str] = []
    probabilities: list[list[float]] = []
    active_batches = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, batch in enumerate(loader):
        if batch_index >= active_batches:
            break
        labels = batch["target"].to(device, non_blocking=True)
        sample_weight = batch["sample_weight"].to(device, non_blocking=True)
        view_logits = []
        view_losses = []
        for view in batch["views"]:
            images = view.to(device, non_blocking=True)
            amp_enabled = device.type == "cuda" and scaler is not None
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = (criterion(logits, labels) * sample_weight).mean()
            view_logits.append(logits.detach())
            view_losses.append(float(loss.detach().item()))
            if training:
                scaled_loss = loss / (accumulation_steps * len(batch["views"]))
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        if training:
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == active_batches
            )
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        mean_probabilities = torch.stack(
            [torch.softmax(logits, dim=1) for logits in view_logits]
        ).mean(dim=0)
        batch_size = int(labels.size(0))
        batch_predictions = mean_probabilities.argmax(dim=1).cpu().tolist()
        total_loss += float(np.mean(view_losses)) * batch_size
        total_items += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(batch_predictions)
        probabilities.extend(mean_probabilities.cpu().tolist())
        capture_ids.extend(int(value) for value in batch["capture_id"])
        patient_groups.extend(str(value) for value in batch["patient_group"])
    metrics = classification_metrics(targets, predictions, patient_groups)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics, targets, predictions, capture_ids, probabilities


def flat_history_row(epoch: int, stage: str, train_metrics: dict, val_metrics: dict) -> dict:
    row = {"epoch": epoch, "stage": stage}
    for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
        for key in ("loss", "accuracy", "balanced_accuracy", "macro_f1", "mae", "quadratic_weighted_kappa"):
            row[f"{prefix}_{key}"] = metrics[key]
    return row


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(path: Path, model, epoch: int, stage: str, validation_metrics: dict, seed: int, loss_mode: str, model_name: str) -> None:
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "stage": stage,
                "validation_metrics": validation_metrics, "seed": seed, "loss_mode": loss_mode,
                "model_name": model_name}, path)


def train(args: argparse.Namespace, config: dict, run_dir: Path) -> None:
    training_config = config["training"]
    model_name = args.model or config["models"]["names"][0]
    if args.seed not in training_config["seeds"] and not args.smoke_test:
        raise ValueError(f"seed must be one of {training_config['seeds']}")
    seed_everything(args.seed)
    manifest_dir = Path(args.manifest_dir).resolve()
    train_frame = add_patient_weights(load_manifest(manifest_dir / "train.csv"))
    validation_frame = load_manifest(manifest_dir / "validation.csv")
    validate_split_isolation({"train": train_frame, "validation": validation_frame})
    input_size = int(config["data"]["input_size"])
    batch_size = int(training_config["batch_size"])
    accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    num_workers = int(training_config["num_workers"])
    train_loader = make_loader(train_frame, manifest_dir, build_train_transform(input_size, config["augmentation"]), batch_size, num_workers, True, args.seed)
    validation_loader = make_loader(validation_frame, manifest_dir, build_eval_transform(input_size), batch_size, num_workers, False, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = bool(config["models"].get("pretrained", True)) and not args.no_pretrained
    pretrained_error = None
    try:
        model = build_classifier(model_name, 5, input_size, pretrained, float(training_config["dropout"])).to(device)
    except Exception as exc:
        if not pretrained:
            raise
        pretrained_error = f"{type(exc).__name__}: {exc}"
        print(f"pretrained initialization failed for {model_name}; retrying with random initialization: {pretrained_error}", flush=True)
        model = build_classifier(model_name, 5, input_size, False, float(training_config["dropout"])).to(device)
    effective_pretrained = pretrained and getattr(model, "implementation", "") != "dependency_free_vision_mamba_fallback"
    weights = class_weights(train_frame, device) if args.loss_mode == "weighted_ce" else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(training_config.get("label_smoothing", 0.0)), reduction="none")
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" and not args.no_amp else None
    max_batches = 2 if args.smoke_test else None
    warmup_epochs = 1 if args.smoke_test else int(training_config["warmup_epochs"])
    finetune_epochs = 1 if args.smoke_test else int(training_config["finetune_epochs"])
    metadata = {"seed": args.seed, "model_name": model_name, "model_implementation": getattr(model, "implementation", model_name), "requested_pretrained": pretrained, "pretrained": effective_pretrained, "pretrained_error": pretrained_error,
                "num_classes": 5, "input_size": input_size, "cache_long_edge": int(config["data"]["cache_long_edge"]),
                "batch_size": batch_size, "gradient_accumulation_steps": accumulation_steps,
                "effective_batch_size": batch_size * accumulation_steps, "patient_balanced": bool(training_config.get("patient_balanced", False)),
                "parameter_count": count_parameters(model), "train_images": len(train_frame), "validation_images": len(validation_frame),
                "train_patient_groups": int(train_frame["patient_group"].nunique()), "validation_patient_groups": int(validation_frame["patient_group"].nunique()),
                "device": device.type, "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "smoke_test": args.smoke_test, "test_used": False, "class_weights": weights.detach().cpu().tolist() if weights is not None else None}
    save_json(run_dir / "metadata.json", metadata)
    history, best_macro_f1, best_epoch, best_stage, epoch_number = [], -math.inf, 0, "", 0
    started = time.perf_counter()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.head_parameters(), lr=float(training_config["warmup_head_lr"]), weight_decay=float(training_config["weight_decay"]))

    def execute_epoch(stage: str) -> bool:
        nonlocal epoch_number, best_macro_f1, best_epoch, best_stage
        epoch_number += 1
        train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler, stage, float(training_config["gradient_clip_norm"]), accumulation_steps, max_batches)
        with torch.no_grad():
            validation_metrics, _, _, _ = run_epoch(model, validation_loader, criterion, device, None, None, None, 0.0, 1, max_batches)
        history.append(flat_history_row(epoch_number, stage, train_metrics, validation_metrics))
        improved = validation_metrics["macro_f1"] > best_macro_f1 + float(training_config["early_stopping_min_delta"])
        if improved:
            best_macro_f1, best_epoch, best_stage = validation_metrics["macro_f1"], epoch_number, stage
            save_checkpoint(run_dir / "best.pt", model, epoch_number, stage, validation_metrics, args.seed, args.loss_mode, model_name)
            save_json(run_dir / "validation_metrics.json", validation_metrics)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(f"epoch={epoch_number} stage={stage} train_f1={train_metrics['macro_f1']:.4f} val_f1={validation_metrics['macro_f1']:.4f}", flush=True)
        return improved

    for _ in range(warmup_epochs):
        execute_epoch("warmup")
    model.unfreeze_last()
    optimizer = torch.optim.AdamW([{"params": model.last_parameters(), "lr": float(training_config["finetune_backbone_lr"])}, {"params": model.head_parameters(), "lr": float(training_config["finetune_head_lr"])}], weight_decay=float(training_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, finetune_epochs))
    epochs_without_improvement = 0
    for _ in range(finetune_epochs):
        improved = execute_epoch("finetune")
        scheduler.step()
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if not args.smoke_test and epochs_without_improvement >= int(training_config["early_stopping_patience"]):
            break
    metadata.update({"best_epoch": best_epoch, "best_stage": best_stage, "best_validation_macro_f1": best_macro_f1,
                     "epochs_completed": epoch_number, "training_seconds": time.perf_counter() - started})
    save_json(run_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def test(args: argparse.Namespace, config: dict, run_dir: Path) -> None:
    seed_everything(args.seed)
    manifest_dir = Path(args.manifest_dir).resolve()
    train_frame = load_manifest(manifest_dir / "train.csv")
    validation_frame = load_manifest(manifest_dir / "validation.csv")
    test_frame = load_manifest(manifest_dir / "test.csv")
    validate_split_isolation({"train": train_frame, "validation": validation_frame, "test": test_frame})
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["seed"]) != args.seed or checkpoint["loss_mode"] != args.loss_mode:
        raise ValueError("checkpoint seed/loss mode does not match command arguments")
    model_name = args.model or checkpoint["model_name"]
    model = build_classifier(model_name, 5, int(config["data"]["input_size"]), False, float(config["training"]["dropout"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    training_config = config["training"]
    loader = make_loader(test_frame, manifest_dir, build_eval_transform(int(config["data"]["input_size"])), int(training_config["batch_size"]), int(training_config["num_workers"]), False, args.seed)
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        metrics, targets, predictions, image_paths = run_epoch(model, loader, criterion, device, None, None, None, 0.0)
    save_json(run_dir / "test_metrics.json", metrics)
    pd.DataFrame({"image_path": image_paths, "true_label": targets, "prediction": predictions}).to_csv(run_dir / "test_predictions.csv", index=False)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["test_used"] = True
    save_json(metadata_path, metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def train_direct(args: argparse.Namespace, config: dict, run_dir: Path) -> None:
    if not args.data_root:
        raise ValueError("--data-root is required for direct training")
    ensure_free_space(run_dir, float(config["storage"]["min_free_gb"]))
    training_config = config["training"]
    if args.seed not in training_config["seeds"] and not args.smoke_test:
        raise ValueError(f"seed must be one of {training_config['seeds']}")
    seed_everything(args.seed)
    manifest_dir = Path(args.manifest_dir).resolve()
    train_frame = load_manifest(manifest_dir / "train.csv")
    validation_frame = load_manifest(manifest_dir / "validation.csv")
    validate_split_isolation({"train": train_frame, "validation": validation_frame})
    train_loader, train_dataset = make_direct_loader(
        train_frame, args, config, True, args.seed
    )
    validation_loader, validation_dataset = make_direct_loader(
        validation_frame, args, config, False, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke_test:
        raise RuntimeError("direct high-resolution training requires CUDA")
    pretrained = bool(config["models"].get("pretrained", True)) and not args.no_pretrained
    model = build_classifier(
        "resnet18",
        5,
        input_dimensions(args.input_long_edge, config)[1],
        pretrained,
        float(training_config["dropout"]),
        input_channels_for_modality(args.modality),
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(training_config.get("label_smoothing", 0.0)),
        reduction="none",
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and not args.no_amp
        else None
    )
    accumulation_steps = int(
        config["direct_training"]["gradient_accumulation_steps"]
    )
    max_batches = 2 if args.smoke_test else None
    warmup_epochs = 1 if args.smoke_test else int(training_config["warmup_epochs"])
    finetune_epochs = 1 if args.smoke_test else int(training_config["finetune_epochs"])
    width, height = input_dimensions(args.input_long_edge, config)
    metadata = {
        "seed": args.seed,
        "model_name": "resnet18",
        "model_implementation": "torchvision_resnet18",
        "pretrained": pretrained,
        "num_classes": 5,
        "source_mode": "original_jpg_direct_read",
        "modality": args.modality,
        "required_modalities": parse_modalities(args.required_modalities),
        "view_mode": args.view_mode,
        "sampling_mode": args.sampling_mode,
        "input_long_edge": str(args.input_long_edge),
        "input_width": width,
        "input_height": height,
        "input_channels": input_channels_for_modality(args.modality),
        "minimum_epochs": int(training_config["minimum_epochs"]),
        "maximum_epochs": warmup_epochs + finetune_epochs,
        "local_crop_long_edge": int(config["data"]["local_crop_long_edge"]),
        "micro_batch_size": int(config["direct_training"]["micro_batch_size"]),
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": int(config["direct_training"]["micro_batch_size"])
        * accumulation_steps,
        "excluded_filenames": config["data"].get("excluded_filenames", []),
        "parameter_count": count_parameters(model),
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "train_patient_groups": int(train_dataset.frame["patient_group"].nunique()),
        "validation_patient_groups": int(
            validation_dataset.frame["patient_group"].nunique()
        ),
        "device": device.type,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "smoke_test": args.smoke_test,
        "test_used": False,
    }
    save_json(run_dir / "metadata.json", metadata)
    history: list[dict] = []
    best_macro_f1, best_epoch, best_stage, epoch_number = -math.inf, 0, "", 0
    started = time.perf_counter()
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.head_parameters(),
        lr=float(training_config["warmup_head_lr"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    def execute_epoch(stage: str) -> bool:
        nonlocal epoch_number, best_macro_f1, best_epoch, best_stage
        epoch_number += 1
        train_metrics, _, _, _, _ = run_epoch_direct(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            stage,
            float(training_config["gradient_clip_norm"]),
            accumulation_steps,
            max_batches,
        )
        with torch.no_grad():
            validation_metrics, _, _, _, _ = run_epoch_direct(
                model,
                validation_loader,
                criterion,
                device,
                None,
                None,
                None,
                0.0,
                1,
                max_batches,
            )
        history.append(
            flat_history_row(epoch_number, stage, train_metrics, validation_metrics)
        )
        improved = validation_metrics["macro_f1"] > best_macro_f1 + float(
            training_config["early_stopping_min_delta"]
        )
        if improved:
            best_macro_f1, best_epoch, best_stage = (
                validation_metrics["macro_f1"],
                epoch_number,
                stage,
            )
            save_checkpoint(
                run_dir / "best.pt",
                model,
                epoch_number,
                stage,
                validation_metrics,
                args.seed,
                args.sampling_mode,
                "resnet18",
            )
            save_json(run_dir / "validation_metrics.json", validation_metrics)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"epoch={epoch_number} stage={stage} train_f1={train_metrics['macro_f1']:.4f} "
            f"val_f1={validation_metrics['macro_f1']:.4f}",
            flush=True,
        )
        return improved

    for _ in range(warmup_epochs):
        execute_epoch("warmup")
    model.unfreeze_last()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.last_parameters(),
                "lr": float(training_config["finetune_backbone_lr"]),
            },
            {
                "params": model.head_parameters(),
                "lr": float(training_config["finetune_head_lr"]),
            },
        ],
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, finetune_epochs)
    )
    epochs_without_improvement = 0
    for _ in range(finetune_epochs):
        improved = execute_epoch("finetune")
        scheduler.step()
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if not args.smoke_test and should_stop_early(
            epoch_number,
            epochs_without_improvement,
            int(training_config["minimum_epochs"]),
            int(training_config["early_stopping_patience"]),
        ):
            break
    metadata.update(
        {
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "best_validation_macro_f1": best_macro_f1,
            "epochs_completed": epoch_number,
            "training_seconds": time.perf_counter() - started,
        }
    )
    save_json(run_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def evaluate_direct(args: argparse.Namespace, config: dict, run_dir: Path) -> None:
    if not args.data_root:
        raise ValueError("--data-root is required for direct evaluation")
    metadata_path = run_dir / "metadata.json"
    checkpoint_path = run_dir / "best.pt"
    if not metadata_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError("direct evaluation requires metadata.json and best.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eval_args = argparse.Namespace(**vars(args))
    for key in (
        "modality",
        "view_mode",
        "sampling_mode",
        "input_long_edge",
    ):
        setattr(eval_args, key, metadata[key])
    eval_args.required_modalities = ",".join(metadata["required_modalities"])
    seed_everything(int(metadata["seed"]))
    manifest_dir = Path(args.manifest_dir).resolve()
    frames = {
        name: load_manifest(manifest_dir / f"{name}.csv") for name in SPLIT_NAMES
    }
    validate_split_isolation(frames)
    loader, dataset = make_direct_loader(
        frames[args.split], eval_args, config, False, int(metadata["seed"])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_classifier(
        "resnet18",
        5,
        int(metadata["input_height"]),
        False,
        float(config["training"]["dropout"]),
        int(metadata.get("input_channels", input_channels_for_modality(metadata["modality"]))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        metrics, targets, predictions, capture_ids, probabilities = run_epoch_direct(
            model, loader, criterion, device, None, None, None, 0.0
        )
    save_json(run_dir / f"{args.split}_metrics.json", metrics)
    prediction_frame = pd.DataFrame(
        {
            "capture_id": capture_ids,
            "true_label": targets,
            "prediction": predictions,
        }
    )
    for label in LABELS:
        prediction_frame[f"probability_{label}"] = [row[label] for row in probabilities]
    prediction_frame.to_csv(run_dir / f"{args.split}_predictions.csv", index=False)
    metadata[f"{args.split}_evaluated"] = True
    if args.split == "test":
        metadata["test_used"] = True
    metadata[f"{args.split}_images"] = len(dataset)
    save_json(metadata_path, metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def memory_smoke(args: argparse.Namespace, config: dict, run_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("memory smoke test requires CUDA")
    ensure_free_space(run_dir, float(config["storage"]["min_free_gb"]))
    device = torch.device("cuda")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    threshold = float(config["direct_training"]["max_vram_fraction"])
    batch_size = int(config["direct_training"]["micro_batch_size"])
    results = []
    selected = None
    for candidate in config["data"]["resolution_candidates"]:
        width, height = input_dimensions(candidate, config)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            model = build_classifier(
                "resnet18",
                5,
                height,
                False,
                float(config["training"]["dropout"]),
                input_channels_for_modality(args.modality),
            ).to(device)
            model.unfreeze_last()
            model.set_training_stage("finetune")
            images = torch.zeros(
                (batch_size, input_channels_for_modality(args.modality), height, width),
                device=device,
            )
            labels = torch.zeros(batch_size, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", enabled=True):
                loss = nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved(device)
            fraction = peak / total_memory
            accepted = fraction <= threshold
            results.append(
                {
                    "candidate": str(candidate),
                    "width": width,
                    "height": height,
                    "success": True,
                    "peak_reserved_gb": peak / 1024**3,
                    "vram_fraction": fraction,
                    "accepted": accepted,
                    "seconds": time.perf_counter() - started,
                }
            )
            if selected is None and accepted:
                selected = str(candidate)
            del images, labels, loss, model
        except torch.cuda.OutOfMemoryError as exc:
            results.append(
                {
                    "candidate": str(candidate),
                    "width": width,
                    "height": height,
                    "success": False,
                    "accepted": False,
                    "error": type(exc).__name__,
                }
            )
        finally:
            torch.cuda.empty_cache()
    payload = {
        "gpu_name": torch.cuda.get_device_name(device),
        "total_vram_gb": total_memory / 1024**3,
        "maximum_fraction": threshold,
        "batch_size": batch_size,
        "modality": args.modality,
        "input_channels": input_channels_for_modality(args.modality),
        "selected_long_edge": selected,
        "results": results,
    }
    save_json(run_dir / "memory_smoke.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def fuse_predictions(args: argparse.Namespace, run_dir: Path) -> None:
    if not args.input_runs:
        raise ValueError("--input-runs is required for late fusion")
    input_runs = [Path(value).resolve() for value in args.input_runs.split(",")]
    prediction_frames = []
    input_metadata = []
    for index, input_run in enumerate(input_runs):
        prediction_path = input_run / f"{args.split}_predictions.csv"
        metadata_path = input_run / "metadata.json"
        if not prediction_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing predictions or metadata in {input_run}")
        frame = pd.read_csv(prediction_path).sort_values("capture_id").reset_index(drop=True)
        frame = frame.rename(
            columns={f"probability_{label}": f"probability_{label}_{index}" for label in LABELS}
        )
        prediction_frames.append(frame)
        input_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    combined = prediction_frames[0]
    for index, frame in enumerate(prediction_frames[1:], start=1):
        combined = combined.merge(
            frame.drop(columns="prediction"),
            on=["capture_id", "true_label"],
            how="inner",
            validate="one_to_one",
        )
    expected_counts = {len(frame) for frame in prediction_frames}
    if len(expected_counts) != 1 or len(combined) != next(iter(expected_counts)):
        raise ValueError("fusion inputs are not capture-aligned")
    probability_columns = []
    for label in LABELS:
        columns = [f"probability_{label}_{index}" for index in range(len(input_runs))]
        combined[f"probability_{label}"] = combined[columns].mean(axis=1)
        probability_columns.append(f"probability_{label}")
    combined["prediction"] = combined[probability_columns].to_numpy().argmax(axis=1)

    manifest_dir = Path(args.manifest_dir).resolve()
    manifest = load_manifest(manifest_dir / f"{args.split}.csv").copy()
    manifest["capture_id"] = manifest["source_id"].map(
        lambda value: int(str(value).strip()[1:])
    )
    patient_lookup = dict(zip(manifest["capture_id"], manifest["patient_group"]))
    patient_groups = [patient_lookup[int(value)] for value in combined["capture_id"]]
    metrics = classification_metrics(
        combined["true_label"].astype(int).tolist(),
        combined["prediction"].astype(int).tolist(),
        patient_groups,
    )
    save_json(run_dir / f"{args.split}_metrics.json", metrics)
    combined[["capture_id", "true_label", "prediction", *probability_columns]].to_csv(
        run_dir / f"{args.split}_predictions.csv", index=False
    )
    seeds = sorted({int(item["seed"]) for item in input_metadata})
    if len(seeds) != 1:
        raise ValueError("fusion inputs must use the same seed")
    metadata_path = run_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {
            "model_name": "late_fusion",
            "fusion_method": "equal_probability_average",
            "seed": seeds[0],
            "modalities": [item["modality"] for item in input_metadata],
            "view_mode": input_metadata[0]["view_mode"],
            "sampling_mode": input_metadata[0]["sampling_mode"],
            "input_long_edge": input_metadata[0]["input_long_edge"],
            "input_runs": [str(path) for path in input_runs],
            "test_used": False,
        }
    )
    metadata[f"{args.split}_evaluated"] = True
    if args.split == "test":
        metadata["test_used"] = True
    save_json(metadata_path, metadata)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = validate_run_dir(args.run_dir)
    if args.action == "fuse":
        fuse_predictions(args, run_dir)
        return
    if args.action == "memory-smoke":
        memory_smoke(args, config, run_dir)
        return
    if args.data_root:
        if args.action == "train":
            train_direct(args, config, run_dir)
        elif args.action in {"evaluate", "test"}:
            if args.action == "test":
                args.split = "test"
            evaluate_direct(args, config, run_dir)
        return
    if args.action == "train":
        train(args, config, run_dir)
    else:
        test(args, config, run_dir)


if __name__ == "__main__":
    main()
