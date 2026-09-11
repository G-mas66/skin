"""Create safe splits or audit the original multi-modal CEA dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps


WHITE_LIGHT_PATTERN = re.compile(r"^M(\d+)\.jpe?g$", re.IGNORECASE)
MODALITY_PATTERN = re.compile(r"^(MUV|MB|MP|MR|M)(\d+)\.jpe?g$", re.IGNORECASE)
SOURCE_ID_PATTERN = re.compile(r"^(?:MUV|MB|MP|MR|M)(\d+)$", re.IGNORECASE)
MODALITIES = ("M", "MB", "MP", "MR", "MUV")
SPLIT_NAMES = ("train", "validation", "test")


class UnionFind:
    def __init__(self, values) -> None:
        self.parent = {value: value for value in values}

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left, right) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0 for value in values) or not math.isclose(sum(values), 1.0):
            raise ValueError("split fractions must be positive and sum to 1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cache", "audit"), default="cache")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--label-csv", required=True)
    parser.add_argument("--group-ranges-csv")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--candidate-splits", type=int, default=10000)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def capture_id_from_source_id(value: object) -> int:
    match = SOURCE_ID_PATTERN.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f"invalid modality source id: {value}")
    return int(match.group(1))


def modality_file_parts(path: str | Path) -> tuple[str, int] | None:
    match = MODALITY_PATTERN.fullmatch(Path(path).name)
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


def _manifest_path(manifest_dir: Path, split_name: str) -> Path:
    for filename in (f"{split_name}.csv", f"{split_name}_manifest.csv"):
        candidate = manifest_dir / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no manifest found for split: {split_name}")


def load_existing_split_manifests(manifest_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(manifest_dir).resolve()
    frames: dict[str, pd.DataFrame] = {}
    required = {"source_id", "label", "patient_group", "split_unit"}
    for split_name in SPLIT_NAMES:
        frame = pd.read_csv(_manifest_path(root, split_name))
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{split_name} manifest missing columns: {sorted(missing)}"
            )
        frame = frame.copy()
        frame["capture_id"] = frame["source_id"].map(capture_id_from_source_id)
        frame["label"] = frame["label"].astype(int)
        frame["split"] = split_name
        if frame["capture_id"].duplicated().any():
            raise ValueError(f"duplicate capture_id in {split_name} manifest")
        frames[split_name] = frame
    return frames


def split_leakage_summary(frames: dict[str, pd.DataFrame]) -> dict:
    overlaps = {
        "patient_group": {},
        "split_unit": {},
        "capture_id": {},
        "exact_duplicate": {},
    }
    for index, left_name in enumerate(SPLIT_NAMES):
        left = frames[left_name]
        for right_name in SPLIT_NAMES[index + 1 :]:
            right = frames[right_name]
            key = f"{left_name}_vs_{right_name}"
            for column in ("patient_group", "split_unit", "capture_id"):
                overlaps[column][key] = len(
                    set(left[column].astype(str)) & set(right[column].astype(str))
                )
            if "sha256" in left.columns and "sha256" in right.columns:
                overlaps["exact_duplicate"][key] = len(
                    set(left["sha256"].dropna().astype(str))
                    & set(right["sha256"].dropna().astype(str))
                )
            else:
                overlaps["exact_duplicate"][key] = None
    numeric_values = [
        value
        for values in overlaps.values()
        for value in values.values()
        if value is not None
    ]
    return {"overlaps": overlaps, "passed": not any(numeric_values)}


def load_capture_labels(label_csv: str | Path) -> tuple[dict[int, int], dict]:
    labels = pd.read_csv(Path(label_csv))
    required = {"M_id", "CEA_class"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"label CSV missing columns: {sorted(missing)}")
    candidates: dict[int, set[int]] = defaultdict(set)
    mismatched_m_mr_rows = 0
    invalid_label_rows = 0
    for row in labels.itertuples(index=False):
        try:
            label = int(row.CEA_class)
        except (TypeError, ValueError):
            invalid_label_rows += 1
            continue
        if label not in range(5):
            invalid_label_rows += 1
            continue
        ids = []
        for column in ("M_id", "MR_id"):
            value = getattr(row, column, None)
            if value is None or pd.isna(value) or not str(value).strip():
                continue
            ids.append(capture_id_from_source_id(value))
        if len(set(ids)) > 1:
            mismatched_m_mr_rows += 1
        for capture_id in ids:
            candidates[capture_id].add(label)
    conflicts = sorted(
        capture_id for capture_id, values in candidates.items() if len(values) > 1
    )
    lookup = {
        capture_id: next(iter(values))
        for capture_id, values in candidates.items()
        if len(values) == 1
    }
    return lookup, {
        "csv_rows": int(len(labels)),
        "labelled_captures": int(len(lookup)),
        "conflicting_capture_labels": int(len(conflicts)),
        "mismatched_m_mr_rows": int(mismatched_m_mr_rows),
        "invalid_label_rows": int(invalid_label_rows),
    }


def discover_modality_inventory(data_root: str | Path) -> pd.DataFrame:
    root = Path(data_root).resolve()
    rows = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        parts = modality_file_parts(path)
        if parts is None:
            continue
        modality, capture_id = parts
        rows.append(
            {
                "modality": modality,
                "capture_id": capture_id,
                "file_size": int(path.stat().st_size),
                "source_path": str(path),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no M/MB/MP/MR/MUV JPG images were found")
    return frame


def exact_duplicate_summary(
    inventory: pd.DataFrame, split_lookup: dict[int, str]
) -> dict:
    hash_groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    repeated_sizes = inventory.groupby("file_size").filter(lambda rows: len(rows) > 1)
    for row in repeated_sizes.itertuples(index=False):
        digest = sha256_file(Path(row.source_path))
        hash_groups[digest].append((str(row.modality), int(row.capture_id)))
    duplicate_groups = [values for values in hash_groups.values() if len(values) > 1]
    cross_split_groups = 0
    cross_modality_groups = 0
    for values in duplicate_groups:
        splits = {split_lookup[capture] for _, capture in values if capture in split_lookup}
        modalities = {modality for modality, _ in values}
        cross_split_groups += int(len(splits) > 1)
        cross_modality_groups += int(len(modalities) > 1)
    return {
        "candidate_files_hashed": int(len(repeated_sizes)),
        "exact_duplicate_groups": int(len(duplicate_groups)),
        "exact_duplicate_files": int(sum(len(values) for values in duplicate_groups)),
        "cross_split_duplicate_groups": int(cross_split_groups),
        "cross_modality_duplicate_groups": int(cross_modality_groups),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run_data_audit(
    data_root: str | Path,
    label_csv: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
    config: dict,
) -> dict:
    output = Path(output_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    if output == script_dir or script_dir in output.parents:
        raise ValueError("output-dir must be outside the skin source directory")
    output.mkdir(parents=True, exist_ok=True)

    frames = load_existing_split_manifests(manifest_dir)
    manifest = pd.concat(frames.values(), ignore_index=True)
    split_leakage = split_leakage_summary(frames)
    if manifest["capture_id"].duplicated().any():
        raise ValueError("capture_id appears in more than one existing manifest")
    capture_meta = manifest[
        ["capture_id", "label", "patient_group", "split_unit", "split"]
    ].copy()
    split_lookup = dict(zip(capture_meta["capture_id"], capture_meta["split"]))

    label_lookup, label_checks = load_capture_labels(label_csv)
    raw_inventory = discover_modality_inventory(data_root)
    excluded_filenames = {
        str(name).strip().upper()
        for name in config.get("data", {}).get("excluded_filenames", [])
    }
    excluded_mask = raw_inventory["source_path"].map(
        lambda value: Path(value).name.upper() in excluded_filenames
    )
    excluded_images = int(excluded_mask.sum())
    inventory = raw_inventory.loc[~excluded_mask].copy()
    modality_capture_sizes = inventory.groupby(["modality", "capture_id"]).size()
    duplicate_modality_capture = modality_capture_sizes[modality_capture_sizes > 1]
    duplicate_modality_capture_keys = int(len(duplicate_modality_capture))
    duplicate_modality_capture_extra_files = int(
        (duplicate_modality_capture - 1).sum()
    )
    inventory["label"] = inventory["capture_id"].map(label_lookup)
    inventory = inventory.merge(
        capture_meta.drop(columns="label"), on="capture_id", how="left"
    )
    eligible = inventory.dropna(subset=["label", "patient_group", "split"]).copy()
    eligible["label"] = eligible["label"].astype(int)

    manifest_label_mismatches = int(
        sum(
            label_lookup.get(int(row.capture_id)) != int(row.label)
            for row in manifest.itertuples(index=False)
            if int(row.capture_id) in label_lookup
        )
    )

    modality_rows = []
    for modality in MODALITIES:
        raw = raw_inventory[raw_inventory["modality"] == modality]
        considered = inventory[inventory["modality"] == modality]
        usable = eligible[eligible["modality"] == modality]
        modality_rows.append(
            {
                "modality": modality,
                "raw_images": int(len(raw)),
                "excluded_images": int(len(raw) - len(considered)),
                "labelled_images": int(considered["label"].notna().sum()),
                "split_covered_images": int(considered["split"].notna().sum()),
                "eligible_images": int(len(usable)),
                "patient_groups": int(usable["patient_group"].nunique()),
            }
        )
    modality_summary = pd.DataFrame(modality_rows)

    capture_modality = (
        eligible.groupby("capture_id")["modality"]
        .agg(lambda values: "+".join(item for item in MODALITIES if item in set(values)))
        .rename("modality_set")
        .reset_index()
    )
    pairing_summary = (
        capture_modality.groupby("modality_set")
        .size()
        .rename("capture_count")
        .reset_index()
        .sort_values("capture_count", ascending=False)
    )
    complete_capture_count = int(
        (capture_modality["modality_set"] == "+".join(MODALITIES)).sum()
    )

    capture_rows = eligible[
        ["capture_id", "label", "patient_group", "split"]
    ].drop_duplicates("capture_id")
    class_rows = []
    for label in range(5):
        class_images = eligible[eligible["label"] == label]
        class_captures = capture_rows[capture_rows["label"] == label]
        row = {
            "label": label,
            "images": int(len(class_images)),
            "captures": int(len(class_captures)),
            "patient_groups": int(class_captures["patient_group"].nunique()),
        }
        for modality in MODALITIES:
            row[f"{modality}_images"] = int(
                (class_images["modality"] == modality).sum()
            )
        class_rows.append(row)
    class_summary = pd.DataFrame(class_rows)

    modality_counts = eligible.groupby("capture_id")["modality"].nunique()
    patient_group_summary = (
        capture_rows.groupby(["patient_group", "split"])
        .agg(
            capture_count=("capture_id", "nunique"),
            class_count=("label", "nunique"),
            min_label=("label", "min"),
            max_label=("label", "max"),
        )
        .reset_index()
    )
    image_counts = (
        eligible.groupby("patient_group").size().rename("image_count").reset_index()
    )
    patient_group_summary = patient_group_summary.merge(
        image_counts, on="patient_group", how="left"
    )

    duplicate_checks = exact_duplicate_summary(inventory, split_lookup)
    aligned_capture_labels = bool(
        (eligible.groupby("capture_id")["label"].nunique() <= 1).all()
    )
    expected = int(config.get("audit", {}).get("expected_labelled_captures", 997))
    checks = {
        "expected_labelled_capture_count": label_checks["labelled_captures"] == expected,
        "no_label_conflicts": label_checks["conflicting_capture_labels"] == 0,
        "m_and_mr_ids_match": label_checks["mismatched_m_mr_rows"] == 0,
        "manifest_labels_match_csv": manifest_label_mismatches == 0,
        "all_modality_labels_aligned": aligned_capture_labels,
        "patient_split_isolation": bool(split_leakage["passed"]),
        "no_cross_split_exact_duplicates": duplicate_checks[
            "cross_split_duplicate_groups"
        ]
        == 0,
        "unique_modality_per_capture": duplicate_modality_capture_keys == 0,
    }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_policy": "original JPG read-only; no image cache created",
        "expected_labelled_captures": expected,
        "label_checks": label_checks,
        "manifest_captures": int(len(manifest)),
        "raw_images": int(len(raw_inventory)),
        "audited_images_after_exclusion": int(len(inventory)),
        "excluded_images": excluded_images,
        "excluded_filenames": sorted(excluded_filenames),
        "eligible_images": int(len(eligible)),
        "eligible_captures": int(eligible["capture_id"].nunique()),
        "patient_groups": int(capture_rows["patient_group"].nunique()),
        "complete_all_modality_captures": complete_capture_count,
        "manifest_label_mismatches": manifest_label_mismatches,
        "split_leakage": split_leakage,
        "duplicate_checks": duplicate_checks,
        "modality_capture_conflicts": {
            "duplicate_keys": duplicate_modality_capture_keys,
            "extra_files": duplicate_modality_capture_extra_files,
        },
        "patient_group_distribution": {
            "min_images": int(patient_group_summary["image_count"].min()),
            "median_images": float(patient_group_summary["image_count"].median()),
            "max_images": int(patient_group_summary["image_count"].max()),
            "min_captures": int(patient_group_summary["capture_count"].min()),
            "median_captures": float(patient_group_summary["capture_count"].median()),
            "max_captures": int(patient_group_summary["capture_count"].max()),
            "multi_class_groups": int((patient_group_summary["class_count"] > 1).sum()),
        },
        "checks": checks,
        "passed": bool(all(checks.values())),
    }

    class_summary.to_csv(output / "class_summary.csv", index=False)
    class_summary.assign(table_type="class_distribution").to_csv(
        output / "benchmark_summary.csv", index=False
    )
    modality_summary.to_csv(output / "modality_summary.csv", index=False)
    modality_summary.assign(table_type="modality_coverage").to_csv(
        output / "run_comparison.csv", index=False
    )
    pairing_summary.to_csv(output / "pairing_summary.csv", index=False)
    patient_group_summary.to_csv(output / "patient_group_summary.csv", index=False)
    (output / "audit_summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    snapshot = {
        "audit": {
            "capture_id_definition": "numeric suffix of M/MB/MP/MR/MUV filename",
            "expected_labelled_captures": expected,
            "excluded_filenames": sorted(excluded_filenames),
            "original_images": "read_only",
            "image_cache_created": False,
            "split_policy": "reuse_existing_patient_level_split",
            "modalities": list(MODALITIES),
        }
    }
    (output / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return summary


def load_group_lookup(path: str | Path) -> dict[int, str]:
    ranges = pd.read_csv(Path(path))
    required = {"group_id", "start", "end"}
    missing = required.difference(ranges.columns)
    if missing:
        raise ValueError(f"group ranges missing columns: {sorted(missing)}")
    lookup: dict[int, str] = {}
    for row in ranges.itertuples(index=False):
        start, end = int(row.start), int(row.end)
        if start > end:
            raise ValueError(f"invalid group range: {start}-{end}")
        group = f"group_{int(row.group_id):03d}"
        for source_index in range(start, end + 1):
            if source_index in lookup:
                raise ValueError(f"overlapping group range at index {source_index}")
            lookup[source_index] = group
    return lookup


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_records(
    data_root: str | Path,
    label_csv: str | Path,
    group_lookup: dict[int, str],
) -> tuple[pd.DataFrame, dict]:
    data_root = Path(data_root).resolve()
    labels = pd.read_csv(Path(label_csv))
    required = {"M_id", "CEA_class"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"label CSV missing columns: {sorted(missing)}")
    labels = labels[["M_id", "CEA_class"]].copy()
    labels["M_id"] = labels["M_id"].astype(str).str.strip().str.upper()
    labels = labels.dropna(subset=["CEA_class"])
    labels["CEA_class"] = labels["CEA_class"].astype(int)
    invalid = sorted(set(labels["CEA_class"]) - set(range(5)))
    if invalid:
        raise ValueError(f"CEA_class must be 0-4, found: {invalid}")
    if labels["M_id"].duplicated().any():
        raise ValueError("label CSV contains duplicate M_id values")
    label_lookup = dict(zip(labels["M_id"], labels["CEA_class"]))

    records = []
    counters = {"white_light_files": 0, "missing_label": 0, "missing_group": 0}
    for path in sorted(data_root.iterdir(), key=lambda item: item.name.lower()):
        match = WHITE_LIGHT_PATTERN.fullmatch(path.name)
        if not path.is_file() or match is None:
            continue
        counters["white_light_files"] += 1
        source_index = int(match.group(1))
        source_id = path.stem.upper()
        if source_id not in label_lookup:
            counters["missing_label"] += 1
            continue
        if source_index not in group_lookup:
            counters["missing_group"] += 1
            continue
        records.append(
            {
                "source_path": str(path),
                "source_id": source_id,
                "source_index": source_index,
                "label": int(label_lookup[source_id]),
                "patient_group": group_lookup[source_index],
                "sha256": sha256_file(path),
            }
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no labelled and grouped M white-light images were found")
    return frame, counters


def assign_split_units(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = sorted(result["patient_group"].unique())
    union_find = UnionFind(groups)
    for _, duplicate_rows in result.groupby("sha256"):
        duplicate_groups = sorted(duplicate_rows["patient_group"].unique())
        for group in duplicate_groups[1:]:
            union_find.union(duplicate_groups[0], group)
    root_names = {root: f"unit_{index:03d}" for index, root in enumerate(
        sorted({union_find.find(group) for group in groups}), start=1
    )}
    result["split_unit"] = result["patient_group"].map(
        lambda group: root_names[union_find.find(group)]
    )
    return result


def _split_score(
    frame: pd.DataFrame,
    assignment: dict[str, str],
    fractions: SplitFractions,
) -> float:
    assigned = frame.copy()
    assigned["split"] = assigned["split_unit"].map(assignment)
    total = len(assigned)
    global_counts = assigned["label"].value_counts().reindex(range(5), fill_value=0)
    global_distribution = global_counts / total
    targets = {
        "train": fractions.train,
        "validation": fractions.validation,
        "test": fractions.test,
    }
    score = 0.0
    for split_name, target_fraction in targets.items():
        selected = assigned[assigned["split"] == split_name]
        if selected.empty:
            return float("inf")
        score += abs(len(selected) / total - target_fraction) * 3.0
        distribution = (
            selected["label"].value_counts().reindex(range(5), fill_value=0)
            / len(selected)
        )
        score += float(np.abs(distribution - global_distribution).sum())
        actual_counts = selected["label"].value_counts().reindex(range(5), fill_value=0)
        expected_counts = global_counts * target_fraction
        score += 3.0 * float(
            np.mean(np.abs(actual_counts - expected_counts) / np.maximum(expected_counts, 1))
        )
        score += 10.0 * int((distribution == 0).any())
    return score


def choose_split(
    frame: pd.DataFrame,
    seed: int,
    candidates: int,
    fractions: SplitFractions = SplitFractions(),
) -> tuple[pd.DataFrame, float]:
    fractions.validate()
    units = sorted(frame["split_unit"].unique())
    if len(units) < 7:
        raise ValueError("at least seven split units are required for 70/15/15")
    validation_count = max(1, round(len(units) * fractions.validation))
    test_count = max(1, round(len(units) * fractions.test))
    train_count = len(units) - validation_count - test_count
    if train_count <= 0:
        raise ValueError("not enough split units for the requested fractions")

    best_assignment = None
    best_score = float("inf")
    for candidate in range(candidates):
        rng = np.random.default_rng(seed + candidate)
        shuffled = list(rng.permutation(units))
        assignment = {
            unit: (
                "train"
                if index < train_count
                else "validation"
                if index < train_count + validation_count
                else "test"
            )
            for index, unit in enumerate(shuffled)
        }
        score = _split_score(frame, assignment, fractions)
        if score < best_score:
            best_score = score
            best_assignment = assignment
    if best_assignment is None:
        raise RuntimeError("failed to create a patient-group split")
    result = frame.copy()
    result["split"] = result["split_unit"].map(best_assignment)
    return result, best_score


def resize_long_edge(image: Image.Image, long_edge: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = long_edge / max(image.size)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS, reducing_gap=2.0)


def create_cache(
    frame: pd.DataFrame,
    output_dir: Path,
    long_edge: int,
    jpeg_quality: int,
) -> pd.DataFrame:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    result = frame.copy()
    relative_paths = []
    for row in result.itertuples(index=False):
        relative_path = Path("images") / f"{row.source_id}.jpg"
        destination = output_dir / relative_path
        if not destination.exists():
            with Image.open(row.source_path) as image:
                resized = resize_long_edge(image, long_edge)
                resized.save(
                    destination,
                    format="JPEG",
                    quality=jpeg_quality,
                    subsampling=0,
                    optimize=True,
                )
        relative_paths.append(relative_path.as_posix())
    result["image_path"] = relative_paths
    return result


def leakage_summary(frame: pd.DataFrame) -> dict:
    patient_overlaps = {}
    unit_overlaps = {}
    hash_overlaps = {}
    for index, left_name in enumerate(SPLIT_NAMES):
        left = frame[frame["split"] == left_name]
        for right_name in SPLIT_NAMES[index + 1 :]:
            right = frame[frame["split"] == right_name]
            key = f"{left_name}_vs_{right_name}"
            patient_overlaps[key] = len(
                set(left["patient_group"]) & set(right["patient_group"])
            )
            unit_overlaps[key] = len(
                set(left["split_unit"]) & set(right["split_unit"])
            )
            hash_overlaps[key] = len(set(left["sha256"]) & set(right["sha256"]))
    passed = not any((*patient_overlaps.values(), *unit_overlaps.values(), *hash_overlaps.values()))
    return {
        "patient_group_overlap": patient_overlaps,
        "split_unit_overlap": unit_overlaps,
        "exact_duplicate_overlap": hash_overlaps,
        "passed": passed,
    }


def split_counts(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name in SPLIT_NAMES:
        selected = frame[frame["split"] == split_name]
        row = {
            "split": split_name,
            "image_count": int(len(selected)),
            "patient_group_count": int(selected["patient_group"].nunique()),
        }
        for label in range(5):
            row[f"label_{label}_count"] = int((selected["label"] == label).sum())
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "audit":
        if not args.manifest_dir:
            raise ValueError("--manifest-dir is required in audit mode")
        summary = run_data_audit(
            args.data_root,
            args.label_csv,
            args.manifest_dir,
            args.output_dir,
            config,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if not args.group_ranges_csv:
        raise ValueError("--group-ranges-csv is required in cache mode")
    output_dir = Path(args.output_dir).resolve()
    script_dir = Path(__file__).resolve().parent
    if output_dir == script_dir or script_dir in output_dir.parents:
        raise ValueError("output-dir must be outside the skin source directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    group_lookup = load_group_lookup(args.group_ranges_csv)
    records, counters = discover_records(args.data_root, args.label_csv, group_lookup)
    records = assign_split_units(records)
    records, score = choose_split(
        records,
        int(config["project"]["split_seed"]),
        args.candidate_splits,
    )
    records = create_cache(
        records,
        output_dir,
        int(config["data"]["cache_long_edge"]),
        int(config["data"]["jpeg_quality"]),
    )

    manifest_columns = [
        "image_path",
        "source_id",
        "label",
        "patient_group",
        "split_unit",
        "sha256",
        "split",
    ]
    for split_name in SPLIT_NAMES:
        records.loc[records["split"] == split_name, manifest_columns].sort_values(
            "source_id"
        ).to_csv(output_dir / f"{split_name}.csv", index=False)

    leakage = leakage_summary(records)
    if not leakage["passed"]:
        raise RuntimeError("patient or exact-duplicate leakage detected")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "M white-light images with CEA_class 0-4",
        "source_counters": counters,
        "usable_images": int(len(records)),
        "patient_groups": int(records["patient_group"].nunique()),
        "split_units": int(records["split_unit"].nunique()),
        "split_seed": int(config["project"]["split_seed"]),
        "candidate_score": float(score),
        "cache_long_edge": int(config["data"]["cache_long_edge"]),
        "jpeg_quality": int(config["data"]["jpeg_quality"]),
        "split_counts": split_counts(records),
        "leakage_checks": leakage,
    }
    (output_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
