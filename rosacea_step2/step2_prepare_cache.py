#!/usr/bin/env python3
"""Build protocol-matched long-edge-768 caches for Step 2 image channels."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


EXCLUDED_IDS = {69, 296, 769, 770}
DEFAULT_CHANNELS = ("MB", "MP", "MR", "MUV")


def source_id(value: str) -> int:
    match = re.fullmatch(r"M(\d+)", value.strip())
    if match is None:
        raise ValueError(f"invalid M source id: {value!r}")
    return int(match.group(1))


def convert_one(task: tuple[Path, Path, int]) -> tuple[str, bool, int]:
    source, destination, quality = task
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        scale = 768 / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        rgb.resize(size, Image.Resampling.LANCZOS).save(
            destination,
            quality=quality,
            subsampling=1,
            optimize=True,
        )
    return destination.name, destination.is_file(), destination.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    channels = tuple(value.strip().upper() for value in args.channels.split(",") if value.strip())
    if not channels or set(channels) - {"M", "MB", "MP", "MR", "MUV"}:
        raise ValueError("channels must be a non-empty subset of M,MB,MP,MR,MUV")

    with args.label_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 997:
        raise ValueError(f"expected 997 raw label rows, found {len(rows)}")
    sample_ids = [source_id(row["M_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate M_id in label CSV")
    excluded_in_raw = sorted(set(sample_ids) & EXCLUDED_IDS)
    sample_ids = sorted(sample_id for sample_id in sample_ids if sample_id not in EXCLUDED_IDS)
    if len(sample_ids) != 996:
        raise ValueError(f"expected 996 eligible IDs, found {len(sample_ids)}")

    tasks: list[tuple[Path, Path, int]] = []
    for channel in channels:
        channel_dir = args.output_root / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        for sample_id in sample_ids:
            source = args.data_root / f"{channel}{sample_id:04d}.JPG"
            destination = channel_dir / f"{channel}{sample_id:04d}.JPG"
            if not source.is_file():
                raise FileNotFoundError(source)
            if not destination.exists():
                tasks.append((source, destination, args.jpeg_quality))

    print(f"channels={channels} eligible_ids={len(sample_ids)} pending_images={len(tasks)}", flush=True)
    completed = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(convert_one, task) for task in tasks]
            for future in as_completed(futures):
                name, saved, size = future.result()
                if not saved or size <= 0:
                    raise RuntimeError(f"cache write failed: {name}")
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    print(f"cached {completed}/{len(tasks)} pending images", flush=True)

    for channel in channels:
        paths = list((args.output_root / channel).glob(f"{channel}*.JPG"))
        ids = [int(re.fullmatch(rf"{channel}(\d+)\.JPG", path.name).group(1)) for path in paths]
        duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
        unexpected = sorted((set(ids) - set(sample_ids)) & ({69, 296, 769, 770} | set(range(1001, 10000))))
        if len(paths) != 996 or set(ids) != set(sample_ids) or duplicates or unexpected:
            raise RuntimeError(
                f"cache validation failed for {channel}: files={len(paths)}, duplicates={duplicates}, unexpected={unexpected}"
            )
        print(f"{channel}: 996 files validated", flush=True)
    print(
        f"cache complete under {args.output_root}; raw_rows=997; "
        f"excluded_labelled_ids={excluded_in_raw}; global_exclusions={sorted(EXCLUDED_IDS)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
