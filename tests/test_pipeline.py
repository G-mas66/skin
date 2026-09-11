from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SKIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKIN_ROOT))

from dataset import (
    DirectCEADataset,
    ResizePad,
    direct_collate,
    load_manifest,
    validate_split_isolation,
)
from model import ResNet18Classifier, UNetPPEncoderClassifier
from preprocess import (
    SplitFractions,
    assign_split_units,
    capture_id_from_source_id,
    choose_split,
    discover_records,
    modality_file_parts,
    resize_long_edge,
    run_data_audit,
)
from train import class_weights, classification_metrics, should_stop_early


class PipelineTest(unittest.TestCase):
    def test_modality_and_capture_id_parsing(self) -> None:
        self.assertEqual(modality_file_parts("MUV0771.JPG"), ("MUV", 771))
        self.assertEqual(modality_file_parts("MR0001.jpg"), ("MR", 1))
        self.assertEqual(capture_id_from_source_id("MB0042"), 42)
        self.assertIsNone(modality_file_parts("CEA.csv"))

    def test_resize_preserves_aspect_ratio(self) -> None:
        image = Image.new("RGB", (3448, 4600), (120, 50, 30))
        cached = resize_long_edge(image, 512)
        self.assertEqual(cached.size, (384, 512))
        model_input = ResizePad(224)(cached)
        self.assertEqual(model_input.size, (224, 224))

    def test_discovery_uses_only_labelled_grouped_m_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (30, 40), "red").save(root / "M0001.JPG")
            Image.new("RGB", (30, 40), "blue").save(root / "MR0001.JPG")
            Image.new("RGB", (30, 40), "green").save(root / "M0002.JPG")
            labels = root / "CEA.csv"
            pd.DataFrame({"M_id": ["M0001", "M0002"], "CEA_class": [0, 4]}).to_csv(
                labels, index=False
            )
            frame, counters = discover_records(root, labels, {1: "group_001"})
            self.assertEqual(list(frame["source_id"]), ["M0001"])
            self.assertEqual(list(frame["label"]), [0])
            self.assertEqual(counters["white_light_files"], 2)
            self.assertEqual(counters["missing_group"], 1)

    def test_patient_and_duplicate_units_do_not_cross_splits(self) -> None:
        rows = []
        for group in range(10):
            for label in range(5):
                rows.append(
                    {
                        "source_id": f"M{group:02d}{label}",
                        "source_path": "unused",
                        "source_index": group * 5 + label,
                        "label": label,
                        "patient_group": f"group_{group:03d}",
                        "sha256": f"hash_{group}_{label}",
                    }
                )
        frame = assign_split_units(pd.DataFrame(rows))
        split, _ = choose_split(
            frame,
            seed=2026,
            candidates=100,
            fractions=SplitFractions(),
        )
        group_split_counts = split.groupby("patient_group")["split"].nunique()
        self.assertTrue((group_split_counts == 1).all())
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})

    def test_manifest_validation_detects_patient_leakage(self) -> None:
        left = pd.DataFrame(
            [{"image_path": "images/a.jpg", "label": 0, "patient_group": "p1", "split_unit": "u1"}]
        )
        right = pd.DataFrame(
            [{"image_path": "images/b.jpg", "label": 1, "patient_group": "p1", "split_unit": "u2"}]
        )
        with self.assertRaisesRegex(ValueError, "patient_group leakage"):
            validate_split_isolation({"train": left, "validation": right})

    def test_manifest_rejects_out_of_range_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.csv"
            pd.DataFrame(
                [{"image_path": "a.jpg", "label": 5, "patient_group": "p1", "split_unit": "u1"}]
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "invalid labels"):
                load_manifest(path)

    def test_audit_reuses_split_and_creates_no_image_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            manifests = root / "manifests"
            output = root / "audit"
            data_root.mkdir()
            manifests.mkdir()
            modalities = ("M", "MB", "MP", "MR", "MUV")
            for capture_id in range(1, 4):
                for modality_index, modality in enumerate(modalities):
                    color = (capture_id * 40, modality_index * 30, 10)
                    Image.new("RGB", (30, 40), color).save(
                        data_root / f"{modality}{capture_id:04d}.JPG"
                    )
            Image.new("RGB", (30, 40), (1, 2, 3)).save(data_root / "MR01.JPG")
            pd.DataFrame(
                {
                    "M_id": ["M0001", "M0002", "M0003"],
                    "MR_id": ["MR0001", "MR0002", "MR0003"],
                    "CEA_class": [0, 2, 4],
                }
            ).to_csv(data_root / "CEA.csv", index=False)
            for split_name, capture_id in zip(
                ("train", "validation", "test"), range(1, 4)
            ):
                pd.DataFrame(
                    [
                        {
                            "source_id": f"M{capture_id:04d}",
                            "image_path": f"images/M{capture_id:04d}.jpg",
                            "label": (capture_id - 1) * 2,
                            "patient_group": f"group_{capture_id:03d}",
                            "split_unit": f"unit_{capture_id:03d}",
                        }
                    ]
                ).to_csv(manifests / f"{split_name}_manifest.csv", index=False)
            before = sorted(path.name for path in data_root.iterdir())
            summary = run_data_audit(
                data_root,
                data_root / "CEA.csv",
                manifests,
                output,
                {
                    "audit": {"expected_labelled_captures": 3},
                    "data": {
                        "excluded_filenames": [
                            "M0001.JPG",
                            "MR0001.JPG",
                            "MR01.JPG",
                        ],
                    }
                },
            )
            after = sorted(path.name for path in data_root.iterdir())
            self.assertEqual(before, after)
            self.assertEqual(summary["eligible_captures"], 3)
            self.assertEqual(summary["excluded_images"], 3)
            self.assertTrue(summary["checks"]["all_modality_labels_aligned"])
            self.assertTrue(summary["checks"]["unique_modality_per_capture"])
            self.assertEqual(summary["complete_all_modality_captures"], 2)
            self.assertTrue(summary["passed"])
            self.assertFalse((data_root / "images").exists())
            self.assertTrue((output / "audit_summary.json").exists())

    def test_direct_dataset_uses_common_capture_and_aligned_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for capture_id in (1, 2):
                for modality in ("M", "MB", "MP", "MR", "MUV"):
                    Image.new("RGB", (30, 40), (capture_id * 30, 20, 10)).save(
                        root / f"{modality}{capture_id:04d}.JPG"
                    )
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "M0001",
                        "image_path": "unused/M0001.jpg",
                        "label": 1,
                        "patient_group": "p1",
                        "split_unit": "u1",
                    },
                    {
                        "source_id": "M0002",
                        "image_path": "unused/M0002.jpg",
                        "label": 4,
                        "patient_group": "p2",
                        "split_unit": "u2",
                    },
                ]
            )
            dataset = DirectCEADataset(
                frame,
                root,
                "MB",
                ["M", "MB", "MP", "MR", "MUV"],
                (24, 32),
                (24, 32),
                "global",
                False,
                {
                    "horizontal_flip_p": 0.0,
                    "rotation_degrees": 0.0,
                },
                ["M0001.JPG"],
            )
            self.assertEqual(len(dataset), 1)
            item = dataset[0]
            self.assertEqual(item["capture_id"], 2)
            self.assertEqual(item["target"], 4)
            self.assertEqual(tuple(item["views"][0].shape), (3, 32, 24))
            batch = direct_collate([item, item])
            self.assertEqual(tuple(batch["views"][0].shape), (2, 3, 32, 24))

    def test_direct_dataset_packs_five_modalities_into_fifteen_channels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, modality in enumerate(("M", "MB", "MP", "MR", "MUV")):
                Image.new("RGB", (30, 40), (20 + index * 20, 30, 40)).save(
                    root / f"{modality}0001.JPG"
                )
            frame = pd.DataFrame(
                [
                    {
                        "source_id": "M0001",
                        "image_path": "unused/M0001.jpg",
                        "label": 3,
                        "patient_group": "p1",
                        "split_unit": "u1",
                    }
                ]
            )
            dataset = DirectCEADataset(
                frame,
                root,
                "ALL",
                ["M", "MB", "MP", "MR", "MUV"],
                (24, 32),
                (24, 32),
                "global",
                False,
                {"horizontal_flip_p": 0.0, "rotation_degrees": 0.0},
            )
            item = dataset[0]
            self.assertEqual(item["target"], 3)
            self.assertEqual(tuple(item["views"][0].shape), (15, 32, 24))

    def test_metrics_and_class_weights(self) -> None:
        metrics = classification_metrics([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)
        frame = pd.DataFrame(
            {"label": [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]}
        )
        weights = class_weights(frame, device="cpu")
        self.assertGreater(float(weights[0]), float(weights[4]))

    def test_model_freeze_and_output(self) -> None:
        import torch

        model = ResNet18Classifier(pretrained=False)
        model.freeze_backbone()
        self.assertTrue(all(parameter.requires_grad for parameter in model.head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.layer4.parameters()))
        model.unfreeze_last()
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.layer4.parameters()))
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 5))

    def test_resnet18_accepts_fifteen_channel_early_fusion(self) -> None:
        import torch

        model = ResNet18Classifier(pretrained=False, input_channels=15)
        self.assertEqual(model.backbone.conv1.in_channels, 15)
        self.assertTrue(
            torch.allclose(
                model.backbone.conv1.weight[:, :3],
                model.backbone.conv1.weight[:, 3:6],
            )
        )
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 15, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 5))

    def test_early_stopping_respects_minimum_epoch(self) -> None:
        self.assertFalse(should_stop_early(49, 20, 50, 8))
        self.assertTrue(should_stop_early(50, 8, 50, 8))
        self.assertFalse(should_stop_early(50, 7, 50, 8))

    def test_unetpp_encoder_output(self) -> None:
        import torch

        model = UNetPPEncoderClassifier()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
