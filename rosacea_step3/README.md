# Step 3 execution notes

## Predeclared combinations

Based on the Step 2 three-seed Macro-F1 ranking (`MR > MB > MUV > M > MP`), Step 3 predeclares:

1. `M + MR`
2. `M + MB`
3. `M + MB + MR`
4. `MB + MR`
5. `M + MB + MP + MR + MUV` (all channels)

The Step 2 `M` results are reused directly as the White baseline. Step 3 trains only the five multi-channel methods above, each with seeds 42, 3407, and 2026.

## Protocol-fixed settings

- Binary task: CEA 0,1,2 -> class 0; CEA 3,4 -> class 1
- Input resolution: 384 x 384
- Train/Test: the fixed sample-ID split from `protocol.md`
- Model: ResNet50, ImageNet pretrained
- Optimizer: AdamW, lr 1e-4, weight decay 1e-4
- Batch size: 16
- Epochs: 50
- No validation split, no scheduler, no early stopping
- Final evaluation uses `last.pth`

For each selected channel, its RGB tensor is retained. Selected channels are concatenated channel-wise. For more than one channel, ResNet50's first convolution is expanded to the required input-channel count without adding layers; its ImageNet weights are tiled by RGB group and divided by the group count. Geometric and photometric augmentation parameters are synchronized across channels in each sample.

## Remote layout

Upload this directory as:

```text
/root/autodl-tmp/rosacea_step3
```

Keep Step 2 available as:

```text
/root/autodl-tmp/rosacea_step2/cache_768
/root/autodl-tmp/rosacea_step2/CEA.csv
/root/autodl-tmp/rosacea_step2/results
```

Then run:

```bash
bash run_step3.sh
```

The script performs audit, smoke test, all 15 formal multi-channel runs, and final report generation. Outputs are saved under `/root/autodl-tmp/rosacea_step3/results`, including raw run JSON/CSV files, `last.pth`, Mean ± Std summaries, charts, confusion matrices, and `step3_report.html`.
