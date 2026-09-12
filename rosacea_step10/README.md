# Step 10 execution notes

This directory performs explanation-only DeepLIFT and channel-ablation analysis on the representative Step 9 five-class Soft-label multi-channel model.

The representative method is `M+MB`, selected by the predeclared Step 9 ranking rule: highest three-seed Test Macro-F1 mean, with smaller Macro-F1 std as the tie-break. The current user request explicitly based Step 10 on the Step 8 and Step 9 results. Because this uses a Test ranking to choose the explanation target, the run records a protocol deviation and is gated by:

```text
USER_CONFIRMED_STEP9_TEST_RANKING_SELECTION
```

All three Step 9 epoch-50 `last.pth` checkpoints are analyzed. DeepLIFT uses Captum, the model-predicted class as target, and normalized black RGB groups as baselines. Channel contribution is the within-sample share of summed absolute input attributions. Ablation replaces one channel RGB group at a time with normalized black pixels and evaluates the unchanged fixed Test split. No images, labels, split, model weights, or training hyperparameters are changed.

The direct five-class Hard-label explanation comparison remains pending until external Step 7 results are supplied. Step 4 is a binary Hard-label analysis and is not treated as a direct comparator.
