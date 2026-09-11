# Step 4 execution notes

This directory performs explanation-only DeepLIFT and channel-ablation analysis on the representative Step 3 binary multi-channel model.

The prepared representative method is `M+MB+MR`, using all three Step 3 epoch-50 `last.pth` checkpoints. Formal execution is intentionally gated by `--selection-approval USER_CONFIRMED_TEST_RANKING_DEVIATION`; this marker records that the user explicitly approved selecting the method from the Step 3 Test Macro-F1 ranking.

DeepLIFT uses Captum, the model-predicted class as target, and normalized black RGB groups as baselines. Channel contribution is the within-sample share of summed absolute input attributions. Ablation replaces one channel RGB group at a time with normalized black pixels and evaluates the unchanged fixed Test split. No images, labels, split, model weights, or training hyperparameters are changed.
