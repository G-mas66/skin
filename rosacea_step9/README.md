# Step 9 execution notes

This directory contains the five-class Soft-label multi-channel benchmark. It reuses the completed Step 8 M (White) Soft-label runs as the baseline and trains the repository's predeclared multi-channel combinations:

1. `M + MR`
2. `M + MB`
3. `M + MB + MR`
4. `MB + MR`
5. `M + MB + MP + MR + MUV`

Each combination uses Seeds 42, 3407, and 2026 at 384 x 384 with ResNet50, ImageNet pretrained weights, AdamW, batch size 16, and 50 epochs. Images are matched by sample ID and augmented synchronously across channels. Soft labels are generated dynamically with true-class weight 0.90 and adjacent-class weight 0.05 (boundary classes use 0.10 for the only adjacent class).

The corresponding external Step 6 Hard Label multi-channel results were not supplied. Therefore the Hard-versus-Soft comparison and final input-plus-label choice remain pending; this stage makes no Soft-over-Hard improvement claim.
