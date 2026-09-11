# Step 8 execution notes

This directory contains the protocol-compliant Step 8 soft-label single-channel benchmark. It uses the original CEA 0-4 five-class task at 384 x 384, ResNet50 with ImageNet pretrained weights, AdamW, batch size 16, 50 epochs, and Seeds 42, 3407, and 2026.

Soft labels are generated dynamically from original hard labels. Interior classes use 0.90 for the true class and 0.05 for each adjacent class; boundary classes use 0.90 for the true class and 0.10 for the only adjacent class.

The formal soft-label training and standalone summary are complete. The required Hard Label comparison is intentionally pending until the external Step 5 results are supplied; no Soft-versus-Hard improvement claim is made in this stage.
