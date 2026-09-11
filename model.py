"""Classification backbones used by the CEA benchmark."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ClassifierBase(nn.Module):
    """Small common interface used by the staged training loop."""

    def _set_requires_grad(self, modules: Iterable[nn.Module], value: bool) -> None:
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = value

    def freeze_backbone(self) -> None:
        raise NotImplementedError

    def unfreeze_last(self) -> None:
        raise NotImplementedError

    def head_parameters(self):
        raise NotImplementedError

    def last_parameters(self):
        raise NotImplementedError

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"warmup", "finetune"}:
            raise ValueError(f"unknown training stage: {stage}")
        self.train()


class ResNet18Classifier(ClassifierBase):
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if input_channels not in {3, 15}:
            raise ValueError("ResNet18 input_channels must be 3 or 15")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        if input_channels == 15:
            original = self.backbone.conv1
            replacement = nn.Conv2d(
                input_channels,
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=False,
            )
            with torch.no_grad():
                replacement.weight.copy_(original.weight.repeat(1, 5, 1, 1) / 5.0)
            self.backbone.conv1 = replacement
        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))

    def forward(self, images):
        return self.head(self.backbone(images))

    def freeze_backbone(self) -> None:
        self._set_requires_grad([self.backbone], False)
        self._set_requires_grad([self.head], True)

    def unfreeze_last(self) -> None:
        self.freeze_backbone()
        self._set_requires_grad([self.backbone.layer4], True)

    def head_parameters(self):
        return self.head.parameters()

    def last_parameters(self):
        return self.backbone.layer4.parameters()

    def set_training_stage(self, stage: str) -> None:
        super().set_training_stage(stage)
        frozen = [self.backbone.conv1, self.backbone.bn1, self.backbone.layer1,
                  self.backbone.layer2, self.backbone.layer3]
        if stage == "warmup":
            frozen.append(self.backbone.layer4)
        for module in frozen:
            module.eval()
        self.head.train()


class ViTSmallClassifier(ClassifierBase):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, input_size: int = 512, dropout: float = 0.3) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("ViT-small requires timm on the training environment") from exc
        self.backbone = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained, num_classes=0, img_size=input_size
        )
        features = int(self.backbone.num_features)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))

    def forward(self, images):
        return self.head(self.backbone(images))

    def freeze_backbone(self) -> None:
        self._set_requires_grad([self.backbone], False)
        self._set_requires_grad([self.head], True)

    def unfreeze_last(self) -> None:
        self.freeze_backbone()
        self._set_requires_grad(list(self.backbone.blocks)[-2:], True)
        self._set_requires_grad([self.backbone.norm], True)

    def head_parameters(self):
        return self.head.parameters()

    def last_parameters(self):
        return (parameter for module in [*list(self.backbone.blocks)[-2:], self.backbone.norm] for parameter in module.parameters())

    def set_training_stage(self, stage: str) -> None:
        super().set_training_stage(stage)
        if stage == "warmup":
            self.backbone.eval()
        self.head.train()


class VisionMambaFallback(nn.Module):
    """Dependency-free fallback used when the official CUDA extension is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.stages = nn.Sequential(
            nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU()),
            nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU()),
            nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU()),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        features = self.stages(self.stem(images))
        return self.pool(features).flatten(1)


class MambaVisionClassifier(ClassifierBase):
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        self.implementation = "official_mambavision"
        try:
            from mambavision import create_model
            self.backbone = create_model("mamba_vision_T", pretrained=pretrained)
            self._replace_classifier(num_classes, dropout)
        except Exception:
            self.implementation = "dependency_free_vision_mamba_fallback"
            self.backbone = VisionMambaFallback()
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, num_classes))

    def _replace_classifier(self, num_classes: int, dropout: float) -> None:
        for name in ("head", "classifier", "fc"):
            module = getattr(self.backbone, name, None)
            if isinstance(module, nn.Linear):
                features = module.in_features
                setattr(self.backbone, name, nn.Identity())
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
                return
            if isinstance(module, nn.Sequential) and module and isinstance(module[-1], nn.Linear):
                features = module[-1].in_features
                setattr(self.backbone, name, nn.Identity())
                self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, num_classes))
                return
        raise RuntimeError("could not locate the MambaVision classifier head")

    def forward(self, images):
        return self.head(self.backbone(images))

    def freeze_backbone(self) -> None:
        self._set_requires_grad([self.backbone], False)
        self._set_requires_grad([self.head], True)

    def unfreeze_last(self) -> None:
        self.freeze_backbone()
        modules = [module for module in self.backbone.children() if any(True for _ in module.parameters())]
        self._set_requires_grad(modules[-1:], True)

    def head_parameters(self):
        return self.head.parameters()

    def last_parameters(self):
        modules = [module for module in self.backbone.children() if any(True for _ in module.parameters())]
        return (parameter for module in modules[-1:] for parameter in module.parameters())

    def set_training_stage(self, stage: str) -> None:
        super().set_training_stage(stage)
        if stage == "warmup":
            self.backbone.eval()
        self.head.train()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPPEncoderClassifier(ClassifierBase):
    """UNet++-style nested encoder used as an exploratory classifier."""

    def __init__(self, num_classes: int = 5, dropout: float = 0.3) -> None:
        super().__init__()
        widths = (32, 64, 128, 256)
        self.enc1, self.enc2 = ConvBlock(3, widths[0]), ConvBlock(widths[0], widths[1])
        self.enc3, self.enc4 = ConvBlock(widths[1], widths[2]), ConvBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)
        self.nested3 = ConvBlock(widths[1] + widths[2], widths[2])
        self.nested4 = ConvBlock(widths[2] + widths[3], widths[3])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(widths[3], num_classes))

    def forward(self, images):
        x1 = self.enc1(images)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x3_nested = self.nested3(torch.cat([x3, nn.functional.interpolate(x2, size=x3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        x4_nested = self.nested4(torch.cat([x4, nn.functional.interpolate(x3_nested, size=x4.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        return self.head(x4_nested)

    def freeze_backbone(self) -> None:
        self._set_requires_grad([self.enc1, self.enc2, self.enc3, self.enc4, self.nested3, self.nested4], False)
        self._set_requires_grad([self.head], True)

    def unfreeze_last(self) -> None:
        self.freeze_backbone()
        self._set_requires_grad([self.enc4, self.nested3, self.nested4], True)

    def head_parameters(self):
        return self.head.parameters()

    def last_parameters(self):
        return (parameter for module in [self.enc4, self.nested3, self.nested4] for parameter in module.parameters())

    def set_training_stage(self, stage: str) -> None:
        super().set_training_stage(stage)
        if stage == "warmup":
            for module in [self.enc1, self.enc2, self.enc3, self.enc4, self.nested3, self.nested4]:
                module.eval()
        self.head.train()


def build_classifier(
    name: str,
    num_classes: int,
    input_size: int,
    pretrained: bool,
    dropout: float,
    input_channels: int = 3,
) -> ClassifierBase:
    if name == "resnet18":
        return ResNet18Classifier(num_classes, pretrained, dropout, input_channels)
    if input_channels != 3:
        raise ValueError(f"{name} supports 3-channel input only")
    if name == "vit_small":
        return ViTSmallClassifier(num_classes, pretrained, input_size, dropout)
    if name == "mamba_vision_t":
        return MambaVisionClassifier(num_classes, pretrained, dropout)
    if name == "unetpp_encoder":
        return UNetPPEncoderClassifier(num_classes, dropout)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
