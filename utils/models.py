"""
Model zoo for ASD experiments.
All models expose .features(x) and .forward(x) returning (logits, features).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from typing import Tuple


# ═══════════════════════════════════════════════════════════
#  CIFAR-specific models (32×32 input)
# ═══════════════════════════════════════════════════════════

class CIFARResNetBlock(nn.Module):
    """Basic block for CIFAR ResNets."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CIFARResNet(nn.Module):
    """ResNet for CIFAR (32×32). Returns (logits, penultimate_features)."""

    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        feat = out.view(out.size(0), -1)
        return feat

    def extract_features_all_layers(self, x) -> dict:
        """Return feature maps from every residual stage + penultimate vector."""
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.avgpool(f4)
        penultimate = pooled.view(pooled.size(0), -1)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4,
                "penultimate": penultimate}

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


class CIFARBottleneckBlock(nn.Module):
    """Bottleneck block for CIFAR ResNet-50 (expansion=4)."""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class WideResNetBlock(nn.Module):
    """Wide ResNet basic block."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """Wide ResNet for CIFAR. Returns (logits, features)."""

    def __init__(self, depth, widen_factor, num_classes=100):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nstages[0], 3, stride=1, padding=1, bias=False)
        self.in_planes = nstages[0]
        self.layer1 = self._make_layer(WideResNetBlock, nstages[1], n, stride=1)
        self.layer2 = self._make_layer(WideResNetBlock, nstages[2], n, stride=2)
        self.layer3 = self._make_layer(WideResNetBlock, nstages[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nstages[3], num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        out = self.conv1(x)
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        out = F.relu(self.bn1(f3))
        pooled = self.avgpool(out)
        penultimate = pooled.view(pooled.size(0), -1)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "penultimate": penultimate}

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


class CIFAR3StageResNet(nn.Module):
    """
    3-stage CIFAR ResNet with variable depth and channel multiplier.
    Used in CRD / DKD benchmarks as resnet8x4 and resnet32x4.

    Architecture: conv1 → stage1 → stage2 → stage3 → avgpool → fc
    Channels: [16*k, 32*k, 64*k] where k = widen_factor
    Feature dim = 64 * widen_factor
    """

    def __init__(self, depth: int, widen_factor: int = 4, num_classes: int = 100):
        super().__init__()
        assert (depth - 2) % 6 == 0, "Depth must be 6n+2 (e.g., 8, 14, 20, 32, 44, 56)"
        n = (depth - 2) // 6
        k = widen_factor
        channels = [16 * k, 32 * k, 64 * k]

        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(CIFARResNetBlock, channels[0], n, stride=1)
        self.layer2 = self._make_layer(CIFARResNetBlock, channels[1], n, stride=2)
        self.layer3 = self._make_layer(CIFARResNetBlock, channels[2], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], num_classes)
        self._feat_dim = channels[2]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        out = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        pooled = self.avgpool(f3)
        penultimate = pooled.view(pooled.size(0), -1)
        return {"layer1": f1, "layer2": f2, "layer3": f3, "penultimate": penultimate}

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


# VGG config: list of channel widths; 'M' = MaxPool2d
_VGG_CONFIGS = {
    "vgg8":  [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


class CIFARVGGNet(nn.Module):
    """
    VGG for CIFAR (32×32 input). Returns (logits, penultimate_features).
    Feature dim = 512 for all VGG variants.

    Variants available: vgg8, vgg11, vgg13, vgg16.
    """

    def __init__(self, variant: str = "vgg13", num_classes: int = 100):
        super().__init__()
        assert variant in _VGG_CONFIGS, f"Unknown VGG variant: {variant}. Choose from {list(_VGG_CONFIGS)}"
        self.features = self._make_layers(_VGG_CONFIGS[variant])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, cfg):
        layers, in_channels = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_channels = v
        return nn.Sequential(*layers)

    def extract_features(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        """Capture spatial feature maps after each MaxPool boundary."""
        stage_idx = 0
        out_maps = {}
        out = x
        for layer in self.features:
            out = layer(out)
            if isinstance(layer, nn.MaxPool2d):
                stage_idx += 1
                out_maps[f"stage{stage_idx}"] = out
        # final pooled vector as penultimate
        pooled = self.avgpool(out)
        out_maps["penultimate"] = pooled.view(pooled.size(0), -1)
        return out_maps

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


class ShuffleNetV2Wrapper(nn.Module):
    """ShuffleNetV2 wrapper returning (logits, features)."""

    def __init__(self, num_classes=100, width_mult=1.0):
        super().__init__()
        base = tvm.shufflenet_v2_x1_0(weights=None)
        # Keep named stages for extract_features_all_layers
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 1024
        self.fc = nn.Linear(feat_dim, num_classes)

    def _upsample_if_needed(self, x):
        if x.shape[-1] < 56:
            x = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        return x

    def extract_features(self, x):
        x = self._upsample_if_needed(x)
        out = self.maxpool(self.conv1(x))
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        x = self._upsample_if_needed(x)
        out = self.maxpool(self.conv1(x))
        f2 = self.stage2(out)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.conv5(f4)
        pooled = self.avgpool(f5)
        return {
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
            "penultimate": pooled.view(pooled.size(0), -1),
        }

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


class MobileNetV2Wrapper(nn.Module):
    """MobileNetV2 wrapper returning (logits, features)."""

    def __init__(self, num_classes=100):
        super().__init__()
        base = tvm.mobilenet_v2(weights=None)
        self.features_extractor = base.features  # 19 blocks (0..18)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = 1280
        self.fc = nn.Linear(feat_dim, num_classes)

    def _upsample_if_needed(self, x):
        if x.shape[-1] < 56:
            x = F.interpolate(x, size=64, mode="bilinear", align_corners=False)
        return x

    def extract_features(self, x):
        x = self._upsample_if_needed(x)
        out = self.features_extractor(x)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        x = self._upsample_if_needed(x)
        # Capture at quarter-points: after block 6, 13, 18 (last)
        out = x
        checkpoints = {6: "layer2", 13: "layer3"}
        out_maps = {}
        for i, block in enumerate(self.features_extractor):
            out = block(out)
            if i in checkpoints:
                out_maps[checkpoints[i]] = out
        out_maps["layer4"] = out  # block 18 (final features block)
        pooled = self.avgpool(out)
        out_maps["penultimate"] = pooled.view(pooled.size(0), -1)
        return out_maps

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


class ImageNetResNetWrapper(nn.Module):
    """Standard torchvision ResNet wrapper for ImageNet-100."""

    def __init__(self, arch="resnet50", num_classes=100):
        super().__init__()
        constructor = getattr(tvm, arch)
        base = constructor(weights=None)
        # Expose individual layers for extract_features_all_layers
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        feat_dim = base.fc.in_features
        self.fc = nn.Linear(feat_dim, num_classes)

    def extract_features(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x) -> dict:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        f1 = self.layer1(out)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pooled = self.avgpool(f4)
        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
            "penultimate": pooled.view(pooled.size(0), -1),
        }

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


# ═══════════════════════════════════════════════════════════
#  DeiT / ViT student (transformer) — timm backbone, 224×224
#  Returns (logits, CLS-token feature). feat_dim = embed_dim (192 for DeiT-Tiny).
# ═══════════════════════════════════════════════════════════

class DeiTWrapper(nn.Module):
    """timm DeiT/ViT student returning (logits, pre-logit CLS feature)."""

    def __init__(self, arch="deit_tiny_patch16_224", num_classes=100, pretrained=False):
        super().__init__()
        import timm
        # num_classes=0 -> backbone outputs the pooled CLS feature; we add our own head
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.backbone.num_features
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def extract_features(self, x):
        return self.backbone(x)                      # (B, embed_dim) pooled CLS

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        return self.fc(feat), feat


# ═══════════════════════════════════════════════════════════
#  ShuffleNet-V1 for CIFAR
#  Based on Zhang et al. (CVPR 2018). Groups=3, feat_dim=960.
#  Matches the CRD/mdistiller implementation.
# ═══════════════════════════════════════════════════════════

def _channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    B, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(B, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class ShuffleV1Unit(nn.Module):
    """ShuffleNet-V1 basic unit (with or without stride)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, groups: int):
        super().__init__()
        self.stride = stride
        self.groups = groups
        mid_channels = out_channels // 4
        if stride == 2:
            out_channels -= in_channels   # concatenation, not addition

        # 1×1 GConv → Channel Shuffle → 3×3 DWConv → 1×1 GConv
        self.gconv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=groups if in_channels > groups else 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.gconv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if stride == 2:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gconv1(x)
        out = _channel_shuffle(out, self.groups)
        out = self.dw_conv(out)
        out = self.gconv2(out)
        if self.stride == 2:
            out = F.relu(torch.cat([out, self.shortcut(x)], dim=1), inplace=True)
        else:
            out = F.relu(out + x, inplace=True)
        return out


class ShuffleNetV1CIFAR(nn.Module):
    """
    ShuffleNet-V1 for CIFAR-100 (32×32 input).
    groups=3, feature dim = 960.
    Matches CRD / mdistiller benchmark configuration.
    """
    # Stage output channels indexed by groups
    _STAGE_CHANNELS = {
        1: [144, 288, 576],
        2: [200, 400, 800],
        3: [240, 480, 960],
        4: [272, 544, 1088],
        8: [384, 768, 1536],
    }

    def __init__(self, num_classes: int = 100, groups: int = 3):
        super().__init__()
        self.groups = groups
        stage_channels = self._STAGE_CHANNELS[groups]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.stage2 = self._make_stage(24,              stage_channels[0], n_blocks=4, stride=2)
        self.stage3 = self._make_stage(stage_channels[0], stage_channels[1], n_blocks=8, stride=2)
        self.stage4 = self._make_stage(stage_channels[1], stage_channels[2], n_blocks=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_channels[2], num_classes)
        self._feat_dim = stage_channels[2]

    def _make_stage(self, in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ShuffleV1Unit(in_ch, out_ch, stride=stride, groups=self.groups)]
        for _ in range(n_blocks - 1):
            layers.append(ShuffleV1Unit(out_ch, out_ch, stride=1, groups=self.groups))
        return nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x: torch.Tensor) -> dict:
        out = self.conv1(x)
        f2 = self.stage2(out)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        pooled = self.avgpool(f4)
        return {
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
            "penultimate": pooled.view(pooled.size(0), -1),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        logits = self.fc(feat)
        return logits, feat


# ═══════════════════════════════════════════════════════════
#  Detection Backbone Wrapper
#  Exposes multi-scale feature maps (FPN-ready) for ASD.
#  Returns (None, features_dict) — detection head is separate.
# ═══════════════════════════════════════════════════════════

class DetectionBackboneWrapper(nn.Module):
    """
    Torchvision ResNet backbone for object detection.
    Exposes C2-C5 feature maps (FPN style) for ASD distillation.

    Unlike classification wrappers, forward() returns (None, penultimate_feat)
    so the existing ASD loss machinery works unchanged. The detection head
    (Faster R-CNN, DETR, etc.) attaches separately via torchvision FasterRCNN.

    Use extract_features_all_layers() to get C2–C5 maps for AT / L_TC / L_SS.
    """

    def __init__(self, arch: str = "resnet50"):
        super().__init__()
        constructor = getattr(tvm, arch)
        base = constructor(weights="IMAGENET1K_V1")   # pretrained for detection
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1   # C2: stride 4
        self.layer2  = base.layer2   # C3: stride 8
        self.layer3  = base.layer3   # C4: stride 16
        self.layer4  = base.layer4   # C5: stride 32
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._feat_dim = base.fc.in_features   # 2048 for resnet50

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def extract_features_all_layers(self, x: torch.Tensor) -> dict:
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(out)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        pooled = self.avgpool(c5)
        return {
            "layer1": c2, "layer2": c3,
            "layer3": c4, "layer4": c5,
            "penultimate": pooled.view(pooled.size(0), -1),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.extract_features(x)
        return None, feat   # no classifier — detection head attaches separately


# ═══════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════

def build_model(arch: str, num_classes: int = 100, dataset: str = "cifar100") -> nn.Module:
    """Build a model by architecture name."""
    # tinyimagenet and cifar use the same CIFAR-style models
    cifar_like = dataset.startswith("cifar") or dataset == "tinyimagenet"
    cifar = dataset.startswith("cifar")

    # ── CIFAR-style 4-stage ResNets (32×32 input) ──
    if arch == "resnet18" and cifar_like:
        return CIFARResNet(CIFARResNetBlock, [2, 2, 2, 2], num_classes)
    elif arch == "resnet34" and cifar_like:
        return CIFARResNet(CIFARResNetBlock, [3, 4, 6, 3], num_classes)
    elif arch == "resnet50" and cifar_like:
        # Bottleneck ResNet-50 for CIFAR (feat_dim=2048)
        return CIFARResNet(CIFARBottleneckBlock, [3, 4, 6, 3], num_classes)

    # ── CIFAR-style 3-stage ResNets with 4× channels (CRD/DKD benchmark) ──
    elif arch == "resnet32x4":
        return CIFAR3StageResNet(depth=32, widen_factor=4, num_classes=num_classes)
    elif arch == "resnet8x4":
        return CIFAR3StageResNet(depth=8, widen_factor=4, num_classes=num_classes)

    # ── Wide ResNets ──
    elif arch == "wrn_40_2":
        return WideResNet(40, 2, num_classes)
    elif arch == "wrn_40_1":
        return WideResNet(40, 1, num_classes)
    elif arch == "wrn_16_2":
        return WideResNet(16, 2, num_classes)
    elif arch == "wrn_16_4":
        return WideResNet(16, 4, num_classes)
    elif arch == "wrn_28_2":
        return WideResNet(28, 2, num_classes)
    elif arch == "wrn_40_4":
        return WideResNet(40, 4, num_classes)

    # ── VGG (CIFAR-100) ──
    elif arch in ("vgg8", "vgg11", "vgg13", "vgg16"):
        return CIFARVGGNet(arch, num_classes)

    # ── CIFAR standard 3-stage ResNets (width=1×, original He et al. 2016) ──
    elif arch == "resnet20":
        return CIFAR3StageResNet(depth=20, widen_factor=1, num_classes=num_classes)
    elif arch == "resnet56":
        return CIFAR3StageResNet(depth=56, widen_factor=1, num_classes=num_classes)

    # ── Mobile/Efficient models ──
    elif arch == "shufflenet_v1":
        return ShuffleNetV1CIFAR(num_classes=num_classes)
    elif arch == "shufflenet_v2":
        return ShuffleNetV2Wrapper(num_classes)
    elif arch == "mobilenet_v2":
        return MobileNetV2Wrapper(num_classes)

    # ── ImageNet-style models (224×224 input) ──
    elif arch in ("resnet18", "resnet34", "resnet50", "resnet101") and not cifar_like:
        return ImageNetResNetWrapper(arch, num_classes)

    # ── Transformer (ViT/DeiT) students (224×224) ──
    elif arch in ("deit_tiny", "deit_small", "vit_tiny"):
        _timm_name = {"deit_tiny": "deit_tiny_patch16_224",
                      "deit_small": "deit_small_patch16_224",
                      "vit_tiny": "vit_tiny_patch16_224"}[arch]
        return DeiTWrapper(_timm_name, num_classes)

    # ── Detection backbone wrappers (ImageNet pretrain) ──
    elif arch in ("resnet50_det", "resnet101_det"):
        base_arch = arch.replace("_det", "")
        return DetectionBackboneWrapper(base_arch)

    else:
        raise ValueError(f"Unknown architecture: {arch} for dataset: {dataset}")


def get_feature_dim(model: nn.Module, dataset: str = "cifar100") -> int:
    """Get feature dimensionality of a model."""
    device = next(model.parameters()).device
    size = 32 if dataset.startswith("cifar") else (64 if dataset == "tinyimagenet" else 224)
    dummy = torch.randn(2, 3, size, size).to(device)
    with torch.no_grad():
        _, feat = model(dummy)
    return feat.shape[1]
