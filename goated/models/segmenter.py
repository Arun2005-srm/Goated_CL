"""BatchNorm CNN encoders + a DeepLabV3+-style decoder. Output: logits at stride 4 and the
pre-classifier pixel features (used by class-prototype replay)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(i, o, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TinyEncoder(nn.Module):
    """Small random-init CNN for offline tests and the synthetic smoke run."""

    def __init__(self, width=16):
        super().__init__()
        self.low = nn.Sequential(conv_bn(3, width, s=2), conv_bn(width, width, s=2))
        self.high = nn.Sequential(conv_bn(width, width * 2, s=2), conv_bn(width * 2, width * 4, s=2))

    def forward(self, x):
        low = self.low(x)
        return low, self.high(low)


class ResNetEncoder(nn.Module):
    def __init__(self, depth, pretrained, output_stride):
        super().__init__()
        import torchvision.models as tvm
        fn = {18: tvm.resnet18, 34: tvm.resnet34, 50: tvm.resnet50, 101: tvm.resnet101}[depth]
        kw = {}
        if depth >= 50:
            kw["replace_stride_with_dilation"] = {32: [False, False, False], 16: [False, False, True],
                                                  8: [False, True, True]}[output_stride]
        m = fn(weights="DEFAULT" if pretrained else None, **kw)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = m.layer1, m.layer2, m.layer3, m.layer4

    def forward(self, x):
        low = self.layer1(self.stem(x))
        return low, self.layer4(self.layer3(self.layer2(low)))


class MobileNetEncoder(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        import torchvision.models as tvm
        m = getattr(tvm, name)(weights="DEFAULT" if pretrained else None).features
        # split after the last block whose output is still at stride 4
        with torch.no_grad():
            x, split = torch.zeros(1, 3, 64, 64), 0
            for i, blk in enumerate(m):
                x = blk(x)
                if x.shape[-1] == 16:
                    split = i + 1
        self.low, self.high = m[:split], m[split:]

    def forward(self, x):
        low = self.low(x)
        return low, self.high(low)


def build_encoder(spec):
    name = spec["encoder"]
    if name == "tiny":
        return TinyEncoder(spec.get("tiny_width", 16))
    if name.startswith("resnet"):
        return ResNetEncoder(int(name[6:]), spec.get("pretrained", True), spec.get("output_stride", 16))
    if name in ("mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small"):
        return MobileNetEncoder(name, spec.get("pretrained", True))
    raise ValueError(f"Unknown encoder {name}")


class ASPP(nn.Module):
    def __init__(self, cin, cout, rates):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn(cin, cout, 1)] + [conv_bn(cin, cout, 3, d=r) for r in rates])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cin, cout, 1, bias=False), nn.ReLU(inplace=True))
        self.project = conv_bn(cout * (len(rates) + 2), cout, 1)

    def forward(self, x):
        p = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([b(x) for b in self.branches] + [p], 1))


class Segmenter(nn.Module):
    def __init__(self, spec, num_classes):
        super().__init__()
        self.encoder = build_encoder(spec)
        with torch.no_grad():
            was = self.encoder.training
            self.encoder.eval()
            low, high = self.encoder(torch.zeros(1, 3, 64, 64))
            self.encoder.train(was)
        w = spec.get("width", 256)
        self.aspp = ASPP(high.shape[1], w, spec.get("aspp_rates", [6, 12, 18]))
        self.low_proj = conv_bn(low.shape[1], spec.get("low_width", 48), 1)
        self.fuse = nn.Sequential(conv_bn(w + spec.get("low_width", 48), w), conv_bn(w, w))
        self.classifier = nn.Conv2d(w, num_classes, 1)
        self.feature_dim = w

    def forward(self, x, return_features=False):
        low, high = self.encoder(x)
        h = F.interpolate(self.aspp(high), size=low.shape[-2:], mode="bilinear", align_corners=False)
        feats = self.fuse(torch.cat([h, self.low_proj(low)], 1))
        logits = self.classifier(feats)
        return (logits, feats) if return_features else logits

    def classify_features(self, z):
        """Apply the 1x1 classifier to feature vectors [N, D] -> logits [N, C]."""
        return F.linear(z, self.classifier.weight.flatten(1), self.classifier.bias)


def upsample(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)
