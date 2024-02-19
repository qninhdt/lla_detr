from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.utils import _log_api_usage_once
from torchvision.models._utils import _ovewrite_named_param
from torchvision.models.resnet import ResNet50_Weights
from .lla_conv2d import LLAConv2d


def conv3x3(
    layer: nn.Module,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    num_embeddings: int = 1,
) -> nn.Module:
    """3x3 convolution with padding"""
    if layer is LLAConv2d:
        return layer(
            in_planes,
            out_planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=dilation,
            num_embeddings=num_embeddings,
        )
    else:
        return layer(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )


def conv1x1(
    layer: nn.Module,
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    num_embeddings: int = 1,
) -> nn.Module:
    """1x1 convolution"""
    if layer is LLAConv2d:
        return layer(
            in_planes,
            out_planes,
            kernel_size=(1, 1),
            stride=stride,
            num_embeddings=num_embeddings,
        )
    else:
        return layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class _Conv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(x)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        layer: nn.Module,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_embeddings: int = 1,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(layer, inplanes, width, num_embeddings=num_embeddings)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            layer, width, width, stride, groups, dilation, num_embeddings=num_embeddings
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(
            layer, width, planes * self.expansion, num_embeddings=num_embeddings
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, input) -> Tensor:
        x, alpha = input
        identity = x

        out = self.conv1(x, alpha)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, alpha)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, alpha)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, alpha


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Bottleneck],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_embeddings: int = 1,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = LLAConv2d(
            3,
            self.inplanes,
            kernel_size=(7, 7),
            stride=2,
            padding=3,
            num_embeddings=num_embeddings,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 64, layers[0], num_embeddings=num_embeddings, use_lla=True
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            num_embeddings=num_embeddings,
            use_lla=True,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            num_embeddings=num_embeddings,
            use_lla=True,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            num_embeddings=num_embeddings,
            use_lla=False,
        )

        for m in self.modules():
            if isinstance(m, _Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        num_embeddings: int = 1,
        use_lla: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(_Conv2d, self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        cnn = LLAConv2d if use_lla else _Conv2d

        layers.append(
            block(
                cnn,
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                num_embeddings=num_embeddings,
            )
        )

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    cnn,
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    num_embeddings=num_embeddings,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, alpha: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x, alpha)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        print(torch.argmax(alpha, dim=1).tolist(), end=" ")

        x, _ = self.layer1((x, alpha))

        x, _ = self.layer2((x, alpha))
        x2 = x

        x, _ = self.layer3((x, alpha))
        x3 = x

        x, _ = self.layer4((x, alpha))
        x4 = x

        return {
            "0": x2,
            "1": x3,
            "2": x4,
        }

    def forward(self, x: Tensor, alpha: Tensor) -> Tensor:
        return self._forward_impl(x, alpha)


def _resnet(
    block: Type[Union[Bottleneck]],
    layers: List[int],
    weights: Any,
    progress: bool,
    num_embeddings: int,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, num_embeddings=num_embeddings, **kwargs)

    if weights is not None:
        state_dict = weights.get_state_dict(progress=progress, check_hash=True)

        targets = [
            "conv1.weight",
            "layer1.0.conv1.weight",
            "layer1.0.conv2.weight",
            "layer1.0.conv3.weight",
            "layer1.1.conv1.weight",
            "layer1.1.conv2.weight",
            "layer1.1.conv3.weight",
            "layer1.2.conv1.weight",
            "layer1.2.conv2.weight",
            "layer1.2.conv3.weight",
            "layer2.0.conv1.weight",
            "layer2.0.conv2.weight",
            "layer2.0.conv3.weight",
            "layer2.1.conv1.weight",
            "layer2.1.conv2.weight",
            "layer2.1.conv3.weight",
            "layer2.2.conv1.weight",
            "layer2.2.conv2.weight",
            "layer2.2.conv3.weight",
            "layer3.0.conv1.weight",
            "layer3.0.conv2.weight",
            "layer3.0.conv3.weight",
            "layer3.1.conv1.weight",
            "layer3.1.conv2.weight",
            "layer3.1.conv3.weight",
            "layer3.2.conv1.weight",
            "layer3.2.conv2.weight",
            "layer3.2.conv3.weight",
        ]

        state_dict_ = dict(state_dict.items())
        for k, v in state_dict.items():
            if k in targets:
                state_dict_[k.replace("weight", "kernel_embed")] = v.unsqueeze(
                    0
                ).repeat(num_embeddings, 1, 1, 1, 1)
                del state_dict_[k]

        model.load_state_dict(state_dict_, False)

    return model


def resnet50(weights, **kwargs: Any) -> ResNet:
    weights = ResNet50_Weights.verify(weights)
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, True, **kwargs)
