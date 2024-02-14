# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models.resnet import ResNet50_Weights
from typing import Dict, List

from utils.misc import NestedTensor

from .position_encoding import build_position_encoding
from . import lla_resnet


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self, backbone: nn.Module, return_interm_layers: bool
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters():
        #     if (
        #         not train_backbone
        #         or "layer2" not in name
        #         and "layer3" not in name
        #         and "layer4" not in name
        #     ):
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4": "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = backbone

    def forward(self, tensor_list: NestedTensor, alpha: torch.Tensor = None):
        if alpha is not None:
            xs = self.body(tensor_list.tensors, alpha)
        else:
            xs = self.body(tensor_list.tensors)
            
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        return_interm_layers: bool,
        dilation: bool,
        use_lla: bool,
        num_embeddings: int,
    ):
        norm_layer = FrozenBatchNorm2d

        if use_lla:
            backbone = lla_resnet.resnet50(
                replace_stride_with_dilation=[False, False, dilation],
                weights=ResNet50_Weights.IMAGENET1K_V2,
                norm_layer=norm_layer,
                num_embeddings=num_embeddings,
            )
        else:
            backbone = torchvision.models.resnet50(
                replace_stride_with_dilation=[False, False, dilation],
                weights=ResNet50_Weights.IMAGENET1K_V2,
                norm_layer=norm_layer,
            )

        super().__init__(backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

        self.use_lla = use_lla

        if use_lla:
            self.cls_branch = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, num_embeddings),
            )

    def compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        low_x = F.interpolate(x, scale_factor=0.25, mode="bilinear", align_corners=False)
        alpha = self.cls_branch(low_x)
        alpha = F.softmax(alpha / 10, dim=1)

        return alpha

    def forward(self, tensor_list: NestedTensor):
        if self.use_lla:
            alpha = self.compute_alpha(tensor_list.tensors)
            out = super().forward(tensor_list, alpha)
        else:
            out = super().forward(tensor_list)

        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    return_interm_layers = args.num_feature_levels > 1
    backbone = Backbone(return_interm_layers, args.dilation, args.use_lla, args.num_embeddings)
    model = Joiner(backbone, position_embedding)
    return model