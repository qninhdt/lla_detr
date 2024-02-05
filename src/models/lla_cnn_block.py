import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class LowLightApdaptiveCNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, num_embeddings
    ):
        super(LowLightApdaptiveCNNBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_embeddings = num_embeddings

        self.kernal_embed = nn.Parameter(
            torch.randn(1, num_embeddings, in_channels, kernel_size, kernel_size)
        )

        self.bias_embed = nn.Parameter(torch.randn(1, num_embeddings, in_channels))

        self.bn = nn.BatchNorm2d(in_channels)

        self.cls_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.cls_bn = nn.BatchNorm2d(in_channels)
        self.cls_dense = nn.Linear(in_channels, num_embeddings)

        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_embeddings):
            init.kaiming_uniform_(self.kernal_embed[:, i], a=math.sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernal_embed[:, i])
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_embed[:, i], -bound, bound)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels"

        batch_size = x.shape[0]

        weights = self.cls_conv(x)  # (B, C, H, W)
        weights = self.cls_bn(weights)  # (B, C, H, W)
        weights = F.adaptive_avg_pool2d(weights, 1).flatten(1)  # (B, C)
        weights = F.relu(weights)  # (B, C)
        weights = self.cls_dense(weights)  # (B, num_embeddings)
        weights = F.softmax(weights, dim=1)  # (B, num_embeddings)

        kernals = (self.kernal_embed * weights[:, :, None, None, None]).sum(1)
        biases = (self.bias_embed * weights[:, :, None]).sum(1)

        output = []
        for i in range(batch_size):
            kernal = kernals[i, :, None, :, :]
            bias = biases[i]

            # depthwise conv
            o = F.conv2d(
                x[i],
                kernal,
                bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.in_channels,
            )

            output.append(o)

        output = torch.stack(output, dim=0)
        output = self.bn(output) + x
        output = F.relu(output)

        return output
