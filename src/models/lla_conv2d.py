from typing import Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class LLAConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=(1,1), stride=1, padding=0, num_embeddings=1
    ):
        super(LLAConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_embeddings = num_embeddings

        self.kernel_embed = nn.Parameter(
            torch.Tensor(
                num_embeddings,
                out_channels,
                in_channels,
                kernel_size[0],
                kernel_size[1],
            ),
            requires_grad=True,
        )

        # self.bias_embed = nn.Parameter(
        #     torch.Tensor(num_embeddings, out_channels, in_channels), requires_grad=True
        # )

        # self.bn = nn.BatchNorm2d(in_channels)

        self.initialize_parameters()

    def initialize_parameters(self):
        for i in range(self.num_embeddings):
            init.kaiming_uniform_(self.kernel_embed[i], a=math.sqrt(5))
        # bound = 1 / math.sqrt(self.kernel_embed[0, 0].numel())
        # nn.init.uniform_(self.bias_embed, -bound, bound)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels"
        assert (
            alpha.shape[1] == self.num_embeddings
        ), f"Expected {self.num_embeddings} embeddings"

        batch_size = x.shape[0]
        kernels = (self.kernel_embed[None] * alpha[:, :, None, None, None, None]).sum(1)
        # biases = (self.bias_embed[None] * alpha[:, :, None]).sum(1)
        output = []
        for i in range(batch_size):
            kernel = kernels[i]
            # bias = biases[i]

            # depthwise conv
            o = F.conv2d(
                x[i][None],
                kernel,
                None,
                stride=self.stride,
                padding=self.padding
            )

            output.append(o)

        output = torch.cat(output, dim=0)
        # output = self.bn(output)
        # output = F.relu(output)

        return output

    def __str__(self):
        return f"LLAConv2d(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, num_embeddings={self.num_embeddings})"
