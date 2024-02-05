import torch
import torch.nn as nn
import torch.nn.functional as F


class WeatherApdaptiveCNNBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, num_embeddings
    ):
        super(WeatherApdaptiveCNNBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_embeddings = num_embeddings

        self.kernal_embed = nn.Parameter(
            torch.randn(
                num_embeddings, in_channels, out_channels, kernel_size, kernel_size
            )
        )

        self.cls_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.cls_dense = nn.Linear(in_channels, num_embeddings)

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.in_channels, f"Expected {self.in_channels} channels"

        batch_size = x.shape[0]

        weights = self.cls_conv(x)  # (B, C, H, W)
        weights = F.adaptive_avg_pool2d(weights, 1).flatten(1)  # (B, C)
        weights = self.cls_dense(weights)  # (B, num_embeddings)
        weights = F.softmax(weights, dim=1)  # (B, num_embeddings)

        kernal = (self.kernal_embed * weights[:, :, None, None, None, None]).sum(
            dim=1
        )  # (B, out_channels, kernel_size, kernel_size)

        # conv2d with different kernal for each batch
        # https://discuss.pytorch.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2
        output = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            kernal.view(-1, self.in_channels, self.kernel_size, self.kernel_size),
            stride=self.stride,
            padding=self.padding,
            groups=batch_size,
        ).view(batch_size, self.out_channels, x.shape[2], x.shape[3])

        return output
