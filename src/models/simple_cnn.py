import torch


class SimpleCNN(torch.nn.Module):

    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, 1, 1)
        self.global_pool = torch.nn.AdaptiveAvgPool2d(num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.global_pool(x)
        return x
