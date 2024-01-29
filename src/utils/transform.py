from typing import Any, Dict

import torchvision.transforms.v2 as T
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image, target) -> Dict[str, Any]:
        if "boxes" in target:
            boxes = target["boxes"]

            h, w = boxes.canvas_size
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)

            target["nboxes"] = torch.clone(boxes) / scale
            target["boxes"] = torch.clone(boxes)

        return T.Compose([T.Normalize(self.mean, self.std)])(image, target)
