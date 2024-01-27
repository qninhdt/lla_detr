from typing import Any, Dict

import torchvision.transforms.v2 as T
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "boxes" in sample:
            boxes = sample["boxes"]

            w, h = boxes.canvas_size
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)

            sample["nboxes"] = torch.clone(boxes) / scale
            sample["boxes"] = torch.clone(boxes)

        return T.Compose([T.Normalize(self.mean, self.std)])(sample)
