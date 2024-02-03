from typing import Any, Dict

import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
from utils.box_ops import box_xyxy_to_cxcywh


class Normalize(nn.Module):
    def __init__(self, mean: list, std: list) -> None:
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, image, target) -> Dict[str, Any]:
        if "boxes" in target:
            boxes = target["boxes"]

            h, w = boxes.canvas_size
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)[None, :]

            target["boxes"] = torch.clone(boxes).to(torch.float32)
            target["nboxes"] = box_xyxy_to_cxcywh(boxes / scale)

        return T.Compose([T.Normalize(self.mean, self.std)])(image, target)
