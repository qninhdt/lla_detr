from typing import List, Literal

import json
from pathlib import Path

import torch
import cv2
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
from tqdm import tqdm

from utils.dataset import Mapping
from utils.box_ops import box_xyxy_to_cxcywh

CATEGORIES = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
    "pedestrian",
    "motorcycle",
    "bicycle",
]

TIMEOFDAY = ["daytime", "dawn/dusk", "night"]


class BDD100KDataset(Dataset):
    def __init__(
        self,
        dir: str,
        version: Literal["10k", "100k"] = "100k",
        type: Literal[
            "train",
            "val",
        ] = "train",
    ) -> None:
        super().__init__()

        self.dir = Path(dir)
        self.type = type
        self.version = version
        self.categories = Mapping(CATEGORIES)
        self.timeofday = Mapping(TIMEOFDAY)

        self.load_labels()

    def load_labels(self):
        with open(self.dir / f"labels/bdd100k_labels_images_{self.type}.json") as f:
            data = json.load(f)

        self.labels = []

        for image in tqdm(data, desc=f"Loading {self.type} labels"):
            if image["attributes"]["timeofday"] not in TIMEOFDAY:
                image["attributes"]["timeofday"] = "daytime"

            sample = {
                "name": image["name"],
                "timeofday": self.timeofday[image["attributes"]["timeofday"]],
                "labels": [],
            }

            for label in image["labels"]:
                sample["labels"].append(
                    {
                        "category": self.categories[label["category"]],
                        "box2d": [
                            label["box2d"]["x1"],
                            label["box2d"]["y1"],
                            label["box2d"]["x2"],
                            label["box2d"]["y2"],
                        ],
                    }
                )

            if len(sample["labels"]) > 0:
                self.labels.append(sample)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]

        image = cv2.imread(str(self.dir / f"images/100k/{self.type}/{label['name']}"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        boxes = torch.tensor(
            [label["box2d"] for label in label["labels"]], dtype=torch.float32
        )
        boxes = BoundingBoxes(boxes, format="xyxy", canvas_size=(h, w))
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        categories = torch.tensor(
            [obj["category"] for obj in label["labels"]], dtype=torch.int64
        )
        timeofday = label["timeofday"]
        name = label["name"]

        target = {
            "name": name,
            "boxes": boxes,
            "labels": categories,
            "timeofday": timeofday,
            "area": area,
            "orig_size": (h, w),
        }

        return image, target

    def __len__(self) -> int:
        return len(self.labels)
