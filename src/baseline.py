import json
import os

import torch
from torchmetrics.detection import MeanAveragePrecision

from datasets.bdd100k import CATEGORIES, WEATHERS, TIMEOFDAY
from utils.dataset import Mapping

categoriy_mapping = Mapping(CATEGORIES)
weather_mapping = Mapping(WEATHERS)
timeofday_mapping = Mapping(TIMEOFDAY)

alter_categories = {"pedestrian": "person", "motorcycle": "motor", "bicycle": "bike"}
attr_mapping = {}


def load_targets():
    with open("../datasets/bdd100k/labels/bdd100k_labels_images_val.json") as f:
        data = json.load(f)

    targets = []

    for image in data:
        attr_mapping[image["name"]] = {
            "timeofday": timeofday_mapping[image["attributes"]["timeofday"]],
            "weather": weather_mapping[image["attributes"]["weather"]],
        }
        target = {
            "name": image["name"],
            "timeofday": attr_mapping[image["name"]]["timeofday"],
            "weather": attr_mapping[image["name"]]["weather"],
            "boxes": [],
            "labels": [],
        }

        for label in image["labels"]:
            target["labels"].append(categoriy_mapping[label["category"]])
            target["boxes"].append(
                [
                    label["box2d"]["x1"],
                    label["box2d"]["y1"],
                    label["box2d"]["x2"],
                    label["box2d"]["y2"],
                ]
            )

        target["boxes"] = torch.tensor(target["boxes"])
        target["labels"] = torch.tensor(target["labels"])
        targets.append(target)

    return targets


def load_predictions(path: str):
    with open(path) as f:
        data = json.load(f)

    preds = []
    for image in data["frames"]:
        pred = {
            "name": image["name"],
            "timeofday": attr_mapping[image["name"]]["timeofday"],
            "weather": attr_mapping[image["name"]]["weather"],
            "boxes": [],
            "scores": [],
            "labels": [],
        }

        for label in image["labels"]:
            if label["category"] not in CATEGORIES:
                label["category"] = alter_categories[label["category"]]

            pred["labels"].append(categoriy_mapping[label["category"]])
            pred["boxes"].append(
                [
                    label["box2d"]["x1"],
                    label["box2d"]["y1"],
                    label["box2d"]["x2"],
                    label["box2d"]["y2"],
                ]
            )
            pred["scores"].append(label["score"])

        pred["boxes"] = torch.tensor(pred["boxes"])
        pred["labels"] = torch.tensor(pred["labels"])
        pred["scores"] = torch.tensor(pred["scores"])
        preds.append(pred)

    return preds


def compute_mAP(preds, targets, weather=None, timeofday=None):
    if weather is not None:
        preds = [pred for pred in preds if pred["weather"] == weather]
        targets = [target for target in targets if target["weather"] == weather]

    if timeofday is not None:
        preds = [pred for pred in preds if pred["timeofday"] == timeofday]
        targets = [target for target in targets if target["timeofday"] == timeofday]

    mAP = MeanAveragePrecision("xyxy", "bbox", [0.5])
    mAP.update(preds, targets)

    return mAP.compute()


print("Loading targets", end="... ")
targets = load_targets()
print("Done")

results = {}

# open all the prediction files in the folder
for file in os.listdir("../baselines"):
    if file.endswith(".json"):
        preds = load_predictions(f"../baselines/{file}")

        result = {
            "map_50": {},
        }

        print(f"Calculating mAP for {file}")

        print("Overall")
        metric = compute_mAP(preds, targets)
        print(metric["map_50"])
        result["map_50"]["overall"] = metric["map_50"].item()

        for weather in WEATHERS:
            print(f"Weather: {weather}")
            metric = compute_mAP(preds, targets, weather=weather_mapping[weather])
            print(metric["map_50"])
            result["map_50"][weather] = metric["map_50"].item()

        for timeofday in TIMEOFDAY:
            print(f"Time of day: {timeofday}")
            metric = compute_mAP(preds, targets, timeofday=timeofday_mapping[timeofday])
            print(metric["map_50"])
            result["map_50"][timeofday] = metric["map_50"].item()

        results[file] = result

        print()

print("Saving results", end="... ")

with open("../baselines/results.json", "w") as f:
    json.dump(results, f)

print("Done")
