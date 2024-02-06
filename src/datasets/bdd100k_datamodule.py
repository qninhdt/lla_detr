from typing import Any, Dict, Optional, Tuple, List, Literal

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.dataset import ApplyTransform
from utils.misc import nested_tensor_from_tensor_list
from utils.transform import Normalize

from .bdd100k import BDD100KDataset


class BDD100KDataModule(LightningDataModule):
    def __init__(
        self,
        dir: str,
        version: Literal["10k", "100k"],
        limit: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        image_size: Tuple[int, int],
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.dir = dir
        self.limit = limit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size_per_device = batch_size
        self.version = version
        self.dataset: Optional[BDD100KDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.image_size = (image_size[0], image_size[1])

        bbox_params = A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.3
        )

        self.train_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Resize(self.image_size[0], self.image_size[1]),
                A.RandomResizedCrop(
                    self.image_size[0],
                    self.image_size[1],
                    scale=(0.075, 1),
                    ratio=(
                        self.image_size[1] / self.image_size[0],
                        self.image_size[1] / self.image_size[0],
                    ),
                    p=0.75,
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )

        self.transforms = A.Compose(
            [
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            bbox_params=bbox_params,
        )

        # normalize = T.Compose(
        #     [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        # )

        # scales = [480, 512, 544]

        # self.train_transforms = T.Compose(
        #     [
        #         T.RandomHorizontalFlip(),
        #         T.RandomSelect(
        #             T.RandomResize(scales, max_size=960),
        #             T.Compose(
        #                 [
        #                     T.RandomResize([400, 500, 600]),
        #                     T.RandomSizeCrop(384, 600),
        #                     T.RandomResize(scales, max_size=960),
        #                 ]
        #             ),
        #         ),
        #         normalize,
        #     ]
        # )

        # self.transforms = T.Compose(
        #     [
        #         T.RandomResize([800], max_size=960),
        #         normalize,
        #     ]
        # )

    @property
    def num_classes(self) -> int:
        return self.dataset.num_classes()

    def prepare_data(self) -> None:
        return

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size

        if not self.train_dataset and not self.val_dataset and not self.data_test:
            self.train_dataset = BDD100KDataset(
                self.dir, self.version, "train", self.image_size
            )
            self.val_dataset = BDD100KDataset(
                self.dir, self.version, "val", self.image_size
            )

            if self.limit < 1:
                train_size = int(self.limit * len(self.train_dataset))
                val_size = int(self.limit * len(self.val_dataset))
                self.train_dataset, _ = random_split(
                    self.train_dataset,
                    [train_size, len(self.train_dataset) - train_size],
                )
                self.val_dataset, _ = random_split(
                    self.val_dataset, [val_size, len(self.val_dataset) - val_size]
                )

            self.train_dataset = ApplyTransform(
                self.train_dataset, self.train_transforms
            )

            self.val_dataset = ApplyTransform(self.val_dataset, self.transforms)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.train_dataset, self.batch_size_per_device)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.val_dataset, self.batch_size_per_device)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.val_dataset, self.batch_size_per_device)

    def _create_dataloader(self, dataset: Dataset, batch_size: int) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def _collate_fn(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, List[dict]]:
        images = [x[0] for x in batch]
        images = nested_tensor_from_tensor_list(images)

        targets = [x[1] for x in batch]

        return images, targets
