from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import MeanMetric

from .deformable_detr import DeformableDETR, SetCriterion, PostProcess
from utils.misc import reduce_dict, match_name_keywords

from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DeformableDETRModule(LightningModule):
    def __init__(
        self,
        net: Tuple[DeformableDETR, SetCriterion, PostProcess],
        optimizer: Dict[str, Any],
        compile: bool,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore="net")

        self.model, self.criterion, self.postprocessor = net
        self.backbone = self.model.backbone

        # iou_thresholds = [0.5 + (i * 0.05) for i in range(10)]
        iou_thresholds = [0.5]

        # metric objects for calculating mAP across batches
        # self.train_mAP = MeanAveragePrecision("xyxy", "bbox", iou_thresholds)
        self.val_mAP = MeanAveragePrecision("xyxy", "bbox", iou_thresholds)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_loss_ce = MeanMetric()
        self.train_loss_bbox = MeanMetric()
        self.train_loss_giou = MeanMetric()
        self.train_class_error = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, List[dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, targets = batch
        preds = self.forward(images)
        losses = self.criterion(preds, targets)

        weight_dict = self.criterion.weight_dict

        loss = sum(
            losses[k] * weight_dict[k] for k in losses.keys() if k in weight_dict
        )

        # reduced_losses = reduce_dict(losses)
        # reduced_scaled_losses = {
        #     k: v * weight_dict[k] for k, v in reduced_losses.items() if k in weight_dict
        # }

        # reduced_loss = sum(reduced_scaled_losses.values())

        preds = self.postprocess(preds, targets)

        return (preds, targets, loss, losses)

    def training_step(
        self, batch: Tuple[torch.Tensor, List[dict]], batch_idx: int
    ) -> torch.Tensor:
        (preds, targets, loss, losses) = self.model_step(batch)

        # update and log metrics
        self.train_loss.update(loss)
        self.train_loss_ce.update(losses["loss_ce"])
        self.train_loss_bbox.update(losses["loss_bbox"])
        self.train_loss_giou.update(losses["loss_giou"])
        self.train_class_error.update(losses["class_error"])

        self.log("train/rt_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        if self.global_step % 100 == 0:
            self.log(
                "train/class_error",
                self.train_class_error.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/loss_ce",
                self.train_loss_ce.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/loss_bbox",
                self.train_loss_bbox.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/loss_giou",
                self.train_loss_giou.compute(),
                prog_bar=True,
                sync_dist=True,
            )
            self.log(
                "train/loss", self.train_loss.compute(), prog_bar=True, sync_dist=True
            )

            self.train_loss.reset()
            self.train_loss_ce.reset()
            self.train_loss_bbox.reset()
            self.train_loss_giou.reset()
            self.train_class_error.reset()

        return loss

    def postprocess(self, preds: torch.Tensor, targets: dict) -> None:
        target_sizes = torch.tensor(
            [t["orig_size"] for t in targets], dtype=torch.float32, device=self.device
        )
        preds = self.postprocessor(preds, target_sizes)

        return preds

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        preds, targets, loss, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss.update(loss)
        self.val_mAP.update(preds, targets)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_mAP.compute()
        metrics = {k: v.to(self.device) for k, v in metrics.items()}

        self.log("val/loss", self.val_loss.compute(), prog_bar=True, sync_dist=True)
        self.log("val/mAP_50", metrics["map_50"], prog_bar=True, sync_dist=True)

        self.val_mAP.reset()
        self.val_loss.reset()

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not match_name_keywords(n, ["backbone.0"])
                    and not match_name_keywords(
                        n, ["reference_points", "sampling_offsets"]
                    )
                    and p.requires_grad
                ],
                "lr": self.hparams.optimizer.lr,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if match_name_keywords(n, ["backbone.0"]) and p.requires_grad
                ],
                "lr": self.hparams.optimizer.lr_backbone,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                    and p.requires_grad
                ],
                "lr": self.hparams.optimizer.lr
                * self.hparams.optimizer.lr_linear_proj_mult,
            },
        ]

        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, self.hparams.optimizer.lr_drop, self.hparams.optimizer.lr_gamma
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
