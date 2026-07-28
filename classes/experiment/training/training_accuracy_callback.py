from __future__ import annotations

from typing import Any

import torch
from skorch.callbacks import Callback


class TrainingAccuracyCallback(Callback):
    """Record train/validation accuracy from logits computed per epoch."""

    def on_epoch_begin(
        self,
        net: Any,
        **kwargs: Any,
    ) -> None:
        self.counts_ = {
            "train": self._empty_counts(),
            "valid": self._empty_counts(),
        }

    def on_batch_end(
        self,
        net: Any,
        batch: Any = None,
        training: bool = False,
        y_pred: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        if batch is None or y_pred is None:
            return

        partition = "train" if training else "valid"
        target = batch[1]

        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        target = target.to(y_pred.device).reshape(-1)
        prediction = y_pred.argmax(dim=1).reshape(-1)

        counts = self.counts_[partition]
        counts["correct"] += int(
            (prediction == target).sum().item()
        )
        counts["samples"] += int(target.numel())
        counts["true_negative"] += int(
            ((target == 0) & (prediction == 0)).sum().item()
        )
        counts["false_positive"] += int(
            ((target == 0) & (prediction == 1)).sum().item()
        )
        counts["false_negative"] += int(
            ((target == 1) & (prediction == 0)).sum().item()
        )
        counts["true_positive"] += int(
            ((target == 1) & (prediction == 1)).sum().item()
        )

    def on_epoch_end(
        self,
        net: Any,
        **kwargs: Any,
    ) -> None:
        for partition, counts in self.counts_.items():
            if counts["samples"] == 0:
                continue

            net.history.record(
                f"{partition}_accuracy",
                counts["correct"] / counts["samples"],
            )

            negative_count = (
                counts["true_negative"]
                + counts["false_positive"]
            )
            positive_count = (
                counts["true_positive"]
                + counts["false_negative"]
            )

            if negative_count == 0 or positive_count == 0:
                continue

            specificity = (
                counts["true_negative"] / negative_count
            )
            sensitivity = (
                counts["true_positive"] / positive_count
            )
            net.history.record(
                f"{partition}_balanced_accuracy",
                (specificity + sensitivity) / 2.0,
            )

    @staticmethod
    def _empty_counts() -> dict[str, int]:
        return {
            "correct": 0,
            "samples": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "true_positive": 0,
        }
