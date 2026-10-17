"""Loss weighting and batch dispatch for SAND objectives."""

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch

_BATCH_TYPE = Dict[str, Any]
_STAGE_TYPE = Optional[Literal["train", "val", "test"]]


class Output(torch.nn.Module):
    """Compute and log weighted losses for selected training stages."""

    def __init__(
        self,
        name: str,
        loss_weight: float = 1.0,
        loss: Optional[
            Union[
                Callable[[Union[Dict[str, Any], torch.Tensor]], torch.Tensor],
                Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            ]
        ] = None,
        metrics: Optional[Any] = None,
        log_on_step_stages: List[_STAGE_TYPE] = ["train"],
        log_on_epoch_stages: List[_STAGE_TYPE] = ["val", "test"],
        dataloader_idx: int = -1,
        prog_bar: bool = False,
        sync_metric: bool = False,
        add_dataloader_idx_to_name: bool = False,
    ) -> None:
        """Initialise the Output wrapper.

        Args:
            name (str): Name used as a prefix when logging losses.
            loss_weight (float): Scalar multiplier applied to the computed loss.
                Defaults to 1.0.
            loss (Callable, optional): Callable that computes the scalar loss given
                a batch dict or ``(input, target)`` tensors. Defaults to None.
            metrics (optional): Must be None; external metrics are unsupported.
            log_on_step_stages (List[_STAGE_TYPE]): Stages logged at each step.
                Defaults to ["train"].
            log_on_epoch_stages (List[_STAGE_TYPE]): Stages logged at epoch end.
                Defaults to ["val", "test"].
            dataloader_idx (int): If >= 0, only activate for batches from this
                dataloader index. Defaults to -1 (all dataloaders).
            prog_bar (bool): Show loss in the progress bar. Defaults to False.
            sync_metric (bool): Synchronise logged losses across devices. Defaults to False.
            add_dataloader_idx_to_name (bool): Append dataloader index to logged
                names. Defaults to False.
        """
        super().__init__()

        self.name = name
        self.loss = loss
        if metrics is not None:
            raise ValueError("Local SAND supports the native losses but no external metrics.")
        self.log_on_step_stages = log_on_step_stages
        self.log_on_epoch_stages = log_on_epoch_stages
        self.prog_bar = prog_bar
        self.dataloader_idx = dataloader_idx
        self.sync_metric = sync_metric
        self.add_dataloader_idx_to_name = add_dataloader_idx_to_name

        self.loss_weight = loss_weight

        self.stages = [
            stage
            for stage in ["train", "val", "test"]
            if stage in (log_on_step_stages + log_on_epoch_stages)
        ]

    def batch_to_pred_target(
        self, batch: _BATCH_TYPE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract prediction and target tensors from a batch.

        Subclasses should override this method when the loss accepts an
        ``(input, target)`` pair.

        Args:
            batch (_BATCH_TYPE): Batch dictionary produced by the dataloader.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prediction and target tensors.

        Raises:
            NotImplementedError: Always; must be implemented by subclasses.
        """
        raise NotImplementedError

    def calculate_weighted_loss(
        self,
        batch: Optional[_BATCH_TYPE] = None,
        input: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the raw and weighted loss.

        Accepts either a batch dict or explicit ``(input, target)`` tensors.

        Args:
            batch (_BATCH_TYPE, optional): Full batch dict passed to the loss
                callable. Mutually exclusive with ``input``/``target``.
            input (torch.Tensor, optional): Prediction tensor.
            target (torch.Tensor, optional): Ground-truth tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Raw loss and weighted loss
            (``loss_weight * loss``).
        """
        if batch is not None:
            loss = self.loss(batch)
        elif (input is not None) & (target is not None):
            loss = self.loss(input, target)
        weighted_loss = self.loss_weight * loss
        return loss, weighted_loss

    def maybe_forward_loss(
        self, batch: _BATCH_TYPE, stage: _STAGE_TYPE, log_func: Callable[..., Any]
    ) -> torch.Tensor:
        """Compute and log the loss for the current stage if a loss is configured.

        Args:
            batch (_BATCH_TYPE): Current batch.
            stage (_STAGE_TYPE): Current training stage.
            log_func (Callable): Loss logging callback.

        Returns:
            torch.Tensor: Weighted loss, or 0 if no loss is configured.
        """
        if self.loss is not None:
            try:
                pred, target = self.batch_to_pred_target(batch)
                input = {"input": pred, "target": target}
            except NotImplementedError:
                input = {"batch": batch}
            loss, weighted_loss = self.calculate_weighted_loss(**input)
            log_func(
                name=f"{stage}_{self.name}_loss",
                value=loss,
                on_step=stage in self.log_on_step_stages,
                on_epoch=stage in self.log_on_epoch_stages,
                prog_bar=self.prog_bar,
                batch_size=batch.get("batch_size", None),
                add_dataloader_idx=self.add_dataloader_idx_to_name,
                sync_dist=self.sync_metric and stage != "train",
            )
        else:
            weighted_loss = 0
        return weighted_loss

    def forward(
        self,
        batch: _BATCH_TYPE,
        stage: _STAGE_TYPE,
        log_func: Callable[..., Any],
        dataloader_idx: int,
    ) -> torch.Tensor:
        """Compute the weighted loss for the given stage and dataloader.

        Skips computation when the batch originates from an unregistered
        dataloader index or when the stage is not active.

        Args:
            batch (_BATCH_TYPE): Current batch.
            stage (_STAGE_TYPE): Current training stage.
            log_func (Callable): Loss logging callback.
            dataloader_idx (int): Index of the current dataloader.

        Returns:
            torch.Tensor: Weighted loss contribution (zero if skipped).
        """
        if (self.dataloader_idx != -1) and (self.dataloader_idx != dataloader_idx):
            weighted_loss = torch.tensor(0.0)
        elif stage in self.stages:
            weighted_loss = self.maybe_forward_loss(
                batch, stage=stage, log_func=log_func
            )
        else:
            weighted_loss = torch.tensor(0.0)
        return weighted_loss

