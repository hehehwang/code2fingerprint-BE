from typing import Tuple, List, Dict, Optional

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection, Metric

from code2seq.data.path_context import BatchedLabeledPathContext
from code2seq.data.vocabulary import Vocabulary
from code2seq.model.modules import PathEncoder
from code2seq.utils.optimization import configure_optimizers_alon


class Code2Seq(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        vocabulary: Vocabulary,
        teacher_forcing: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._optim_config = optimizer_config
        self._vocabulary = vocabulary

        if vocabulary.SOS not in vocabulary.label_to_id:
            raise ValueError(f"Can't find SOS token in label to id vocabulary")

        self.__pad_idx = vocabulary.label_to_id[vocabulary.PAD]
        eos_idx = vocabulary.label_to_id[vocabulary.EOS]
        ignore_idx = [vocabulary.label_to_id[vocabulary.SOS]]
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(pad_idx=self.__pad_idx, eos_idx=eos_idx, ignore_idx=ignore_idx)
            for holdout in ["train", "val", "test"]
        }
        self.__metrics = MetricCollection(metrics)

        self._encoder = self._get_encoder(model_config)
        decoder_step = LSTMDecoderStep(model_config, len(vocabulary.label_to_id), self.__pad_idx)
        self._decoder = Decoder(
            decoder_step, len(vocabulary.label_to_id), vocabulary.label_to_id[vocabulary.SOS], teacher_forcing
        )

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx, reduction="batch-mean")

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocabulary

    def _get_encoder(self, config: DictConfig) -> nn.Module:
        return PathEncoder(
            config,
            len(self._vocabulary.token_to_id),
            self._vocabulary.token_to_id[Vocabulary.PAD],
            len(self._vocabulary.node_to_id),
            self._vocabulary.node_to_id[Vocabulary.PAD],
        )

    # ========== Main PyTorch-Lightning hooks ==========

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        return configure_optimizers_alon(self._optim_config, self.parameters())

    def forward(  # type: ignore
        self,
        from_token: torch.Tensor,
        path_nodes: torch.Tensor,
        to_token: torch.Tensor,
        contexts_per_label: torch.Tensor,
        output_length: int,
        target_sequence: torch.Tensor = None,
    ) -> torch.Tensor:
        encoded_paths = self._encoder(from_token, path_nodes, to_token)
        output_logits = self._decoder(encoded_paths, contexts_per_label, output_length, target_sequence)
        return encoded_paths, output_logits