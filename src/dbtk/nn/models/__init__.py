import lightning as L
import torch.nn as nn
from typing import Callable, Optional

class LightningModuleWithHyperparameters(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

class CollatedModule(LightningModuleWithHyperparameters):
    def __init__(
        self,
        base: nn.Module,
        collate_input_fn: Optional[Callable] = None,
        collate_output_fn: Optional[Callable] = None
    ):
        super().__init__()
        self.base = base
        self.collate_input_fn = collate_input_fn
        self.collate_output_fn = collate_output_fn

    def forward(self, *args, **kwargs):
        if self.collate_input_fn is not None:
            args, kwargs = self.collate_input_fn(*args, **kwargs)
        output = self.base(*args, **kwargs)
        if self.collate_output_fn is not None:
            output = self.collate_output_fn(output)
        return output
