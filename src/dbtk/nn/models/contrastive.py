from functools import partial
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..._utils import export

@export
class ContrastivePretrainingModel(L.LightningModule):
    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        embed_dim_a: int,
        embed_dim_b: int,
        projection_dim: Optional[int] = None,
        shared_projections: Optional[bool] = None,
        max_temperature: float = 100.0
    ):
        """
        Create a contrastive pretraining model.

        Args:
            encoder_a (nn.Module): The first encoder.
            encoder_b (nn.Module): The second encoder.
            embed_dim_a (int): The dimensionality of the first encoder embeddings.
            embed_dim_b (int): The dimensionality of the second encoder embeddings.
            projection_dim (Optional[int], optional): The dimensionality to project embeddings to before comparing. Defaults to smallest of `embed_dim_a` and `embed_dim_b`.
            shared_projections (Optional[bool], optional): Use shared projection weights. Defaults to False.
            max_temperature (float, optional): The maximum temperature value. Defaults to 100.0.
        """
        super().__init__()

        if encoder_a is encoder_b:
            assert shared_projections is not False, "Shared projections are required for one encoder."
            shared_projections = True
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.embed_dim_a = embed_dim_a
        self.embed_dim_b = embed_dim_b
        self.shared_projections = shared_projections if shared_projections is not None else False
        self.projection_dim = projection_dim if projection_dim is not None else min(embed_dim_a, embed_dim_b)
        self.max_temperature = max_temperature
        # Parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.w_a = self.w_b = nn.Linear(self.embed_dim_a, self.projection_dim, bias=False)
        if not self.shared_projections:
            self.w_b = nn.Linear(self.embed_dim_b, self.projection_dim, bias=False)

        for stage in ["training", "validation", "test"]:
            setattr(self, f"{stage}_step", partial(self._step, stage=stage))

    def forward(self, batch):
        a, b = batch
        embeddings = torch.stack([
            F.normalize(self.w_a(self.encoder_a(a)), p=2, dim=-1),
            F.normalize(self.w_b(self.encoder_b(b)), p=2, dim=-1)
        ])
        return embeddings

    def _step(self, batch, stage: str):
        # Compute local embeddings
        embeddings = self(batch)
        # all gather and merge
        ndim = embeddings.ndim
        embeddings = self.all_gather(embeddings, sync_grads=True)
        if embeddings.ndim > ndim:
            embeddings = torch.cat(tuple(embeddings), -2)
        a, b = embeddings
        logits = torch.tensordot(a, b.transpose(-1, -2), a.ndim - 1) * torch.exp(self.temperature)
        labels = torch.arange(a.shape[0], device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.transpose(-1, -2), labels)
        loss = (loss_a + loss_b) / 2.0
        accuracy = torch.sum(
            (torch.argmax(logits, dim=-1) == labels) + (torch.argmax(logits, dim=-2) == labels)
        ) / labels.shape[-1] / 2.0
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.temperature.data = torch.clamp(self.temperature, 0.0, self.max_temperature)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
