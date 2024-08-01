from itertools import combinations
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Sequence, Union

class ContrastivePretrainingModel(L.LightningModule):
    def __init__(
        self,
        encoders: Union[Sequence[nn.Module], Dict[Any, nn.Module]],
        projection_dims: Union[int, Sequence[int], Dict[Any, int]],
        embed_dim: int,
        shared_projections: bool = False,
        max_temp: float = 100.0
    ):
        super().__init__()
        if shared_projections:
            w = [nn.Linear(projection_dims, embed_dim, bias=False)]*len(encoders) # type: ignore
        else:
            w = [nn.Linear(d, embed_dim, bias=False) for d in projection_dims] # type: ignore
        if isinstance(encoders, dict):
            self.encoders = encoders
            self.w = nn.ModuleDict(dict(zip(self.encoders.keys(), w)))
        else:
            self.encoders = tuple(encoders)
            self.w = nn.ModuleList(w)
        self.embed_dim = embed_dim
        self.max_temp = max_temp
        self.shared_projections = shared_projections
        self.t = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        if isinstance(batch, dict):
            features = [
                self.w[key](self.encoders[key](x)) for key, x in batch.items()
            ]
        else:
            features = [
                w(encoder(x)) for encoder, w, x in zip(self.encoders, self.w, batch) # type: ignore
            ]
        embeddings = [F.normalize(f, p=2, dim=1) for f in features]
        if isinstance(batch, dict):
            return dict(zip(batch.keys(), embeddings))
        return embeddings

    def _step(self, stage: str, batch):
        embeddings = self(batch)
        comparisons = list(combinations(embeddings, 2))
        loss = 0.0
        for a, b in comparisons:
            logits = torch.tensordot(a, b.T, 1)
            labels = torch.arange(a.size(0)).to(logits.device)
            loss_a = F.cross_entropy(logits * torch.exp(self.t), labels)
            loss_b = F.cross_entropy(logits.T * torch.exp(self.t), labels)
            loss += (loss_a + loss_b) / 2
        loss /= len(comparisons)
        self.log(f"{stage}/loss", loss)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)
