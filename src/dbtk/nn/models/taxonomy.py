from dnadb import taxonomy
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopDownTaxonomyClassificationModel(L.LightningModule):
    """
    The top-down taxonomy architecture proposed by SetBERT.

    encoder: The model used to embed DNA sequences.
    embed_dim: The dimension of the embedded DNA sequences.
    taxonomy_tree: The taxonomy tree used to guide the prediction.
    """
    def __init__(self, encoder: nn.Module, embed_dim: int, taxonomy_tree: taxonomy.TaxonomyTree):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.taxonomy_tree = taxonomy_tree
        self.outputs = nn.ModuleList([
            nn.Linear(embed_dim, len(taxonomy_tree.taxonomy_id_map[rank]))
            for rank in range(taxonomy_tree.depth)
        ])
        self.parent_indices = [
            torch.tensor([t.parent.taxonomy_id for t in taxonomy_tree.taxonomy_id_map[rank]])
            for rank in range(1, taxonomy_tree.depth)
        ]

    def _forward_taxonomy(self, x):
        """
        Top-down taxonomy prediction
        """
        slice_tuple = (slice(None),)*(x.ndim - 1)
        results = [self.outputs[0](x)]
        for output, parent_indices in zip(self.outputs[1:], self.parent_indices): # type: ignore
            results.append(output(x) + results[-1][slice_tuple + (parent_indices,)])
        return results

    def forward(self, x):
        y = self.encoder(x)[0]
        return self._forward_taxonomy(y)

    def _step(self, mode, batch):
        x, labels = batch
        y_pred = self(x)
        accuracies = [(torch.argmax(pred, -1) == target).sum()/target.numel() for pred, target in zip(y_pred, labels)]
        losses = [F.cross_entropy(pred, target) for pred, target in zip(y_pred, labels)]
        loss = torch.stack(losses).sum()
        self.log(f"{mode}/loss", loss, prog_bar=True)
        for rank, rank_accuracy, rank_loss in zip(taxonomy.RANKS, accuracies, losses):
            self.log(f"{mode}/loss/{rank.lower()}", rank_loss, prog_bar=True)
        for rank, rank_accuracy, rank_loss in zip(taxonomy.RANKS, accuracies, losses):
            self.log(f"{mode}/accuracy/{rank.lower()}", rank_accuracy, prog_bar=True)
        return loss

    def training_step(self, batch):
        return self._step("training", batch)

    def validation_step(self, batch):
        return self._step("validation", batch)

    def testing_step(self, batch):
        return self._step("testing", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 1e-4) # type: ignore
