import lightning as L

class LightningModuleWithHyperparameters(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()