import lightning as L
from pathlib import Path
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Union

class DbtkModel(PreTrainedModel, L.LightningModule):
    config_class = PretrainedConfig

    def __init__(self, config: Optional[Union[PretrainedConfig, dict]] = None):
        if config is None:
            config = self.config_class()
        elif isinstance(config, dict):
            config = self.config_class(**config)
        super().__init__(config)

    def instantiate_model(self, config_key: str, model_class: type) -> PreTrainedModel:
        value = getattr(self.config, config_key)
        # Setup base model
        if isinstance(value, model_class):
            model = value
        elif isinstance(value, (str, Path)):
            model = model_class.from_pretrained(value)
        else:
            model = model_class(value)
        setattr(self.config, config_key, model.config)
        return model
