import importlib
import lightning as L
from pathlib import Path
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from typing import List, Optional, Type, Union

class DbtkModel(PreTrainedModel, L.LightningModule):
    """
    A base class for Dbtk models.

    Nested models can be specified in the configuration. Example:

    ```
    config = PretrainedConfig(
        base="bert-base-uncased",
        base_class="transformers.models.bert.modeling_bert.BertModel"
    )

    class CustomConfig(PretrainedConfig):
        base: Optional[Union[str, Path, PretrainedConfig, PreTrainedModel]] = None
        base_class: Optional[str, Type[PreTrainedModel]] = None

    class CustomModel(DbtkModel):
        config_class = CustomConfig
        sub_models = ["base"]

    ```
    """

    config_class = PretrainedConfig

    # List of sub-model keys to instantiate
    sub_models: List[str] = []

    def __init__(self, config: Optional[Union[PretrainedConfig, dict]] = None):
        if config is None:
            config = self.config_class()
        elif isinstance(config, dict):
            config = self.config_class(**config)
        super().__init__(config)
        for model_key in self.sub_models:
            self.instantiate_sub_model(model_key)

    def instantiate_sub_model(self, model_key: str):
        """
        Instantiate a sub-model from the configuration.

        Args:
            model_key: The key of the sub-model to instantiate.
        """
        config_key = f"{model_key}"
        class_key = f"{model_key}_class"

        # Ensure model key correctly exists in the config
        if not hasattr(self.config, config_key):
            raise ValueError(f"Configuration is missing {config_key}")
        if not hasattr(self.config, class_key):
            raise ValueError(f"Configuration is missing {class_key}")

        # Extract sub model information
        model_config: Optional[Union[str, Path, PretrainedConfig, PreTrainedModel]] = getattr(self.config, config_key)
        model_class: Optional[Union[str, Type[PreTrainedModel]]] = getattr(self.config, class_key)
        model_instance: Optional[PreTrainedModel] = None

        # Load model class if provided
        if isinstance(model_class, str):
            module_name, class_name = model_class.rsplit('.', 1)
            model_class = getattr(importlib.import_module(module_name), class_name)
        model_class: Optional[Type[PreTrainedModel]] = model_class

        # If the config is a dictionary, create a PretrainedConfig or model_class.config_class instance
        if isinstance(model_config, dict):
            if model_class is None:
                model_config = PretrainedConfig.from_dict(model_config)
            else:
                model_config = model_class.config_class.from_dict(model_config)

        # If a model class was provided without a config, instantiate with default config
        if model_class is not None and model_config is None:
            model_config = model_class.config_class()
            model_instance = model_class(model_config)

        # If a model instance was supplied directly,
        elif isinstance(model_config, PreTrainedModel):
            if model_class is not None and model_config.__class__ != model_class:
                raise ValueError(f"Model class {model_class} does not match the class of the provided model {model_config.__class__}")
            model_instance = model_config

        elif isinstance(model_config, PretrainedConfig):
            if model_class is None:
                model_instance = AutoModel.from_config(model_config)
            else:
                model_instance = model_class(model_config)

        elif isinstance(model_config, (str, Path)):
            if model_class is None:
                model_class = AutoModel
            model_instance = model_class.from_pretrained(model_config)

        if model_instance is None:
            assert model_class is None and model_config is None, f"Failed to instantiate nested model: config: {model_config}, class: {model_class}"
        else:
            model_class = ".".join([model_instance.__class__.__module__, model_instance.__class__.__name__])
            model_config = model_instance.config

        setattr(self.config, config_key, model_config)
        setattr(self.config, class_key, model_class)
        setattr(self, model_key, model_instance)
