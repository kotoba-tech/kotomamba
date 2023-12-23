# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import inspect
from dataclasses import fields
from typing import Any, Type

from peft import (
    LoraConfig,  # type: ignore
    AdaptionPromptConfig,  # type: ignore
    PrefixTuningConfig,  # type: ignore
)

from llama_recipes.configs import (  # noqa: F401
    datasets,
    lora_config,
    llama_adapter_config,
    prefix_config,
    train_config,
    fsdp_config,
)
from llama_recipes.configs.datasets import (
    samsum_dataset,
    grammar_dataset,
    alpaca_dataset,
    ja_wikipedia_dataset,
    ja_en_parallel_dataset,
)
from llama_recipes.utils.dataset_utils import DATASET_PREPROC
from llama_recipes.utils.distributed import print_rank_0


def update_config(
    config: tuple[Type[train_config | fsdp_config]] | Type[train_config | fsdp_config] | Any,
    **kwargs: dict[str, Any],
) -> None:
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")


def generate_peft_config(train_config: Type[train_config], kwargs: dict[str, Any]):
    configs: tuple[Type[lora_config], Type[llama_adapter_config], Type[prefix_config]] = (
        lora_config,
        llama_adapter_config,
        prefix_config,
    )
    peft_configs: tuple[Type[LoraConfig], Type[AdaptionPromptConfig], Type[PrefixTuningConfig]] = (
        LoraConfig,
        AdaptionPromptConfig,
        PrefixTuningConfig,
    )
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"

    config = configs[names.index(train_config.peft_method)]()

    update_config(config, **kwargs)
    params: dict[str, Any] = {k.name: getattr(config, k.name) for k in fields(config)}
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)

    return peft_config


def generate_dataset_config(
    train_config: Type[train_config], kwargs: dict[str, Any]
) -> samsum_dataset | grammar_dataset | alpaca_dataset | ja_wikipedia_dataset | ja_en_parallel_dataset:
    names = tuple(DATASET_PREPROC.keys())

    assert train_config.dataset in names, f"Unknown dataset: {train_config.dataset}"

    dataset_config: samsum_dataset | grammar_dataset | alpaca_dataset | ja_wikipedia_dataset | ja_en_parallel_dataset = {
        k: v for k, v in inspect.getmembers(datasets)
    }[train_config.dataset]()

    print_rank_0(f"dataset_config: {dataset_config}, type({type(dataset_config)})")

    update_config(dataset_config, **kwargs)

    return dataset_config
