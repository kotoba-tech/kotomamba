# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path

from llama_recipes.datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_customer_support_dataset,
    get_driver_license_dataset,
    get_ja_en_parallel_dataset,
    get_stability_instruct_dataset,
    get_pubmed_dataset,
)

from llama_recipes.datasets.utils import Concatenator
from llama_recipes.utils.distributed import print_rank_0, is_rank_0  # noqa: F401

from torch.utils.data import Dataset
from typing import Any


def load_module_from_py_file(py_file: str) -> object:
    """
    This method loads a module from a py file which is not in the Python path
    """
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)  # type: ignore
    spec = importlib.util.spec_from_loader(module_name, loader)  # type: ignore
    module = importlib.util.module_from_spec(spec)  # type: ignore

    loader.exec_module(module)

    return module


def get_custom_dataset(dataset_config, tokenizer, split: str):
    if ":" in dataset_config.file:
        module_path, func_name = dataset_config.file.split(":")
    else:
        module_path, func_name = dataset_config.file, "get_custom_dataset"

    if not module_path.endswith(".py"):
        raise ValueError(f"Dataset file {module_path} is not a .py file.")

    module_path = Path(module_path)
    if not module_path.is_file():
        raise FileNotFoundError(f"Dataset py file {module_path.as_posix()} does not exist or is not a file.")

    module = load_module_from_py_file(module_path.as_posix())
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except AttributeError as e:
        print(
            f"It seems like the given method name ({func_name}) is not present in the dataset .py file ({module_path.as_posix()})."
        )
        raise e


def apply_qa_prompt_template(sample: dict[str, Any], tokenizer) -> dict[str, Any]:
    prompt = "次のお問い合わせに適切に回答してください:\n表題: {{title}}\nお問い合わせ: {{voice}}\n--\n回答:\n{{reply}}{{eos_token}}"

    return {
        "text": prompt.format(
            title=sample["title"],
            voice=sample["voice"],
            reply=sample["reply"],
            eos_token=tokenizer.eos_token,
        )
    }


DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset, max_words=2048),
    "grammar_dataset": get_grammar_dataset,
    "samsum_dataset": get_samsum_dataset,
    "custom_dataset": get_custom_dataset,
    "customer_support_dataset": partial(get_customer_support_dataset),
    "driver_license_dataset": partial(get_driver_license_dataset),
    "ja_en_parallel_dataset": partial(get_ja_en_parallel_dataset),
    "stability_instruct_dataset": partial(get_stability_instruct_dataset),
    "pubmed_dataset": partial(get_pubmed_dataset),
}


def get_preprocessed_dataset(tokenizer, dataset_config, split: str = "train") -> Dataset:
    if dataset_config.dataset not in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return dataset_config.train_split if split not in ["test", "val"] else dataset_config.test_split

    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        get_split(),
    )
