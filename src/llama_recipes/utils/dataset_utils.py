# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import importlib
from functools import partial
from pathlib import Path
from typing import Type

import datasets

from llama_recipes.datasets import (
    get_grammar_dataset,
    get_alpaca_dataset,
    get_samsum_dataset,
    get_driver_license_dataset,
    get_ja_en_parallel_dataset,
    get_stability_instruct_dataset,
    get_pubmed_dataset,
    get_pile_dataset,
)

from llama_recipes.configs.datasets import ja_wikipedia_dataset, llm_jp_dataset
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


def get_ja_wikipedia_dataset(dataset_config: Type[ja_wikipedia_dataset], tokenizer, split: str = "train"):
    """日本語Wikipediaのデータから text だけを抽出し、Tokenizeを施し、dataset化する
    context size(= sequence length)は tokenize を行う際に行う

    Args:
        dataset_config (Type[ja_wikipedia_dataset]):
            dataset config (llama_recipes.configs.datasets.py)
        tokenizer : Llama-2 Tokenizer or カスタムTokenizer
        split (str): "train" or "test"

    Returns:
        tokenize済みのデータセット
    """
    raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
        path="json",
        data_files=["/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_train_0.jsonl"],
        num_proc=8,
    )
    print_rank_0(f"raw_dataset: {raw_dataset}")

    # if is_rank_0():
    #     example: str = raw_dataset["train"][0]["text"]
    #     tokens = tokenizer.tokenize(example)
    #     de_tokenized_text: str = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
    #     print(f"raw dataset[0]: {example}, tokens: {tokens}, de-tokenized: {de_tokenized_text}")

    dataset = (
        raw_dataset["train"]
        .map(
            lambda sample: tokenizer(sample["text"]),
            batched=True,
            remove_columns=list(raw_dataset["train"].features),
            num_proc=8,
        )
        .map(Concatenator(chunk_size=dataset_config.context_size), batched=True, num_proc=8)
    )

    split_dataset: datasets.DatasetDict = dataset.train_test_split(test_size=0.05)
    train_dataset: datasets.Dataset = split_dataset["train"]
    val_dataset: datasets.Dataset = split_dataset["test"]

    if split == "train":
        return train_dataset
    else:
        return val_dataset


def get_llm_jp_dataset(dataset_config: Type[llm_jp_dataset], tokenizer, split: str = "train"):
    if split == "train":
        dataset_paths: list[str] = [
            f"/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_train_{i}.jsonl"
            for i in range(38)
        ] + ["/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_train_0.jsonl"]

        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=8,
        )
        print_rank_0(f"train raw_dataset: {raw_dataset}")
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
                num_proc=8,
            )
            .map(Concatenator(chunk_size=dataset_config.context_size), batched=True, num_proc=8)
        )
        return dataset
    else:
        dataset_paths: list[str] = [
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_cc/merged_val_0.jsonl",
            "/bb/llm/gaf51275/llama/datasets/llama2-llm-jp-corpus/v1.0.2/sample/ja_wiki/merged_val_0.jsonl",
        ]
        raw_dataset: datasets.DatasetDict = datasets.load_dataset(  # type: ignore
            path="json",
            data_files=dataset_paths,
            num_proc=8,
        )
        print_rank_0(f"test raw_dataset: {raw_dataset}")
        dataset = (
            raw_dataset["train"]
            .map(
                lambda sample: tokenizer(sample["text"]),
                batched=True,
                remove_columns=list(raw_dataset["train"].features),
                num_proc=8,
            )
            .map(Concatenator(chunk_size=dataset_config.context_size), batched=True, num_proc=8)
        )
        return dataset


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
    "ja_wikipedia_dataset": get_ja_wikipedia_dataset,
    "llm_jp_dataset": get_llm_jp_dataset,
    "driver_license_dataset": partial(get_driver_license_dataset),
    "ja_en_parallel_dataset": partial(get_ja_en_parallel_dataset),
    "stability_instruct_dataset": partial(get_stability_instruct_dataset),
    "pubmed_dataset": partial(get_pubmed_dataset),
    "pile_dataset": partial(get_pile_dataset),
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
