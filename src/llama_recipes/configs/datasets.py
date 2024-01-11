# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"
    input_length: int = 2048


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"
    input_length: int = 2048


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "llama-recipes/src/llama_recipes/datasets/alpaca_data.json"


@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class ja_wikipedia_dataset:
    dataset: str = "ja_wiki_dataset"
    context_size: int = 4096  # sequence length
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class llm_jp_dataset:
    dataset: str = "llm_jp_dataset"
    context_size: int = 4096  # sequence length
    train_split: str = "train"
    test_split: str = "test"


@dataclass
class driver_license_dataset:
    dataset: str = "driver_license_dataset"
    context_size: int = 4096
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = ""
    val_data_path: str = ""


@dataclass
class ja_en_parallel_dataset:
    dataset: str = "ja_en_parallel_dataset"
    context_size: int = 4096
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = ""
    val_data_path: str = ""


@dataclass
class stability_instruct_dataset:
    dataset: str = "stability_instruct_dataset"
    context_size: int = 4096
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = ""
    val_data_path: str = ""


@dataclass
class pubmed_dataset:
    dataset: str = "pubmed_dataset"
    context_size: int = 4096
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = ""
    val_data_path: str = ""


@dataclass
class pile_dataset:
    dataset: str = "pile_dataset"
    context_size: int = 4096
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = ""
    val_data_path: str = ""
