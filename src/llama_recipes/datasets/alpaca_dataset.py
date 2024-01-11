# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer: LlamaTokenizer, partition: str = "train", max_words: int = 30) -> None:
        # ann: keys: instruction, input, output
        self.ann: list[dict[str, str]] = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words: int = max_words
        self.tokenizer: LlamaTokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.ann)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann: dict[str, str] = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        example: str = prompt + ann["output"]
        encoded_prompt: torch.Tensor = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        encoded_example: list[int] = self.tokenizer.encode(example)
        encoded_example.append(self.tokenizer.eos_token_id)  # type: ignore
        encoded_tensor_example: torch.Tensor = torch.tensor(encoded_example, dtype=torch.int64)

        padding: int = self.max_words - encoded_tensor_example.shape[0]
        if padding > 0:
            encoded_tensor_example = torch.cat((encoded_tensor_example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            encoded_tensor_example = encoded_tensor_example[: self.max_words]

        labels = copy.deepcopy(encoded_tensor_example)
        # promptの長さ分だけ -1 で埋める -> 損失関数で無視するようになる
        labels[: len(encoded_prompt)] = -1
        # 0より大きい(ge)かどうかの真偽値でmaskを作成
        example_mask = encoded_tensor_example.ge(0)
        label_mask = labels.ge(0)

        if torch.all(label_mask == 0):
            random_index: int = np.random.randint(0, len(self.ann))
            self.__getitem__(random_index)

        # ~example_mask -> paddingの部分を 0 で埋める
        encoded_tensor_example[~example_mask] = 0
        # ~label_mask -> prompt の部分を ignore_index で埋める
        labels[~label_mask] = IGNORE_INDEX

        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": encoded_tensor_example,
            "labels": labels,
            "attention_mask": example_mask,
        }
