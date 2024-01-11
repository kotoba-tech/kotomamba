import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
from typing import Type
from llama_recipes.configs.datasets import ja_en_parallel_dataset
from pathlib import Path


class JapaneseEnglishParallelDataset(Dataset):
    def __init__(
        self,
        dataset_config: Type[ja_en_parallel_dataset],
        tokenizer: LlamaTokenizer,
        partition: str = "train",
    ) -> None:
        # keys: alignment_score, instruction, input, output, lang_pair
        self.data_file_path: str = (
            dataset_config.train_data_path if partition == "train" else dataset_config.val_data_path
        )

        self.max_words: int = dataset_config.context_size
        self.tokenizer: LlamaTokenizer = tokenizer

        dataset_dir = Path(self.data_file_path).parent
        index_cache_dir = dataset_dir / ".index_cache"
        os.makedirs(index_cache_dir, exist_ok=True)
        index_file_path = index_cache_dir / str(os.path.basename(self.data_file_path)).replace(".jsonl", ".idx")
        self.index_file_path: str = str(index_file_path)

        try:
            with open(self.index_file_path, "r", encoding="utf-8") as f:
                self.indexes: list[int] = [int(line.strip()) for line in f]
        except Exception as e:
            print(f"index file error: {e}")
            exit(1)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        with open(self.data_file_path, "r", encoding="utf-8") as file:
            offset: int = self.indexes[index]
            file.seek(offset)
            try:
                line = file.readline()
            except Exception as e:
                print(f"index={index}, offset={offset}, error={e}")
                exit(1)

            try:
                ann: dict[str, str] = json.loads(line)
            except Exception as e:
                print(f"index={index}, offset={offset}, line={line}, error={e}")
                exit(1)

        prompt: str = f"{ann['instruction']}\n\n{ann['input']}\n\n"

        example: str = prompt + ann["output"]
        encoded_prompt: torch.Tensor = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
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
            random_index: int = np.random.randint(0, len(self.indexes))
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
