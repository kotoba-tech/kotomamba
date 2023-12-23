import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Type
from llama_recipes.configs.datasets import pubmed_dataset
from pathlib import Path
import os


class PUBMEDDataset(Dataset):
    def __init__(
        self,
        dataset_config: Type[pubmed_dataset],
        tokenizer: PreTrainedTokenizer,
        partition: str = "train",
    ) -> None:
        # keys: alignment_score, instruction, input, output, lang_pair
        self.data_file_path: str = (
            dataset_config.train_data_path if partition == "train" else dataset_config.val_data_path
        )

        self.max_words: int = dataset_config.context_size
        self.tokenizer: PreTrainedTokenizer = tokenizer

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
        IGNORE_INDEX = -100

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

        english: str = ann['english']
        japanese: str = ann["japanese"]

        encoded_text: list[int] = self.tokenizer.encode(text=english)
        encoded_text.append(self.tokenizer.eos_token_id)  # type: ignore
        encoded_text = encoded_text + self.tokenizer.encode(text=japanese)
        encoded_text.append(self.tokenizer.eos_token_id)  # type: ignore

        encoded_text_tensor = torch.tensor(encoded_text)

        padding_size: int = self.max_words - encoded_text_tensor.size(0)
        if padding_size > 0:
            padding = torch.zeros(padding_size, dtype=torch.long)
            encoded_text_tensor = torch.cat((encoded_text_tensor, padding))
        elif padding_size < 0:
            encoded_text_tensor = encoded_text_tensor[:self.max_words]

        input_ids: torch.Tensor = encoded_text_tensor
        labels: torch.Tensor = encoded_text_tensor.clone()
        labels[0:-1] = labels[1:].clone()
        labels[-1] = IGNORE_INDEX

        attention_mask: torch.Tensor = torch.ones(self.max_words, dtype=torch.long)
        attention_mask[encoded_text_tensor == 0] = 0

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
