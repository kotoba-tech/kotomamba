# Standard Library Imports
import json
import multiprocessing
import os
import struct
import time
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

# Third-Party Imports
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

# IPython
from IPython import embed

_INDEX_HEADER = b"MMIDIDX\x00\x00"

class DType(Enum):
    """The np data type Enum for writing/reading the MMapIndexedDataset indices
    """

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: Type[np.number]) -> int:
        """Get the code from the dtype

        Args:
            value (Type[np.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> Type[np.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            Type[np.number]: The dtype
        """
        return getattr(np, cls(value).name)

    @staticmethod
    def size(key: Union[int, Type[np.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, Type[np.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif np.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> Type[np.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            Type[np.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return np.uint16
        else:
            return np.int32


class _IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    
    Note:
        code largely borrowed from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = np.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        t_beg = time.time()
        self.sequence_lengths = np.frombuffer(
            self.bin_buffer, dtype=np.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()

        t_beg = time.time()
        self.sequence_pointers = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()

        t_beg = time.time()
        self.document_indices = np.frombuffer(
            self.bin_buffer,
            dtype=np.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )
        t_end = time.time()

        self.sequence_modes = None
        if multimodal:
            t_beg = time.time()
            self.sequence_modes = np.frombuffer(
                self.bin_buffer,
                dtype=np.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]
        print(f"> total number of documents: {self.document_indices.shape[0] - 1}")


    def __del__(self) -> None:
        """Clean up the object
        """
        self.bin_buffer_mmap._mmap.close()
        del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[np.int32, np.int64, Optional[np.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            Tuple[np.int32, np.int64, Optional[np.int8]]: The pointer, length and mode at
            the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class PILEDataset(Dataset):
    """
    Note:
        code largely borrowed from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/datasets/indexed_dataset.py
    """
    def __init__(
        self,
        dataset_config,
        tokenizer: PreTrainedTokenizer,
        partition: str = "train",
    ) -> None:
        # keys: alignment_score, instruction, input, output, lang_pair
        self.data_file_path: str = (
            dataset_config.train_data_path if partition == "train" else dataset_config.val_data_path
        )

        self.partition = partition
        self.max_words: int = dataset_config.context_size
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.index = None
        self.bin_buffer = None
        self.bin_buffer_mmap = None
        self.initialize(self.data_file_path)

    def initialize(self, path_prefix: str) -> None:
        """Initialize the dataset

        This method is called by MMapIndexedDataset.__init__ during object creation and by
        MMapIndexedDataset.__setstate__ during un-puckling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix
        """
        self.index = _IndexReader(self.data_file_path.replace(".bin",  ".idx"), False)
        self.bin_buffer_mmap = np.memmap((self.data_file_path), mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        # split the dataset into the chunck of self.max_words
        if self.partition == "train":
            self.batches: List[Tuple] = []  # List of (sequence_pointer, sequence_length, sequence_mode)
            
            # Assuming the calculation of total_bytes is correct as per your data structure
            total_bytes = self.index[-1][0] + self.index[-1][1] * DType.size(self.index.dtype)
            current_pointer = self.index[0][0]

            while current_pointer < total_bytes:
                increment = self.max_words * DType.size(self.index.dtype)
                next_pointer = current_pointer + increment

                # Calculate the number of words for this batch
                num_words = min(self.max_words, (total_bytes - current_pointer) // DType.size(self.index.dtype))

                # Create and append the batch
                batch = (current_pointer, num_words, None)
                self.batches.append(batch)

                current_pointer = next_pointer

    def __len__(self) -> int:
        # TODO
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.batches) if self.partition == "train" else len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100

        # Determine the batch or index to use based on the partition
        batch_data = self.batches if self.partition == "train" else self.index
        sequence_pointer, sequence_length, sequence_mode = batch_data[index]

        # Adjust sequence length if not in training partition
        if self.partition != "train":
            sequence_length = min(sequence_length, self.max_words)

        # Load and process the encoded text
        encoded_text = np.frombuffer(
            self.bin_buffer,
            dtype=self.index.dtype,
            count=sequence_length,
            offset=sequence_pointer,
        ).astype(np.int32)  # Convert to int32 to avoid overflow issues

        # Convert to PyTorch tensor and apply padding or trimming
        encoded_text_tensor = torch.tensor(encoded_text, dtype=torch.long)
        padding_size = self.max_words - len(encoded_text_tensor)

        if padding_size > 0:
            padding = torch.zeros(padding_size, dtype=torch.long)
            encoded_text_tensor = torch.cat((encoded_text_tensor, padding))
        elif padding_size < 0:
            raise ValueError("Padding size should not be negative.")

        # Prepare input_ids, labels, and attention_mask
        input_ids = encoded_text_tensor
        labels = torch.cat((encoded_text_tensor[1:], torch.tensor([IGNORE_INDEX])))
        attention_mask = torch.ones(self.max_words, dtype=torch.long)
        attention_mask[-padding_size:] = 0 if padding_size > 0 else 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
