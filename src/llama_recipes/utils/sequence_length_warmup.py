from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch
from typing import Iterator, Optional, Any
import math
import datasets


class SequenceLengthWarmupDataset(Dataset):
    def __init__(self, data: datasets.Dataset, initial_seq_len: int = 64) -> None:
        """
        Args:
            data: tokenize済みデータ (datasets.Dataset)
        """
        import time

        start_time = time.time()
        self.data = data
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        self.sequence_length: int = initial_seq_len

    def set_sequence_length(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        print(f"index: {index}, sequence_length: {self.sequence_length}, Dataset __getitem__() is called", flush=True)
        # シーケンスの長さを制限
        item: torch.Tensor = self.data["input_ids"][index]
        item = item[: self.sequence_length]
        return item


class SequenceLengthWarmupDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: SequenceLengthWarmupDataset,
        dataset_length: int,
        num_replicas=None,
        rank: Optional[int] = None,
        shuffle=True,
        seed: int = 42,
        start_iteration: int = 0,
        max_sequence_length: int = 4096,
        warmup_iterations: int = 80,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed=seed)
        self.max_sequence_length: int = max_sequence_length
        self.warmup_iterations: int = warmup_iterations
        self.current_iteration = 0
        self.start_iteration: int = start_iteration
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        self.dataset_length: int = dataset_length

    def set_iteration(self, iteration: int) -> None:
        self.current_iteration: int = iteration

    def state_dict(self) -> dict[str, Any]:
        return {
            "start_iteration": self.start_iteration,
            "seed": self.seed,
            "generator": self.generator.get_state(),
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch: int = epoch
        self.generator.manual_seed(self.seed + self.epoch)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.start_iteration: int = state_dict["start_iteration"]
        self.seed: int = state_dict["seed"]
        self.generator.set_state(state_dict["generator"])

    def __iter__(self) -> Iterator[int]:
        print(f"Sampler __iter__() is called. iter={self.current_iteration}", flush=True)
        # シーケンスの長さを計算
        sequence_length: int = 64 + int(
            (self.max_sequence_length - 64) * min(1.0, self.current_iteration / self.warmup_iterations)
        )
        # sequence lengthをdatasetにセット
        self.dataset.set_sequence_length(sequence_length)  # type: ignore

        if self.shuffle:
            self.generator.manual_seed(self.seed + self.current_iteration)
            indices = torch.randperm(self.dataset_length, generator=self.generator).tolist()  # type: ignore
        else:
            indices = list(range(self.dataset_length))  # type: ignore

        return iter(indices)


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)  # seed is defined in the parent class
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch: int = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"current_epoch": self.current_epoch, "generator_state": self.generator.get_state()}

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict["current_epoch"]
        self.generator.set_state(state_dict["generator_state"])

    def __iter__(self):
        # Set the seed for shuffling based on the current epoch and generator state
        g = torch.Generator()
        g.set_state(self.generator.get_state())
        g.manual_seed(self.current_epoch)

        # dataset
        dataset_length: int = len(self.dataset)  # type: ignore

        # The rest of the implementation is similar to the original DistributedSampler
        # Get the number of samples per process and the start index for the current process
        num_samples = int(math.ceil(dataset_length * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas
        self.num_samples = num_samples

        # Shuffle dataset or generate a linear sequence
        if self.shuffle:
            indices = torch.randperm(dataset_length, generator=g).tolist()
        else:
            indices = list(range(dataset_length))

        # Add extra samples to make it evenly divisible
        indices += indices[: (total_size - len(indices))]
        assert len(indices) == total_size

        # Subsample for the current process
        offset = self.rank * self.num_samples
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        # Update the generator state after shuffling
        self.generator.set_state(g.get_state())

        return iter(indices)
