import time
import torch
import torch.distributed as torch_distributed
from torch.distributed.fsdp import (  # noqa: F401
    FullyShardedDataParallel as FSDP,  # type: ignore
    StateDictType,  # type: ignore
    FullStateDictConfig,  # type:ignore : general model non-sharded, non-flattened params
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from pathlib import Path
import os


def get_model_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    return state_dict


def get_optimizer_state_dict(model: FSDP, optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        None,
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = FSDP.optim_state_dict(model, optimizer)

    return state_dict


def save_model_state_dict(model: FSDP, path: str) -> None:
    state_dict = get_model_state_dict(model)
    if torch_distributed.get_rank() == 0:
        print(f"Saving model state dict to {path}")
        torch.save(state_dict, path)
        print(f"Saved model state dict to {path}")


def save_optimizer_state_dict(model: FSDP, optimizer: torch.optim.Optimizer, path: str) -> None:
    state_dict = get_optimizer_state_dict(model, optimizer)
    if torch_distributed.get_rank() == 0:
        print(f"Saving optimizer state dict to {path}")
        torch.save(state_dict, path)
        print(f"Saved optimizer state dict to {path}")


def save_scheduler_state_dict(scheduler: torch.optim.lr_scheduler.LRScheduler, path: str) -> None:
    if torch_distributed.get_rank() == 0:
        print(f"Saving scheduler state dict to {path}")
        torch.save(scheduler.state_dict(), path)
        print(f"Saved scheduler state dict to {path}")


def save_sampler_state_dict(sampler: torch.utils.data.distributed.DistributedSampler, path: str) -> None:
    if torch_distributed.get_rank() == 0:
        print(f"Saving sampler indices to {path}")
        torch.save(sampler.state_dict(), path)
        print(f"Saved sampler indices to {path}")


def save_rng_state(path: str) -> None:
    # PyTorch
    torch_cpu_rng_state = torch.get_rng_state()
    torch_gpu_rng_state = torch.cuda.get_rng_state()
    # Numpy
    import numpy
    np_rng_state = numpy.random.get_state()
    # random
    import random
    py_rng_state = random.getstate()

    # save
    if torch_distributed.get_rank() == 0:
        print(f"Saving RNG states to {path}")
        torch.save({
            'torch_cpu_rng_state': torch_cpu_rng_state,
            'torch_gpu_rng_state': torch_gpu_rng_state,
            'np_rng_state': np_rng_state,
            'py_rng_state': py_rng_state,
        }, path)
        print(f"Saved RNG states to {path}")


def save_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str,
    iteration: int,
) -> None:
    torch_distributed.barrier()

    checkpoint_path: str = get_checkpoint_name(path, iteration)
    os.makedirs(checkpoint_path, exist_ok=True)
    if torch_distributed.get_rank() == 0:
        start = time.time()
        print(f"Saving checkpoint to {checkpoint_path}")

    save_model_state_dict(
        model=model,
        path=f"{checkpoint_path}/model.pt",
    )
    save_optimizer_state_dict(
        model=model,
        optimizer=optimizer,
        path=f"{checkpoint_path}/optimizer.pt",
    )
    save_scheduler_state_dict(
        scheduler=scheduler,
        path=f"{checkpoint_path}/scheduler.pt",
    )
    save_rng_state(
        path=f"{checkpoint_path}/rng.pt",
    )

    torch_distributed.barrier()

    if torch_distributed.get_rank() == 0:
        with open(f"{path}/latest_iteration.txt", "w") as file:
            file.write(str(iteration))
        print(f"Saved checkpoint to {checkpoint_path}, took {time.time() - start:.2f}s")  # type: ignore


def load_model_state_dict(model: torch.nn.Module, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping model loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading model state dict from {latest_checkpoint_path}/model.pt")

    state_dict = torch.load(f"{latest_checkpoint_path}/model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict

    if torch_distributed.get_rank() == 0:
        print(f"Loaded model state dict from {latest_checkpoint_path}/model.pt")


def load_optimizer_state_dict(model: FSDP, optimizer: torch.optim.Optimizer, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        if torch_distributed.get_rank() == 0:
            print(f"No checkpoint found in {path}, skipping optimizer loading")
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)

    if torch_distributed.get_rank() == 0:
        print(f"Loading optimizer state dict from {latest_checkpoint_path}/optimizer.pt")

    state_dict = torch.load(f"{latest_checkpoint_path}/optimizer.pt", map_location="cpu")
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        None,
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict)
        optimizer.load_state_dict(state_dict)
    del state_dict

    if torch_distributed.get_rank() == 0:
        print(f"Loaded optimizer state dict from {latest_checkpoint_path}/optimizer.pt")


def load_scheduler_state_dict(scheduler: torch.optim.lr_scheduler.LRScheduler, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)
    state_dict = torch.load(f"{latest_checkpoint_path}/scheduler.pt", map_location="cpu")
    scheduler.load_state_dict(state_dict)
    del state_dict


def load_sampler_state_dict(sampler: torch.utils.data.distributed.DistributedSampler, path: str) -> None:
    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(path, latest_iteration)
    state_dict = torch.load(f"{latest_checkpoint_path}/sampler.pt", map_location="cpu")
    sampler.load_state_dict(state_dict)
    del state_dict


def load_rng_state_dict(path: str) -> None:
    import numpy
    import random

    latest_iteration: int = get_latest_iteration(path)
    if latest_iteration == 0:
        return

    latest_checkpoint_path: str = get_checkpoint_name(
        path, latest_iteration
    )
    rng_states = torch.load(f"{latest_checkpoint_path}/rng.pt", map_location="cpu")
    torch.set_rng_state(rng_states['torch_cpu_rng_state'])
    torch.cuda.set_rng_state(rng_states['torch_gpu_rng_state'])
    numpy.random.set_state(rng_states['np_rng_state'])
    random.setstate(rng_states['py_rng_state'])

    del rng_states


def read_latest_value(file_path: str) -> int:
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()  # `strip` removes any leading/trailing whitespace
            return int(content)
    except FileNotFoundError:
        if torch_distributed.get_rank() == 0:
            print(f"File not found: {file_path}")
        raise FileNotFoundError
    except ValueError:
        print(f"Unable to convert file content to integer: {file_path}")
        raise ValueError


def get_latest_iteration(path: str) -> int:
    if Path(path).exists():
        try:
            latest_iteration: int = read_latest_value(f"{path}/latest_iteration.txt")
            return latest_iteration
        except (FileNotFoundError, ValueError):
            if torch_distributed.get_rank() == 0:
                print(f"Unable to read latest iteration from {path}/latest_iteration.txt")

    return 0


def get_checkpoint_name(checkpoints_path: str, iteration: int) -> str:
    """Determine the directory name for this rank's checkpoint.

    Args:
        checkpoints_path (str): チェックポイントの保存先
        iteration (int): 学習のiteration

    Returns:
        str: チェエクポイント名
    """
    checkpoint_directory: str = "iter_{:07d}".format(iteration)
    return os.path.join(checkpoints_path, checkpoint_directory)
