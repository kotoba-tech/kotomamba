# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time
import os

from torch.distributed.fsdp import (  # noqa: F401
    FullyShardedDataParallel as FSDP,  # type: ignore
    StateDictType,  # type: ignore
    FullStateDictConfig,  # type:ignore : general model non-sharded, non-flattened params
    LocalStateDictConfig,  # type: ignore : flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed._shard.checkpoint import (  # noqa: F401
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.default_planner import (  # noqa: F401
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)
import torch.distributed.checkpoint as dist_cp
import torch.distributed as torch_distributed

from llama_recipes.configs import train_config
from typing import Type, Any, Optional

from llama_recipes.utils.distributed import print_rank_0, is_rank_0


def get_date_of_run() -> str:
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run: str = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}", flush=True)
    return date_of_run


# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


def load_model_sharded(
    model: FSDP,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rank: int,
    cfg: Type[train_config],
) -> tuple[int, int]:
    load_dir: str = cfg.load_checkpoint_path

    if not Path(load_dir).exists():
        if rank == 0:
            print("No sharded_state_dict checkpoint directory found...skipping")
        return 0, 0

    try:
        last_iteration: int = read_latest_value(f"{load_dir}/latest")
    except FileNotFoundError or ValueError:
        if rank == 0:
            print("No sharded_state_dict checkpoint directory found...skipping")
        return 0, 0

    if rank == 0:
        print(f"loading model from model path: {load_dir}, iteration: {last_iteration}")
    reader = FileSystemReader(
        get_checkpoint_name(checkpoints_path=str(load_dir), iteration=last_iteration)
    )

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict: dict[str, Any] = {
            "model": model.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }

        dist_cp.load_state_dict(  # type: ignore
            state_dict=state_dict,
            storage_reader=reader,
        )
        model.load_state_dict(state_dict["model"])

        optimizer_state: STATE_DICT_TYPE = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=reader,
        )
        flattened_optimizer_state = FSDP.optim_state_dict_to_load(
            model=model, optim=optimizer, optim_state_dict=optimizer_state["optim"]
        )
        optimizer.load_state_dict(flattened_optimizer_state)
        # scheduler.load_state_dict(state_dict["scheduler"])

    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")

    iteration: int = last_iteration
    consumed_tokens: int = 0

    # schedulerの状態を復元する
    for _ in range(iteration):
        scheduler.step()

    print_rank_0(f"iteration: {iteration} is loaded")

    return iteration, consumed_tokens


def save_model_and_optimizer_sharded(
    model: FSDP,
    rank: int,
    cfg: Type[train_config],
    optimizer: Optional[torch.optim.AdamW] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    epoch: Optional[int] = None,
    iteration: Optional[int] = None,
    consumed_tokens: Optional[int] = None,
) -> None:
    """
    save model and optimizer via sharded_state_dict to save_dir

    Args:
        model: モデル
        rank: rank of distributed process
        cfg: train config
        optim: optimizer (AdamW)
    """

    save_dir: str = cfg.save_checkpoint_path
    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(  # type: ignore
        get_checkpoint_name(
            checkpoints_path=str(save_dir), iteration=iteration if iteration is not None else 0
        )
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict: dict[str, Any] = {"model": model.state_dict()}
        if optimizer is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model=model, optim=optimizer)
        if scheduler is not None:
            state_dict["scheduler"] = scheduler.state_dict()
        if epoch is not None:
            state_dict["epoch"] = epoch
        if iteration is not None:
            state_dict["iteration"] = iteration
        if consumed_tokens is not None:
            state_dict["consumed_tokens"] = consumed_tokens

        dist_cp.save_state_dict(  # type: ignore
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
        )
        if rank == 0:
            print("checkpoint after save_state_dict()")
            ck = state_dict.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
    torch_distributed.barrier()

    # 最新の checkpoint iteration を記録する
    if is_rank_0():
        with open(f"{save_dir}/latest", "w") as f:
            f.write(str(iteration))

    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}, iteration={iteration}")
        print(f"Checkpoint Time = {t1-t0:.4f}\n")


def save_model_checkpoint(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
) -> None:
    """saving model via rank0 cpu streaming and full_state_dict"""

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()

        print(f"saving process: rank {rank}  done w model state_dict\n")

    if rank == 0:
        print("--> saving model ...")
        # create save path
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = cfg.model_name + "-" + str(epoch) + ".pt"
        save_full_path = str(save_dir) + "/" + save_name

        # save model
        torch.save(cpu_state, save_full_path)
        print(f"model checkpoint saved for epoch {epoch} at {save_full_path}\n")


def load_model_checkpoint(model, rank, cfg) -> None:
    """load local checkpoint to rank0 cpu
    must be called * before * passing to FSDP"""

    if rank != 0:
        return

    # where is the checkpoint at...
    full_state_dict_model_path = Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    # is it present...
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    model_checkpoint = torch.load(full_state_dict_model_path)
    # integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print("model checkpoint loaded to rank0 cpu")


def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """save optimizer state via full state dict"""

    print(f"--> optim state call on rank {rank}\n")

    # pull all sharded optimizer states to rank0 cpu...

    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    print_rank_0(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        opt_save_name = "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        opt_save_full_path = save_dir / opt_save_name

        print("--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank: int) -> None:
    """load an fsdp optimizer full_state checkpoint using scatter method
    this ensures only rank 0 loads the optimizer state dict and scatters to other ranks
    """

    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    full_osd = None

    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # called from all ranks, though only rank0 has a valid param for full_osd
    sharded_osd = FSDP.scatter_full_optim_state_dict(  # noqa: F841
        full_optim_state_dict=full_osd, model=model
    )  # noqa: F841

    print(f"optimizer shard loaded on rank {rank}")


def load_sharded_model_single_gpu(model, model_path: str):
    state_dict = {"model": model.state_dict()}

    dist_cp.load_state_dict(  # type: ignore
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,
    )

    model.load_state_dict(state_dict["model"])
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model


def save_checkpoint(
    model,
    optimizer: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_config: Type[train_config],
    rank: int,
    epoch: int,
    iteration: int,
) -> None:
    if train_config.checkpoint_type == StateDictType.FULL_STATE_DICT:  # type: ignore
        save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
        if train_config.save_optimizer:
            save_optimizer_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
            print_rank_0(f" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT, rank {rank}")
            print_rank_0("=====================================================")

    # ABCI Llama-2 Continual Learning use below
    elif train_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:  # type: ignore
        print(f" Saving the FSDP model checkpoints using SHARDED_STATE_DICT, rank {rank}")
        print("=====================================================")

        save_model_and_optimizer_sharded(model, rank, train_config, iteration=iteration)
        if train_config.save_optimizer:
            save_model_and_optimizer_sharded(
                model,
                rank,
                train_config,
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=iteration,
            )
            print(f" Saved the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT, rank {rank}")
            print("=====================================================")


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


def read_latest_value(file_path: str) -> int:
    try:
        with open(file_path, "r") as file:
            content = file.read().strip()  # `strip` removes any leading/trailing whitespace
            return int(content)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except ValueError:
        print(f"Unable to convert file content to integer: {file_path}")
        raise


def get_model_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    return state_dict


def get_optimizer_state_dict(
    model: FSDP, optimizer: torch.optim.Optimizer
) -> dict[str, torch.Tensor]:
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
        torch.save(state_dict, path)


def save_optimizer_state_dict(model: FSDP, optimizer: torch.optim.Optimizer, path: str) -> None:
    state_dict = get_optimizer_state_dict(model, optimizer)
    if torch_distributed.get_rank() == 0:
        torch.save(state_dict, path)
