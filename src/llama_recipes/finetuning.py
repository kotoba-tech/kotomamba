# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys
import logging
import random

import fire
import numpy as np
import torch
import torch.distributed as torch_distributed
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # type: ignore
import torch.optim as optim
import wandb
import typing
import deepspeed  # noqa: F401
from peft import get_peft_model, prepare_model_for_int8_training  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
)

from llama_recipes.configs import fsdp_config, train_config
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    update_config,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset
from llama_recipes.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    get_policies,
    print_model_size,
    setup_environ_flags,
    train,
)
from llama_recipes.optimizer import WarmupCosineAnnealingLR
from llama_recipes.utils.sequence_length_warmup import (  # noqa: F401
    SequenceLengthWarmupDistributedSampler,  # noqa: F401
    SequenceLengthWarmupDataset,  # noqa: F401
    CustomDistributedSampler,
)
from llama_recipes.utils.random import set_seed
from llama_recipes.utils.distributed import (
    print_rank_0,
    is_rank_0,
    set_mpi_env,
    get_rank,
    get_local_rank,
    get_world_size,
)
from llama_recipes.get_models import get_model
from llama_recipes.get_tokenizer import get_tokenizer
from llama_recipes.get_model_decoder_layer import get_model_decoder_layer
from llama_recipes.utils.checkpoint import (
    load_model_state_dict,
    load_optimizer_state_dict,
    load_scheduler_state_dict,
    load_sampler_state_dict,
)


current_path: str = os.getcwd()
sys.path.append(f"{current_path}/llama-recipes/src/")


def main(**kwargs) -> None:
    # logging 設定
    logging.basicConfig(level=logging.WARNING)

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)  # type: ignore

    # Set the seeds for reproducibility
    set_seed(train_config)

    # Distributed args.
    if train_config.use_mpi:
        set_mpi_env()

    if train_config.enable_fsdp:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch_distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # wandb setting
    if train_config.wandb_name is not None and is_rank_0():
        import datetime
        from llama_recipes.utils.wandb_utils import set_config

        wandb_configs: dict[str, typing.Any] = {}
        set_config(wandb_configs=wandb_configs)

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": train_config.wandb_entity,
            "project": train_config.wandb_project,
            "name": train_config.wandb_name,
            "config": wandb_configs,
        }
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(get_local_rank())  # type: ignore
        clear_gpu_cache(get_local_rank())  # type: ignore
        setup_environ_flags(get_rank())  # type: ignore

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    model = get_model(train_config, use_cache=use_cache)  # type: ignore

    if train_config.load_checkpoint_path:
        load_model_state_dict(model, train_config.load_checkpoint_path)  # type: ignore

    if train_config.enable_fsdp and train_config.use_fast_kernels:
        """
        For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels
        based on the hardware being used. This would speed up fine-tuning.
        """
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)  # type: ignore
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)  # type: ignore

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)  # type: ignore

    tokenizer = get_tokenizer(train_config)  # type: ignore

    if train_config.use_peft:
        print(f"Using PEFT method: {train_config.peft_method}", flush=True)
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)  # type: ignore
        model.print_trainable_parameters()

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            print_rank_0("NOTE: freeze transformer layers")
            freeze_transformer_layers(model=model, num_layer=train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(
            cfg=fsdp_config,
            rank=get_rank(),
            model_name=train_config.model_name,
        )
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(
            model=model,
            transformer_layer_name=get_model_decoder_layer(
                model_name=train_config.model_name,
            )
        )

        model = FSDP(
            model,  # type: ignore
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)  # type: ignore
            if train_config.low_cpu_fsdp and rank != 0
            else None,
        )
        if fsdp_config.fsdp_activation_checkpointing and train_config.model_name:
            apply_fsdp_checkpointing(model=model, model_name=train_config.model_name)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")  # type: ignore

    dataset_config = generate_dataset_config(train_config, kwargs)

    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or is_rank_0():
        print(f"--> Training Set Length = {len(dataset_train)}")  # type: ignore

    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
        if not train_config.enable_fsdp or is_rank_0():
            print(f"--> Validation Set Length = {len(dataset_val)}")  # type: ignore

    """
    estimated_total_iterations: Number of iterations for learning
    lr_warmup_iterations: Number of iterations for learning rate warmup
    lr_decay_iterations: Number of iterations for learning rate decay
    """
    estimated_total_iterations: int = (
        train_config.num_epochs
        * len(dataset_train)  # type: ignore
        // (train_config.batch_size * get_world_size() * train_config.gradient_accumulation_steps)
    )
    lr_warmup_iterations: int = int(estimated_total_iterations * train_config.lr_warmup)
    lr_decay_iterations: int = int(estimated_total_iterations * train_config.lr_decay)

    dataset_length: int = len(dataset_train)  # type: ignore
    if is_rank_0():
        print(f"dataset_train: {dataset_length}")  # type: ignore

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = CustomDistributedSampler(
            dataset_train,
            rank=torch_distributed.get_rank(),
            num_replicas=torch_distributed.get_world_size(),
            shuffle=True,
            seed=train_config.seed,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,  # type: ignore
                rank=torch_distributed.get_rank(),
                num_replicas=torch_distributed.get_world_size(),
                seed=train_config.seed,
            )

    if train_config.load_checkpoint_path:
        load_sampler_state_dict(train_sampler, train_config.load_checkpoint_path)

    # Create DataLoaders for the training and validation dataset
    # NOTE: we need to set worker_init_fn to set seed for each worker
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = train_config.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataloader: DataLoader = DataLoader(
        dataset=dataset_train,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
        worker_init_fn=worker_init_fn,
    )

    eval_dataloader: typing.Optional[DataLoader] = None
    if train_config.run_validation:
        eval_dataloader = DataLoader(
            dataset_val,  # type: ignore
            batch_size=train_config.batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),  # type: ignore
            lr=train_config.lr,
            betas=train_config.adamw_betas,
            eps=train_config.adamw_eps,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),  # type: ignore
            lr=train_config.lr,
            betas=train_config.adamw_betas,
            eps=train_config.adamw_eps,
            weight_decay=train_config.weight_decay,
        )

    if train_config.load_checkpoint_path:
        load_optimizer_state_dict(model, optimizer, train_config.load_checkpoint_path)  # type: ignore

    # wandb config update
    if train_config.wandb_name is not None and is_rank_0():
        # iteration info
        wandb.config.update(
            {
                "total_iteration": estimated_total_iterations,
                "warmup_iteration": lr_warmup_iterations,
                "decay_iteration": lr_decay_iterations,
            }
        )

    if train_config.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=lr_warmup_iterations,
            decay_iterations=lr_decay_iterations,
            max_iterations=estimated_total_iterations,
            eta_min=train_config.lr_min,
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    if train_config.load_checkpoint_path:
        load_scheduler_state_dict(scheduler, train_config.load_checkpoint_path)  # type: ignore

    # Start the training process
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        sampler=train_sampler,  # type: ignore
        tokenizer=tokenizer,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        train_config=train_config,
        fsdp_config=fsdp_config if train_config.enable_fsdp else None,
        local_rank=get_local_rank() if train_config.enable_fsdp else None,
        rank=get_rank() if train_config.enable_fsdp else None,
    )
    if not train_config.enable_fsdp or is_rank_0():
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
