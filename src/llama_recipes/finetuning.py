import os
import sys

import torch
import torch.distributed as torch_distributed
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # type: ignore
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.optim.lr_scheduler import StepLR
from peft import prepare_model_for_int8_training  # type: ignore
import wandb

from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    get_policies,
    print_model_size,
    setup_environ_flags,
    train,
)
from llama_recipes.optimizer import WarmupCosineAnnealingLR
from llama_recipes.utils.random import set_seed
from llama_recipes.utils.distributed import (
    print_rank_0,
    is_rank_0,
    set_mpi_env,
    get_rank,
    get_local_rank,
)
from llama_recipes.get_models import get_model
from llama_recipes.utils.checkpoint import (
    load_model_state_dict,
    load_optimizer_state_dict,
    load_scheduler_state_dict,
    load_rng_state_dict,
    get_latest_iteration,
)

from llama_recipes.arguments import parse_args
from llama_recipes.get_fsdp import get_sharding_strategy
from megatron_lm.megatron.global_vars import set_global_variables


current_path: str = os.getcwd()
sys.path.append(f"{current_path}/llama-recipes/src/")


def main() -> None:
    # initialize
    args = parse_args()
    set_global_variables(args=args)

    # Set the seeds for reproducibility
    set_seed(seed=args.seed)

    # Distributed args.
    if args.use_mpi:
        set_mpi_env()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.rank = rank
    args.world_size = world_size
    args.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    torch_distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # wandb setting
    if args.wandb_name is not None and is_rank_0():
        import datetime

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(get_local_rank())  # type: ignore
        clear_gpu_cache(get_local_rank())  # type: ignore
        setup_environ_flags(get_rank())  # type: ignore

    iteration: int = get_latest_iteration(args.load)
    args.iteration = iteration
    torch.distributed.barrier()

    # random seed
    if args.load:
        load_rng_state_dict(args.load)
        torch_distributed.barrier()

    use_cache = False
    model = get_model(
        model_name=args.base_model, use_cache=use_cache
    )

    if args.load:
        load_model_state_dict(model, args.load)  # type: ignore

    if args.use_better_transformer:
        try:
            from optimum.bettertransformer import BetterTransformer

            model = BetterTransformer.transform(model)  # type: ignore
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    print_model_size(model, args.base_model, rank)  # type: ignore

    # Prepare the model for int8 training if quantization is enabled
    if args.quantization:
        model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if args.bf16:
        model.to(torch.bfloat16)  # type: ignore
    elif args.fp16:
        model.to(torch.float16)  # type: ignore

    if args.freeze_layers:
        print_rank_0("NOTE: freeze transformer layers")
        freeze_transformer_layers(model=model, num_layer=args.num_freeze_layers)

    mixed_precision_policy, wrapping_policy = get_policies(
        rank=get_rank(),
        model_name=args.base_model,
    )

    model = FSDP(
        model,  # type: ignore
        auto_wrap_policy=wrapping_policy,
        cpu_offload=CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=get_sharding_strategy(),
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=args.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(  # type: ignore
            device=torch.cuda.current_device(), recurse=False,  # type: ignore
        )
        if args.low_cpu_fsdp and rank != 0
        else None,
    )
    if args.fsdp_activation_checkpointing:
        apply_fsdp_checkpointing(model=model, model_name=args.base_model)

    from llama_recipes.datasets.pretrain_dataset import build_train_valid_test_datasets
    from megatron_lm.megatron.data.data_samplers import build_pretraining_data_loader

    train_dataset, validation_dataset, test_dataset = build_train_valid_test_datasets()

    args.consumed_train_samples = args.global_batch_size * args.iteration
    args.consumed_valid_samples = args.global_batch_size * (
        args.iteration // args.eval_interval) * args.eval_iters

    train_dataloader = build_pretraining_data_loader(
        dataset=train_dataset,
        consumed_samples=args.consumed_train_samples,
    )
    validation_dataloader = build_pretraining_data_loader(
        dataset=validation_dataset,
        consumed_samples=args.consumed_valid_samples,
    )

    if args.bf16 and args.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),  # type: ignore
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),  # type: ignore
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    if args.load:
        load_optimizer_state_dict(model=model, optimizer=optimizer, path=args.load)  # type: ignore

    if args.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=args.lr_warmup_iters,
            decay_iterations=args.lr_decay_iters,
            max_iterations=args.train_iters,
            eta_min=args.min_lr,
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    if args.load:
        load_scheduler_state_dict(scheduler, args.load)  # type: ignore

    # Start the training process
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rank=get_local_rank(),
        rank=get_rank(),
    )


if __name__ == "__main__":
    main()
