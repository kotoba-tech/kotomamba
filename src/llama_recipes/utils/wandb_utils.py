from llama_recipes.configs import fsdp_config, train_config
import torch
import wandb
import os
import time
from typing import Any


def set_config(wandb_configs: dict) -> None:
    # train_config
    wandb_configs["model_name"] = train_config.model_name
    wandb_configs["enable_fsdp"] = train_config.enable_fsdp
    wandb_configs["low_cpu_fsdp"] = train_config.low_cpu_fsdp
    wandb_configs["run_validation"] = train_config.run_validation
    wandb_configs["batch_size_training"] = train_config.batch_size_training
    wandb_configs["gradient_accumulation_steps"] = train_config.gradient_accumulation_steps
    wandb_configs["num_epochs"] = train_config.num_epochs
    wandb_configs["num_workers_dataloader"] = train_config.num_workers_dataloader
    wandb_configs["lr"] = train_config.lr
    wandb_configs["lr_min"] = train_config.lr_min
    wandb_configs["lr_decay"] = train_config.lr_decay
    wandb_configs["lr_warmup"] = train_config.lr_warmup
    wandb_configs["lr_decay_style"] = train_config.lr_decay_style
    wandb_configs["use_sequence_length_schedule"] = train_config.use_sequence_length_schedule
    wandb_configs["weight_decay"] = train_config.weight_decay
    wandb_configs["gamma"] = train_config.gamma
    wandb_configs["adamw_eps"] = train_config.adamw_eps
    wandb_configs["adamw_betas"] = train_config.adamw_betas
    wandb_configs["seed"] = train_config.seed
    wandb_configs["use_fp16"] = train_config.use_fp16
    wandb_configs["mixed_precision"] = train_config.mixed_precision
    wandb_configs["val_batch_size"] = train_config.val_batch_size
    wandb_configs["dataset"] = train_config.dataset
    wandb_configs["peft_method"] = train_config.peft_method
    wandb_configs["use_peft"] = train_config.use_peft
    wandb_configs["freeze_layers"] = train_config.freeze_layers
    wandb_configs["num_freeze_layers"] = train_config.num_freeze_layers
    wandb_configs["quantization"] = train_config.quantization
    wandb_configs["one_gpu"] = train_config.one_gpu
    wandb_configs["save_model"] = train_config.save_model
    wandb_configs["save_optimizer"] = train_config.save_optimizer
    wandb_configs["use_fast_kernels"] = train_config.use_fast_kernels
    wandb_configs["use_mpi"] = train_config.use_mpi

    # fsdp_config
    wandb_configs["mixed_precision"] = fsdp_config.mixed_precision
    wandb_configs["use_fp16"] = fsdp_config.use_fp16
    wandb_configs["sharding_strategy"] = fsdp_config.sharding_strategy
    wandb_configs["checkpoint_type"] = fsdp_config.checkpoint_type
    wandb_configs["fsdp_activation_checkpointing"] = fsdp_config.fsdp_activation_checkpointing
    wandb_configs["pure_bf16"] = fsdp_config.pure_bf16
    wandb_configs["optimizer"] = fsdp_config.optimizer
    wandb_configs["fsdp_cpu_offload"] = fsdp_config.fsdp_cpu_offload


def log_model_info(model: torch.nn.Module) -> None:
    model_config: dict[str, Any] = {}
    model_config["activation_function"] = model.config.hidden_act
    model_config["hidden_size"] = model.config.hidden_size
    model_config["model_type"] = model.config.model_type
    model_config["max_position_embeddings"] = model.config.max_position_embeddings
    model_config["num_attention_heads"] = model.config.num_attention_heads
    model_config["num_hidden_layers"] = model.config.num_hidden_layers
    model_config["vocab_size"] = model.config.vocab_size
    model_config["model_architecture"] = model.config.architectures[0]

    print(f"model info: {model}")
    print(f"model config: {model.config}")
    wandb.config.update(model_config)

    # distributed training info
    world_size = int(os.environ["WORLD_SIZE"])
    wandb.config.update({"world_size": world_size})


def log_wandb(
    batch: dict[str, torch.Tensor],
    model: torch.nn.Module,
    accumulation_loss: float,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    gradient_accumulation_steps: int,
    world_size: int,
    iteration_start_time: float,
    wandb_iteration: int,
) -> None:
    wandb_stats: dict[str, Any] = {}

    # training info
    wandb_stats["training/loss"] = accumulation_loss
    wandb_stats["training/perplexity"] = torch.exp(torch.tensor(accumulation_loss)).item()

    # utils info
    batch_size: int = batch["input_ids"].shape[0]
    sequence_length: int = batch["input_ids"].shape[1]

    wandb_stats["utils/batch_size"] = batch_size
    wandb_stats["utils/global_batch_size"] = batch_size * world_size * gradient_accumulation_steps
    wandb_stats["utils/seq_len"] = sequence_length
    wandb_stats["utils/gradient_accumulation_steps"] = gradient_accumulation_steps
    wandb_stats["utils/epoch"] = epoch
    wandb_stats["utils/step"] = step

    # optimizer info
    wandb_stats["optimizer/lr"] = optimizer.param_groups[0]["lr"]

    optimizer_states_1: list[float] = [0.0] * 8
    optimizer_states_2: list[float] = [0.0] * 4

    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            # optimizer state が空の場合は logging しない
            if not optimizer.state:
                continue
            if "exp_avg_sq" not in optimizer.state[param].keys():
                continue

            optimizer_states_1[0] += (torch.norm(optimizer.state[param]["exp_avg_sq"]).item()) ** 2  # type: ignore
            optimizer_states_1[1] += (
                torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt()).item()  # type: ignore
            ) ** 2
            optimizer_states_1[2] += (torch.norm(optimizer.state[param]["exp_avg"]).item()) ** 2  # type: ignore
            optimizer_states_1[3] += (torch.norm(param).item()) ** 2  # type: ignore
            optimizer_states_1[4] += torch.norm(optimizer.state[param]["exp_avg_sq"], p=1).item()  # type: ignore
            optimizer_states_1[5] += torch.norm(optimizer.state[param]["exp_avg_sq"].sqrt(), p=1).item()  # type: ignore
            optimizer_states_1[6] += torch.norm(optimizer.state[param]["exp_avg"], p=1).item()  # type: ignore
            optimizer_states_1[7] += torch.norm(param, p=1).item()
            optimizer_states_2[0] = max(
                optimizer_states_2[0],  # type: ignore
                abs(optimizer.state[param]["exp_avg_sq"].max().item()),  # type: ignore
                abs(optimizer.state[param]["exp_avg_sq"].min().item()),  # type: ignore
            )
            optimizer_states_2[1] = max(
                optimizer_states_2[1],
                optimizer.state[param]["exp_avg_sq"].sqrt().abs_().max().item(),  # type: ignore
            )
            optimizer_states_2[2] = max(
                optimizer_states_2[2],  # type: ignore
                abs(optimizer.state[param]["exp_avg"].max().item()),  # type: ignore
                abs(optimizer.state[param]["exp_avg"].min().item()),  # type: ignore
            )
            optimizer_states_2[3] = max(
                optimizer_states_2[3],
                abs(param.max().item()),  # type: ignore
                abs(param.min().item()),  # type: ignore
            )
    if optimizer.state:  # optimizer stateがない場合はloggingしない
        # rank:0でしかoptimizer stateをloggingしないので world sizeで割る必要はない
        wandb_stats["optimizer/variance_l2"] = optimizer_states_1[0] ** 0.5
        wandb_stats["optimizer/variance_sqrt_l2"] = optimizer_states_1[1] ** 0.5
        wandb_stats["optimizer/momentum_l2"] = optimizer_states_1[2] ** 0.5
        wandb_stats["optimizer/weight_l2"] = optimizer_states_1[3] ** 0.5
        wandb_stats["optimizer/variance_l1"] = optimizer_states_1[4]
        wandb_stats["optimizer/variance_sqrt_l1"] = optimizer_states_1[5]
        wandb_stats["optimizer/momentum_l1"] = optimizer_states_1[6]
        wandb_stats["optimizer/weight_l1"] = optimizer_states_1[7]
        wandb_stats["optimizer/variance_abs_max"] = optimizer_states_2[0]
        wandb_stats["optimizer/variance_sqrt_abs_max"] = optimizer_states_2[1]
        wandb_stats["optimizer/momentum_abs_max"] = optimizer_states_2[2]
        wandb_stats["optimizer/weight_abs_max"] = optimizer_states_2[3]

    # stats
    iteration_elapsed_time = time.perf_counter() - iteration_start_time

    tokens_per_sec = batch_size * sequence_length * gradient_accumulation_steps / iteration_elapsed_time * world_size
    wandb_stats["stats/1_iteration_time"] = iteration_elapsed_time
    wandb_stats["stats/tokens_per_sec"] = tokens_per_sec
    wandb_stats["stats/tokens_per_sec_per_gpu"] = tokens_per_sec / world_size

    checkpoint_activations_factor = 3
    if fsdp_config is not None and fsdp_config.fsdp_activation_checkpointing:  # type ignore
        checkpoint_activations_factor = 4

    num_layers: int = model.config.num_hidden_layers
    hidden_size: int = model.config.hidden_size
    vocab_size: int = model.config.vocab_size
    activation_func: str = model.config.hidden_act
    intermediate_size: int = model.config.intermediate_size

    activation_function_factor: int = 4  # GELU
    if activation_func == "silu":
        activation_function_factor = 4 + 2  # SWiGLU (upscaling + down scaling)

    # tflops calculation
    flops_per_iteration: float = (
        (8 + activation_function_factor * (intermediate_size / hidden_size))
        * checkpoint_activations_factor
        * batch_size
        * sequence_length
        * gradient_accumulation_steps
        * num_layers
        * (hidden_size**2)
    ) * (1.0 + (sequence_length / (6.0 * hidden_size)) + (vocab_size / (16.0 * num_layers * hidden_size)))
    tflops: float = flops_per_iteration / (iteration_elapsed_time * (10**12))
    wandb_stats["stats/tflops"] = tflops

    wandb.log(wandb_stats, step=wandb_iteration + 1)

    print("------------------------------------------------------------------")
    print(f"iteration: {wandb_iteration + 1} , tflops: {tflops}, loss: {accumulation_loss}")
    print(
        "------------------------------------------------------------------",
        flush=True,
    )
