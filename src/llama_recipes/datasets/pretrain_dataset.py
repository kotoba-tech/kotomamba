import torch.distributed as torch_distributed

from llama_recipes.utils.distributed import print_rank_0

from megatron_lm.megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron_lm.megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron_lm.megatron.global_vars import get_tokenizer, get_args


def is_dataset_built_on_rank() -> bool:
    return torch_distributed.is_initialized()


def core_gpt_dataset_config_from_args() -> GPTDatasetConfig:
    args = get_args()

    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        eod_id=get_tokenizer().eod
    )


def train_valid_test_datasets_provider(train_val_test_num_samples: list[int]):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        core_gpt_dataset_config_from_args()
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_datasets():
    """Build pretraining datasets."""

    args = get_args()

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size

    eval_iters: int = (args.train_iters // args.eval_interval + 1) * args.eval_iters
    test_iters: int = args.eval_iters
    train_val_test_num_samples: list[int] = [train_samples,
                                             eval_iters * args.global_batch_size,
                                             test_iters * args.global_batch_size]

    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

    # Build the datasets.
    return train_valid_test_datasets_provider(train_val_test_num_samples)
