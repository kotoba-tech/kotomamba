from transformers import AutoTokenizer, LlamaTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from llama_recipes.configs import train_config
from typing import Type

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = f"{current_dir}/../../"

sys.path.append(project_root_dir)
from megatron_lm.megatron.tokenizer.tokenizer import _SentencePieceTokenizer, _MambaTokenizer


def get_tokenizer(train_config: Type[train_config]) -> (PreTrainedTokenizer | LlamaTokenizer | _SentencePieceTokenizer):
    if "Llama" in train_config.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer  # type: ignore
    elif "Mistral" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer  # type: ignore
    elif "calm2-7b" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-alpha-7b" in train_config.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            "novelai/nerdstash-tokenizer-v1",
            additional_special_tokens=['▁▁']
        )

        return tokenizer  # type: ignore
    elif "stockmark-13b" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name
        )

        return tokenizer  # type: ignore
    elif "plamo-13b" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name,
            trust_remote_code=True,
        )

        return tokenizer  # type: ignore
    elif "llm-jp-13b-v1.0" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name
        )

        return tokenizer  # type: ignore
    elif "ELYZA-japanese-Llama-2-7b" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name
        )

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-ja_vocab-beta-7b" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.tokenizer_name
        )

        return tokenizer  # type: ignore
    elif "japanese-stablelm-base-beta" in train_config.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            train_config.tokenizer_name,
        )

        return tokenizer  # type: ignore

    elif "EleutherAI/gpt-neox-20b" in train_config.tokenizer_name:
        tokenizer = _MambaTokenizer(
            train_config.tokenizer_name
        )

        return tokenizer  # type: ignore

    elif "llm-jp" in train_config.tokenizer_name:
        tokenizer = tokenizer = _SentencePieceTokenizer(train_config.tokenizer_name)

        return tokenizer  # type: ignore

    else:
        raise NotImplementedError(
            f"Tokenizer {train_config.tokenizer_name} is not supported. Please use Llama or Mistral."
        )
