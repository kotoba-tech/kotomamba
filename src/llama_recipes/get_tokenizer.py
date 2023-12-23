from transformers import AutoTokenizer, LlamaTokenizer
from llama_recipes.configs import train_config
from typing import Type


def get_tokenizer(train_config: Type[train_config]) -> AutoTokenizer | LlamaTokenizer:
    if "Llama" in train_config.tokenizer_name:
        tokenizer = LlamaTokenizer.from_pretrained(train_config.tokenizer_name)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        return tokenizer  # type: ignore
    elif "Mistral" in train_config.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(train_config.tokenizer_name)
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

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
    else:
        raise NotImplementedError(
            f"Tokenizer {train_config.tokenizer_name} is not supported. Please use Llama or Mistral."
        )
