# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron global variables."""

import argparse
import typing

from megatron_lm.megatron.tokenizer import build_tokenizer
from megatron_lm.megatron.tokenizer.tokenizer import _SentencePieceTokenizer

_GLOBAL_ARGS = None
_GLOBAL_TOKENIZER = None


def get_args() -> argparse.Namespace:
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_ARGS, 'args')
    return typing.cast(argparse.Namespace, _GLOBAL_ARGS)


def get_tokenizer():
    """Return tokenizer."""
    _ensure_var_is_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    return typing.cast(_SentencePieceTokenizer, _GLOBAL_TOKENIZER)


def set_global_variables(args: argparse.Namespace, build_tokenizer=True) -> None:
    """Set args, tokenizer"""

    assert args is not None

    _ensure_var_is_not_initialized(_GLOBAL_ARGS, 'args')
    set_args(args)

    if build_tokenizer:
        _ = _build_tokenizer(args)


def set_args(args: argparse.Namespace):
    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args


def _build_tokenizer(args: argparse.Namespace):
    """Initialize tokenizer."""
    global _GLOBAL_TOKENIZER
    _ensure_var_is_not_initialized(_GLOBAL_TOKENIZER, 'tokenizer')
    _GLOBAL_TOKENIZER = build_tokenizer(args)
    return _GLOBAL_TOKENIZER


def _ensure_var_is_initialized(var, name: str) -> None:
    """Make sure the input variable is not None."""
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name: str) -> None:
    """Make sure the input variable is not None."""
    assert var is None, '{} is already initialized.'.format(name)
