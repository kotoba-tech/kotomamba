import argparse

import torch

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from megatron_lm.megatron.tokenizer.tokenizer import _SentencePieceTokenizer
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace transformers model name"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (`model.pth`)")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sentencepiece-tokenizer", action="store_true")
    parser.add_argument("--tokenizer-path", type=str, required=True)
    args = parser.parse_args()

    if args.sentencepiece_tokenizer:
        tokenizer = _SentencePieceTokenizer(
            model_file=args.tokenizer_path,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    print(f"Loading HF model: {args.model}", flush=True)
    model = MambaLMHeadModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        vocab_size=tokenizer.vocab_size,
        from_scratch=True,
    )

    print(f"Loading CKPT: {args.ckpt}", flush=True)
    state_dict = torch.load(args.ckpt, map_location="cpu")

    print("Loading state dict into HF model", flush=True)
    model.load_state_dict(state_dict)

    print("Saving HF model", flush=True)
    model.save_pretrained(save_directory=args.out)


if __name__ == "__main__":
    main()
