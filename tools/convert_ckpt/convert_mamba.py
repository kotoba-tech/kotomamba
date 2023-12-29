import argparse

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace transformers model name"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (`model.pth`)")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    print(f"Loading HF model: {args.model}", flush=True)
    model = MambaLMHeadModel.from_pretrained(
        args.model,
        dtype=torch.float16,
    )

    print(f"Loading CKPT: {args.ckpt}", flush=True)
    state_dict = torch.load(args.ckpt, map_location="cpu")

    print("Loading state dict into HF model", flush=True)
    model.load_state_dict(state_dict)

    print("Saving HF model", flush=True)
    model.save_pretrained(save_directory=args.out)


if __name__ == "__main__":
    main()
