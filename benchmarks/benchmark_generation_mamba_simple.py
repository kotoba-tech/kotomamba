# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from megatron_lm.megatron.tokenizer import build_tokenizer


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--tokenizer-type", type=str, default="tokenizer type")
parser.add_argument("--tokenizer-path", type=str, default="EleutherAI/gpt-neox-20b")
parser.add_argument("--tokenizer-model", type=str, default="EleutherAI/gpt-neox-20b")
parser.add_argument("--use-sentencepiece", action="store_true")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=2048)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument(
    '--vocab-extra-ids', type=int, default=0,
    help='Number of additional vocabulary tokens. They are used for span masking in the T5 model'
)
parser.add_argument(
    '--make-vocab-size-divisible-by', type=int, default=128,
    help='Pad the vocab size to be divisible by this value.This is added for computational efficiency reasons.'
)
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = "mamba" in args.model_name

if is_mamba:
    if args.use_sentencepiece:
        megatron_tokenizer = build_tokenizer(args=args)
        tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.tokenizer_path,
            legacy=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.tokenizer_path,
        )
    model = MambaLMHeadModel.from_pretrained(
        pretrained_model_name=args.model_name,
        device=device,
        dtype=dtype
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.tokenizer_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map={"": device},
        torch_dtype=dtype
    )

model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


torch.random.manual_seed(0)
if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")  # type: ignore
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
    max_length: int = input_ids.shape[1] + args.genlen

if args.use_sentencepiece:
    print(f"DEBUG: eod={megatron_tokenizer.eod}")  # type: ignore

if is_mamba:
    fn = lambda: model.generate(  # noqa:
        input_ids=input_ids,
        max_length=max_length,  # type: ignore
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=megatron_tokenizer.eod if args.use_sentencepiece else tokenizer.eos_token_id  # type: ignore
    )
else:
    fn = lambda: model.generate(  # noqa:
        input_ids=input_ids,
        max_length=max_length,  # type: ignore
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,  # type: ignore
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
out = fn()
if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))  # type: ignore

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(
    f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"  # type: ignore
)
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
