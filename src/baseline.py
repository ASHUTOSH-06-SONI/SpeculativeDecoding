import time
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()

PROMPT = "Translate to Hindi: Hello, how are you?"


def run_baseline(model_name: str, prompt: str, max_new_tokens: int):
    print(f"[baseline] Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[baseline] Loaded on {device}", flush=True)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    generated = []

    print(f"[baseline] Generating {max_new_tokens} tokens autoregressively...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids)
            next_token = int(outputs.logits[0, -1, :].argmax().item())
            generated.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device)], dim=1
            )

    elapsed = time.time() - t0
    tps = max_new_tokens / elapsed
    decoded = tokenizer.decode(generated, skip_special_tokens=True)

    print("\n=== Baseline Results ===")
    print(f"Model:          {model_name}")
    print(f"Device:         {device}")
    print(f"Tokens:         {max_new_tokens}")
    print(f"Total time:     {elapsed:.3f}s")
    print(f"Tokens/sec:     {tps:.3f}")
    print(f"Generated text: {decoded}")
    print("========================")
    print(f"\nSave this number → baseline tokens/sec: {tps:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2-large")
    p.add_argument("--prompt", type=str, default=PROMPT)
    p.add_argument("--max_tokens", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline(args.model_name, args.prompt, args.max_tokens)