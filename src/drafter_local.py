"""
Deterministic debug drafter that always samples EXACTLY block_size tokens by
sampling directly from the model logits (no generate() early stopping).
Tries : google/gemma-3-270m (uses HUGGINGFACE_HUB_TOKEN), falls back to gpt2.
Run with: python -u drafter_force_generate.py
"""
import os, sys, time, traceback, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()  # quiet HF warnings so we see our prints

CANDIDATES = ["google/gemma-3-270m", "gpt2"]
BLOCK_SIZE = 4
TEMPERATURE = 1.0
TOP_P = 0.95
SEED = 12345

def try_load(name, token):
    try:
        print(f"[TRY] Loading tokenizer for: {name}", flush=True)
        tok = AutoTokenizer.from_pretrained(name, use_fast=True, **({"use_auth_token": token} if token else {}))
        print(f"[OK] Tokenizer loaded. vocab_size={getattr(tok,'vocab_size',None)}", flush=True)
    except Exception as e:
        print(f"[ERR] Tokenizer load failed for {name}: {e}", flush=True)
        return None, None, None

    try:
        print(f"[TRY] Loading model for: {name} (weights)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(name, **({"use_auth_token": token} if token else {}))
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Moving model to device: {device}", flush=True)
        model = model.to(device)
        model.eval()
        print(f"[OK] Model loaded on {device}", flush=True)
        return tok, model, device
    except Exception as e:
        print(f"[ERR] Model load failed for {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None, None, None

def sample_next_token_from_logits(logits, temperature=1.0, top_p=0.95):
    """
    logits: torch tensor shape (vocab,)
    returns (token_id, logprob)
    """
    # convert to probs (work in logits for numerical stability)
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # nucleus/top-p sampling implementation
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        cutoff = torch.searchsorted(cumulative, top_p)
        # keep indices upto cutoff (inclusive)
        keep = sorted_indices[: cutoff+1]
        filtered = probs.clone()
        mask = torch.ones_like(filtered, dtype=torch.bool)
        mask[keep] = False
        filtered[mask] = 0.0
        if filtered.sum() == 0:
            filtered = probs  # fallback
        probs = filtered / filtered.sum()

    # sample
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    logprob = float(torch.log(probs[token_id] + 1e-45))
    return token_id, logprob

def autoregressive_sample_block(tokenizer, model, device, prompt, block_size=BLOCK_SIZE, temperature=TEMPERATURE, top_p=TOP_P):
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    all_sampled = []
    all_logps = []
    for i in range(block_size):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]   # (vocab,)
            tid, logp = sample_next_token_from_logits(logits.cpu(), temperature=temperature, top_p=top_p)
            # append tid to input_ids for next step
            tid_tensor = torch.tensor([[tid]], device=device)
            input_ids = torch.cat([input_ids, tid_tensor], dim=1)
            all_sampled.append(tid)
            all_logps.append(logp)
    return all_sampled, all_logps

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    print("[ENV] HUGGINGFACE_HUB_TOKEN present?", bool(token), flush=True)
    loaded = False
    for name in CANDIDATES:
        tok, model, device = try_load(name, token)
        if tok is None or model is None:
            print(f"[FALLBACK] Could not fully load {name} — trying next candidate.", flush=True)
            continue

        print(f"\n=== USING MODEL: {name} on device {device} ===\n", flush=True)
        prompt = "Translate to Hindi: Hello, how are you?"
        print("[PROMPT]", prompt, flush=True)
        t0 = time.time()
        ids, logps = autoregressive_sample_block(tok, model, device, prompt, block_size=BLOCK_SIZE)
        elapsed = time.time() - t0
        decoded = tok.decode(ids, skip_special_tokens=True)
        print("\n[RESULT] Sampled token ids:", ids, flush=True)
        print("[RESULT] q_logprobs:", [round(x,6) for x in logps], flush=True)
        print("[RESULT] Decoded text (concatenated):", decoded, flush=True)
        print(f"[TIMING] Took {elapsed:.3f}s to sample {len(ids)} tokens.", flush=True)
        loaded = True
        break

    if not loaded:
        print("[ERROR] Unable to load any drafter model. Exiting.", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
