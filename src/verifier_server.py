"""
Verifier server for speculative decoding.

Loads a target model (GPT-2 Large by default) and exposes a /verify endpoint.
The client (speculative_client.py) sends proposed token IDs + context text,
and this server returns the log-probabilities under the target model's distribution.

Usage:
    pip install fastapi uvicorn transformers torch
    python verifier_server.py
    # or with a different model:
    python verifier_server.py --model_name gpt2-medium --port 8000

The /verify endpoint expects:
    POST /verify
    { "context": "some text", "proposals": [1234, 5678, ...] }

And returns:
    { "p_logprobs": [-0.123, -0.456, ...] }

One p_logprob per proposed token — computed sequentially from the target model
(each token's logprob is conditioned on context + all previously proposed tokens).
"""

import os
import time
import argparse
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()

# ── globals (populated at startup) ──────────────────────────────────────────
app = FastAPI(title="Speculative Decoding Verifier")
tokenizer = None
model = None
device = None


# ── request / response schemas ───────────────────────────────────────────────
class VerifyRequest(BaseModel):
    context: str           # plain-text context the drafter used
    proposals: list[int]   # token IDs proposed by the drafter


class VerifyResponse(BaseModel):
    p_logprobs: list[float]   # one log-prob per proposal under target model


# ── core logic ────────────────────────────────────────────────────────────────
def compute_p_logprobs(
    context_text: str,
    proposals: list[int],
) -> list[float]:
    """
    For each proposal token, compute log P(token | context + previously accepted proposals)
    under the target model.

    This mirrors exactly what the drafter does in drafter_local.py —
    sequential forward passes, one per proposed token.
    """
    enc = tokenizer(context_text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)  # (1, seq_len)

    p_logprobs = []

    with torch.no_grad():
        for token_id in proposals:
            # forward pass: get logits for the *next* token position
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]   # (vocab_size,)

            # convert to log-probs
            log_probs = torch.log_softmax(logits, dim=-1)  # numerically stable

            # look up log-prob for this specific proposed token
            lp = float(log_probs[token_id].item())
            p_logprobs.append(lp)

            # extend context with this token for the next iteration
            tid_tensor = torch.tensor([[token_id]], device=device)
            input_ids = torch.cat([input_ids, tid_tensor], dim=1)

    return p_logprobs


# ── endpoint ──────────────────────────────────────────────────────────────────
@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest):
    if not req.proposals:
        raise HTTPException(status_code=400, detail="proposals list is empty")

    t0 = time.time()
    try:
        p_logprobs = compute_p_logprobs(req.context, req.proposals)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - t0
    print(
        f"[verifier] verified {len(req.proposals)} tokens in {elapsed:.3f}s "
        f"| context_len={len(req.context)} chars",
        flush=True,
    )

    return VerifyResponse(p_logprobs=p_logprobs)


@app.get("/health")
def health():
    return {"status": "ok", "model": model.config._name_or_path, "device": str(device)}


# ── startup: load model ───────────────────────────────────────────────────────
def load_model(model_name: str):
    global tokenizer, model, device

    print(f"[verifier] Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    print(f"[verifier] Loading model: {model_name} ...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # device priority: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)
    model.eval()

    elapsed = time.time() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[verifier] Model loaded on {device} in {elapsed:.1f}s | "
        f"params={params:.0f}M | vocab={tokenizer.vocab_size}",
        flush=True,
    )


# ── entrypoint ────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_name",
        type=str,
        default="gpt2-large",   # 774M — bigger than drafter's gpt2 (117M), fits on M1
        help="HuggingFace model ID to use as verifier/target model",
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_model(args.model_name)
    print(f"[verifier] Starting server on {args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)