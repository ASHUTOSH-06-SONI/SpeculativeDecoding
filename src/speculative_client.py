"""
Speculative decoding client:
- Uses a local drafter model to propose blocks.
- Calls verifier endpoint for p_logprobs (expects JSON {context: str, proposals: [ids]}).
- Performs sequential acceptance test using M (work in log-space).
- Logs timings, acceptance rate, tokens/sec.

Usage:
  export HUGGINGFACE_HUB_TOKEN=hf_xxx...
  python -u speculative_client.py --verifier_url http://<VERIFIER_HOST>:8000/verify

  
  PS- Its kinda obvious that I can't write this entire code on my own :(
"""

import os, time, math, random, argparse, json
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_DRAFTER = "google/gemma-3-270m"   # will fallback if gated access fails (script tries)
VERIFIER_URL = None  # set via CLI

def load_local_drafter(candidate=DEFAULT_DRAFTER):
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        print("[client] loading drafter tokenizer & model:", candidate)
        tok = AutoTokenizer.from_pretrained(candidate, use_fast=True, **({"use_auth_token": token} if token else {}))
        model = AutoModelForCausalLM.from_pretrained(candidate, **({"use_auth_token": token} if token else {}))
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        print(f"[client] drafter loaded on {device}, vocab={getattr(tok,'vocab_size',None)}")
        return tok, model, device
    except Exception as e:
        print(f"[client] failed to load {candidate}: {e}")
        # fallback to gpt2
        print("[client] falling back to gpt2 (public)")
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        device = "mps" if torch.backends.mps.is_available() else ("cpu")
        model = model.to(device)
        model.eval()
        print(f"[client] loaded gpt2 on {device}")
        return tok, model, device

def drafter_propose_block(tokenizer, model, device, context_text, block_size, top_p, temperature):
    # We will sample autoregressively from the drafter (like drafter_force_generate)
    enc = tokenizer(context_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    proposals = []
    q_logps = []
    for i in range(block_size):
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1, :].cpu()
        # temperature + top_p sampling
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            cutoff = torch.searchsorted(cumsum, top_p).item()
            keep = sorted_idx[: cutoff+1]
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[keep] = False
            filtered = probs.clone()
            filtered[mask] = 0.0
            if filtered.sum() == 0:
                filtered = probs
            probs = filtered / filtered.sum()
        token_id = int(torch.multinomial(probs, 1).item())
        logp = float(torch.log(probs[token_id] + 1e-45))
        proposals.append(token_id)
        q_logps.append(logp)
        # append to context for next step
        input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=device)], dim=1)
    return proposals, q_logps

def call_verifier(verifier_url, context_text, proposals, timeout=30):
    payload = {"context": context_text, "proposals": proposals}
    r = requests.post(verifier_url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data.get("p_logprobs", [])

def speculative_loop(args):
    # setup
    tok, model, device = load_local_drafter()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # initial context: for MT use source text; here use a demo prompt or dataset sample
    context_text = args.prompt
    total_tokens_generated = 0
    total_proposals = 0
    total_accepted = 0
    times = []
    start_global = time.time()

    # we will run N_steps iterations (blocks)
    for step in range(args.num_blocks):
        t_block_start = time.time()
        # 1) drafter proposes a block and gives q_logprobs
        proposals, q_logps = drafter_propose_block(tok, model, device, context_text, args.block_size, args.top_p, args.temperature)
        total_proposals += len(proposals)
        t_after_draft = time.time()

        if args.dry_run:
            # if dry_run, don't call verifier; pretend p_logprobs = q_logps (forced accept)
            p_logps = q_logps
        else:
            # 2) call verifier server for p_logprobs sequentially computed by verifier
            p_logps = call_verifier(args.verifier_url, context_text, proposals, timeout=args.timeout)

        # 3) acceptance tests (sequential)
        accepted_in_block = 0
        accepted_tokens = []
        for i, token_id in enumerate(proposals):
            p = p_logps[i]
            q = q_logps[i]
            # compute alpha in log-space: alpha = min(1, exp(p - q)/M)
            # log_alpha = min(0, (p - q) - log(M))
            log_ratio = p - q
            log_M = math.log(args.M)
            log_alpha = min(0.0, log_ratio - log_M)
            # accept with probability exp(log_alpha)
            accept_prob = math.exp(log_alpha)
            u = random.random()
            accepted = (u < accept_prob)
            if accepted:
                accepted_in_block += 1
                accepted_tokens.append(token_id)
                # update context_text by decoding token id and appending
                # decoding single id: tok.decode([token_id]) returns text fragment
                context_text += tok.decode([token_id], skip_special_tokens=True)
                total_accepted += 1
            else:
                # on first rejection we stop accepting further tokens in the block (paper's sequential scheme)
                break

        # update counters and timing
        total_tokens_generated += accepted_in_block
        t_block_end = time.time()
        times.append(t_block_end - t_block_start)

        # logs
        print(f"[block {step+1}/{args.num_blocks}] proposed={len(proposals)}, accepted_in_block={accepted_in_block}, cumulative_accepted={total_accepted}", flush=True)
        print(f"  draft_time={t_after_draft - t_block_start:.3f}s, block_time={t_block_end - t_block_start:.3f}s, accept_rate_block={accepted_in_block/len(proposals):.3f}", flush=True)
        if args.verbose:
            for i, tokid in enumerate(proposals):
                p = p_logps[i]
                q = q_logps[i]
                print(f"    proposal[{i}] id={tokid} q={q:.4f} p={p:.4f} log_ratio={p-q:.4f}", flush=True)

        # early stop if EOS or reaching token budget
        if total_tokens_generated >= args.max_tokens:
            print("[client] reached max_tokens budget; stopping.", flush=True)
            break

    elapsed = time.time() - start_global
    print("\n=== Summary ===", flush=True)
    print(f"Total wall time: {elapsed:.3f}s", flush=True)
    print(f"Total accepted tokens: {total_accepted}", flush=True)
    print(f"Total proposals: {total_proposals}", flush=True)
    print(f"Overall acceptance rate: {total_accepted/total_proposals if total_proposals>0 else 0:.3f}", flush=True)
    print(f"Avg block time: {sum(times)/len(times) if times else 0:.4f}s", flush=True)
    print(f"Tokens/sec (accepted): {total_accepted/elapsed:.3f}", flush=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier_url", type=str, default=None, help="Verifier endpoint e.g. http://host:8000/verify")
    p.add_argument("--prompt", type=str, default="Translate to Hindi: Hello, how are you?", help="Initial context/prompt text")
    p.add_argument("--block_size", type=int, default=4)
    p.add_argument("--M", type=float, default=10.0, help="Conservative bound M (>= sup_x p/q). Tune on validation.")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num_blocks", type=int, default=100, help="Number of drafter blocks to run")
    p.add_argument("--max_tokens", type=int, default=1024, help="Stop after this many accepted tokens")
    p.add_argument("--timeout", type=int, default=60, help="HTTP timeout for verifier calls (s)")
    p.add_argument("--dry_run", action="store_true", help="If set, skip calling verifier and accept all proposals (for testing)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.verifier_url is None and not args.dry_run:
        print("No verifier_url provided and not in dry_run mode. Provide --verifier_url or use --dry_run to test locally.")
    else:
        speculative_loop(args)
 
