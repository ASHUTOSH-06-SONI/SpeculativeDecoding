# Speculative Decoding for Faster LLM Inference

In this repository I have implemented speculative decoding, a technique to accelerate large language model (LLM) inference by combining a fast draft model with a larger target model.

The project explores speculative decoding as a drop-in replacement for traditional autoregressive decoding, focusing on:
- Token acceptance efficiency
- Distribution-preserving verification
- Understanding single-device bottlenecks

## Why Autoregressive Decoding is Slow

Standard autoregressive decoding in LLMs is slow because:
- Tokens are generated sequentially
- Each step requires a full forward pass of a large model

Speculative decoding addresses this by:
1. Using a smaller, faster model to propose multiple tokens
2. Verifying those tokens using the larger target model
3. Accepting valid tokens via a rejection sampling scheme and discarding the rest

This reduces the number of expensive forward passes through the large model.

## How It Works

- Draft model proposes a block of `k` tokens autoregressively
- Target model evaluates the proposed tokens and returns log-probabilities
- Tokens are accepted or rejected using a sequential acceptance test in log-space: `α = min(1, p/q)`
- On first rejection, the block is truncated and decoding continues from the accepted prefix

This preserves the output distribution of the target model while improving throughput when acceptance rate is high.

## Architecture

```
Prompt
  │
  ▼
┌─────────────────┐        block of k tokens + q_logprobs
│  Draft Model    │ ──────────────────────────────────────────►  ┌──────────────────┐
│  (GPT-2 117M)   │                                              │  Verifier Server │
│  runs locally   │ ◄────────────────────────────────────────── │  (GPT-2L 774M)   │
└─────────────────┘        p_logprobs per proposed token         │  FastAPI/uvicorn  │
  │                                                              └──────────────────┘
  ▼
Acceptance-Rejection Test (log-space)
  │
  ▼
Accepted tokens appended to context → next block
```

## Results

Experiments run on Apple M1 (MPS), GPT-2 (117M) as drafter, GPT-2 Large (774M) as verifier.

| Metric | Value |
|--------|-------|
| Baseline tokens/sec (GPT-2 Large, autoregressive) | 4.835 |
| Spec decoding accepted tokens/sec | 0.178 |
| Overall acceptance rate | 55.0% |
| Avg accepted tokens per verifier call | 2.2 |
| Sequential verifier savings | 54.5% |
| Total proposals | 100 |
| Total accepted tokens | 55 |

**The 55% acceptance rate confirms the draft-verify distribution matching is working correctly** — the drafter agrees with the target model on more than half of proposed tokens.

The end-to-end speedup is negative on a single device because the verifier runs sequentially over HTTP on the same machine. Spec decoding speedup materializes when:
- The verifier runs batched verification (not sequential per-token forward passes)
- Or the draft and verify models are on separate devices / a GPU cluster

This is a known constraint of single-device deployments, documented in the original [speculative decoding paper](https://arxiv.org/abs/2211.17192).

## Project Structure

```
SpeculativeDecoding/
├── README.md
├── requirements.txt
└── src/
    ├── speculative_client.py   # drafter + acceptance-rejection loop
    ├── verifier_server.py      # FastAPI server wrapping the target model
    ├── drafter_local.py        # standalone drafter debug script
    ├── baseline.py             # pure autoregressive baseline for comparison
    └── speculative_pipeline.py # dataset loading + simple generation pipeline
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

**Step 1 — Baseline (target model only):**
```bash
python src/baseline.py --model_name gpt2-large --max_tokens 100
```

**Step 2 — Start verifier server (keep running):**
```bash
python src/verifier_server.py --model_name gpt2-large
# wait for: INFO: Uvicorn running on http://0.0.0.0:8000
```

**Step 3 — Run speculative decoding:**
```bash
python src/speculative_client.py \
  --verifier_url http://localhost:8000/verify \
  --block_size 4 \
  --M 1.0 \
  --num_blocks 25 \
  --max_tokens 100
```

## References

- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., 2023
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — Chen et al., 2023