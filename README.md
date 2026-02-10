# Speculative Decoding for Faster LLM Inference

In this repository I have implemented speculative decoding, a technique to accelerate large language model (LLM) inference by combining a fast draft model with a larger target model.

The project explores speculative decoding as a drop-in replacement for traditional greedy / autoregressive decoding, focusing on:
- Latency reduction
- Token acceptance efficiency
- Minimal quality degradation

Standard autoregressive decoding in LLMs is slow because:
- Tokens are generated sequentially
- Each step requires a full forward pass of a large model

Speculative decoding addresses this by:
1. Using a smaller, faster model to propose multiple tokens
2. Verifying those tokens in parallel using the larger model
3. Accepting valid tokens and rejecting incorrect ones

This significantly reduces the number of expensive forward passes.

How Speculative Decoding Works
  - Draft model proposes a block of tokens
  - Target model evaluates the proposed tokens in parallel
  - Tokens are accepted as long as probabilities match acceptance criteria
  - On rejection, decoding falls back to the target model

This preserves correctness while improving throughput.

