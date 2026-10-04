# Deterministic evaluation

Evaluation is authoritative code, not an LLM judgment.

The initial result schema will cover:

- WER
- CER
- conversion/load success
- inference failures
- real-time factor
- inference latency p50/p95
- model/artifact sizes
- optional numerical comparison against a reference backend
- optional word/end-to-end latency when streaming support exists

All transcript scoring uses one versioned text-normalization policy.

An AI agent may orchestrate the evaluator, explain failures and summarize
results. It must not edit metric outputs to make a candidate appear better.

Hardware results must record the benchmark version, model artifact hashes,
repository revision and relevant runtime/firmware provenance.
