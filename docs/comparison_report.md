# Agent vs Single-Turn RAG

## Summary

| Metric | Agent | Single-Turn RAG | Delta |
|---|---|---|---|
| cases | 103 | 103 | +0.0000 |
| citation_accuracy | 1.0 | 1.0 | +0.0000 |
| latency_p50_ms | 8.4 | 2.7 | +5.7000 |
| latency_p95_ms | 11.5 | 3.6 | +7.9000 |
| mean_answer_coverage | 0.7087 | 0.6521 | +0.0566 |
| multi_hop_synthesis_rate | 1.0 | 0.0 | +1.0000 |
| passed_critique_rate | 1.0 | 1.0 | +0.0000 |
| tool_success_rate | 1.0 | 1.0 | +0.0000 |

## Interpretation

- Baseline: one retrieval call with a single top-k pass.
- Agent: multiple entity-targeted retrieval calls plus critic reruns.
- The agent trades latency for broader, failure-recoverable evidence.
- Re-run after enabling real LLM and embedding models to get production numbers.