# Agent vs Single-Turn RAG

## Summary

| Metric | Agent | Single-Turn RAG | Delta |
|---|---|---|---|
| cases | 103 | 103 | +0.0000 |
| citation_accuracy | 1.0 | 1.0 | +0.0000 |
| latency_p50_ms | 8.0 | 2.6 | +5.4000 |
| latency_p95_ms | 10.2 | 3.1 | +7.1000 |
| mean_answer_coverage | 0.8042 | 0.6521 | +0.1521 |
| multi_hop_synthesis_rate | 1.0 | 0.0 | +1.0000 |
| passed_critique_rate | 1.0 | 1.0 | +0.0000 |
| tool_success_rate | 1.0 | 1.0 | +0.0000 |

## Interpretation

- Baseline: one retrieval call with a single top-k pass.
- Agent: multiple entity-targeted retrieval calls plus critic reruns.
- The agent trades latency for broader, failure-recoverable evidence.
- Re-run after enabling real LLM and embedding models to get production numbers.