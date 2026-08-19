# Eval Report

## Summary

| Metric | Value |
|---|---|
| cases | 10 |
| mean_answer_coverage | 0.85 |
| multi_hop_synthesis_rate | 0.5 |
| citation_accuracy | 0.8 |
| tool_success_rate | 1.0 |
| passed_critique_rate | 0.8 |
| latency_p50_ms | 20992.4 |
| latency_p95_ms | 70259.5 |

## Per-case Results

| ID | Type | Passed | Coverage | Citation | Tools | p50 ms |
|---|---|---|---|---|---|---|
| sg-01 | single-hop | True | 1.0 | 1.0 | 1.0 | 22135 |
| sg-02 | single-hop | True | 1.0 | 1.0 | 1.0 | 34792 |
| sg-03 | single-hop | True | 1.0 | 1.0 | 1.0 | 17310 |
| sg-04 | single-hop | True | 1.0 | 1.0 | 1.0 | 19601 |
| sg-05 | multi-hop | False | 1.0 | 0.0 | 1.0 | 70259 |
| sg-06 | single-hop | True | 1.0 | 1.0 | 1.0 | 14898 |
| sg-07 | single-hop | True | 1.0 | 1.0 | 1.0 | 20074 |
| sg-08 | single-hop | True | 1.0 | 1.0 | 1.0 | 21911 |
| sg-09 | single-hop | True | 0.5 | 1.0 | 1.0 | 15566 |
| sg-10 | multi-hop | False | 0.0 | 0.0 | 1.0 | 62736 |

## Notes

- The seed corpus is deterministic and offline; real API/embedding runs can be enabled via .env.