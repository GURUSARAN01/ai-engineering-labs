# Day 7 — HNSW Benchmark

Dataset: 5,000 vectors
Embedding dimension: 384
Metric: Cosine distance
Top-k: 10

## Results

| Search | Latency | Recall@10 |
|---|---:|---:|
| Exact | 6.44 ms | 1.00 |
| HNSW ef=10 | 3.42 ms | 0.70 |
| HNSW ef=40 | 2.65 ms | 1.00 |
| HNSW ef=100 | 2.90 ms | 1.00 |

## Observations

- Exact search provides the ground-truth nearest neighbors.
- HNSW trades exhaustive search for faster approximate retrieval.
- Low ef_search can reduce recall.
- Increasing ef_search explores more candidates and can improve recall.
- A single latency measurement is noisy; production benchmarking requires repeated queries and percentile statistics.
- EXPLAIN ANALYZE confirmed PostgreSQL used the HNSW index.