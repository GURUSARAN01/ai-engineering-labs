# VectorBench Results

## Configuration

- Documents: 5,000
- Queries: 12
- Embedding model: all-MiniLM-L6-v2
- Embedding dimension: 384
- Distance metric: cosine
- Top-k: 10
- HNSW ef: 40

## Results

| Backend | Recall@10 | Mean latency | p50 | p95 |
|---|---:|---:|---:|---:|
| pgvector HNSW | 1.000 | 1.926 ms | 1.730 ms | 3.043 ms |
| Qdrant HNSW | 1.000 | 13.640 ms | 14.479 ms | 27.250 ms |

## Interpretation

Both systems achieved perfect Recall@10 for this workload.

pgvector showed substantially lower application-observed latency in this
small local benchmark.

This result should not be generalized to all workloads. The benchmark uses
only 5,000 vectors and includes client/database communication overhead.
Performance may change with larger datasets, concurrency, filtering,
different protocols, hardware, and index configuration.

For applications already using PostgreSQL, pgvector can provide vector
retrieval without introducing another infrastructure service.

Qdrant remains useful when vector search requires dedicated infrastructure,
independent scaling, or specialized vector-search capabilities.