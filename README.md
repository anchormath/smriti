# Smriti - Adaptive Memory Engine

> **Smriti** is a hybrid retrieval engine that combines Hierarchical Associative Memory Partitioning (HAM-P) with Hopfield Associative Memory Resolution Dial (HAM-R) dynamics. It uses Rust for high-performance inference and Python for orchestration and evaluation.

## System Architecture and language boundaries
┌─────────────────────────────────────────────────────────────────────────────┐
│ Python Layer (Data Engine & Evaluation)                                     │
│   • Data Modeling & Schemas: Pydantic, Faker                                │
│   • Synthetic Generation: Gemini Flash / Ollama LLM API                     │
│   • MRL Feature Encoding: PyTorch, HuggingFace (nomic-embed-text-v1.5)      │
│   • Evaluation & Analysis: Pandas, Scipy, Matplotlib                        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ PyO3 / Maturin Bindings
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Rust Core Engine (`smriti-core`)                                            │
│   • Memory Layout: Contiguous column-major buffers (Zero-allocation query)  │
│   • Hardware Acceleration: SIMD dot products (AVX2/AVX-512/Neon)            │
│   • Parallelism: Rayon multithreaded matrix-vector products                 │
│   • HAM-P Topology: Multi-layer tree with atomic node splitting             │
│   • HAM-R Dynamics: Continuous Hopfield leaf pools with inverse temp β(R)   │
│   • Concurrency: Fine-grained RwLock with background Copy-On-Write splits   │
└─────────────────────────────────────────────────────────────────────────────┘
