"""
Taxonomy definitions and archetype metadata for Enterprise Project Nodes.
"""

from typing import Dict, List, NamedTuple


class SubDomainMeta(NamedTuple):
    name: str
    epics: List[str]
    tech_stacks: List[List[str]]
    problem_archetypes: List[str]
    solution_archetypes: List[str]
    target_audiences: List[str]
    tags: List[str]


DOMAIN_TAXONOMY: Dict[str, List[SubDomainMeta]] = {
    "Developer Productivity & CI/CD": [
        SubDomainMeta(
            name="Build Caching & Artifact Distribution",
            epics=[
                "Distributed Bazel Cache",
                "P2P Artifact Propagation Engine",
                "Incremental Compiler Daemons",
                "Wasm Build Toolchain Acceleration",
            ],
            tech_stacks=[
                ["Rust", "gRPC", "AWS S3", "Bazel", "Redis"],
                ["Go", "RocksDB", "HTTP/3", "WireGuard", "MinIO"],
                ["C++", "gRPC", "NVMe Cache", "Docker", "Zstandard"],
                ["Rust", "Tokio", "Turborepo", "S3", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Local build times exceed {time} minutes for monorepo services, causing high dev friction and blocking CI pipelines.",
                "Global dev teams experience heavy latency when pulling multi-gigabyte build artifacts across multi-region VPCs.",
                "Compilers redundantly recompute identical intermediate object targets across parallel PR builds.",
                "Docker image layer downloads bottleneck ephemeral runner startup time by {percent}% on cold boots.",
            ],
            solution_archetypes=[
                "Deploy a distributed gRPC caching proxy backed by tiered NVMe scratch storage and S3 object backing.",
                "Implement a BitTorrent-inspired P2P layer caching protocol across build runner clusters.",
                "Build a persistent compilation daemon with fine-grained symbol dependency hashing and incremental linking.",
                "Construct a content-addressable chunked artifact registry with deduplication and on-demand streaming.",
            ],
            target_audiences=[
                "All backend and platform software engineers (~600 devs)",
                "Full-stack and monorepo engineers (~1,200 devs)",
                "Mobile and frontend developers (~400 devs)",
                "Core infra and build engineering teams",
            ],
            tags=["infra", "bazel", "caching", "dev-prod", "ci-cd", "rust", "grpc"],
        ),
        SubDomainMeta(
            name="Hermetic Dev Environments",
            epics=[
                "Cloud Workspaces On-Demand",
                "Ephemeral Dev Preview Clusters",
                "Containerized MicroVM Sandbox",
                "Config-as-Code Devbox",
            ],
            tech_stacks=[
                ["Go", "Kubernetes", "Firecracker MicroVMs", "eBPF", "gRPC"],
                ["TypeScript", "Node.js", "Docker", "Terraform", "WireGuard"],
                ["Rust", "Linux Namespaces", "Cgroups v2", "QEMU", "Tailscale"],
                ["Python", "FastAPI", "K3s", "Vault", "React"],
            ],
            problem_archetypes=[
                "New hire onboarding takes up to 4 days to configure local database dependencies, SDKs, and secret keys.",
                "Discrepancies between local macOS workstations and Linux production environments cause frequent 'works on my machine' defects.",
                "Developers cannot spin up isolated, multi-service preview environments for testing integration changes before merge.",
                "Workstation resource constraints prevent running full 20+ microservice topologies simultaneously.",
            ],
            solution_archetypes=[
                "Provide instant, cloud-hosted remote development environments powered by lightweight Firecracker MicroVMs with hot-standby pools.",
                "Create a declarative CLI and controller that provisions lightweight Kubernetes namespaces with mocked external dependencies in < 30 seconds.",
                "Implement a containerized sandboxing engine with copy-on-write snapshotting and automatic hibernation.",
                "Build an automated dev environment orchestrator with pre-warmed database seeds and branch-level ingress routing.",
            ],
            target_audiences=[
                "All engineering new hires and distributed remote engineers",
                "Core product development teams",
                "QA and integration test engineers",
            ],
            tags=["dev-prod", "cloud-dev", "firecracker", "k8s", "sandbox"],
        ),
        SubDomainMeta(
            name="Test Selection & Flakiness",
            epics=[
                "Predictive Test Runner",
                "Flaky Test Auto-Quarantine",
                "Differential Mutation Testing",
                "Dynamic Code Coverage Tracer",
            ],
            tech_stacks=[
                ["Python", "PyTorch", "ClickHouse", "FastAPI", "Kafka"],
                ["Go", "PostgreSQL", "OpenTelemetry", "Docker", "Redis"],
                ["Rust", "LLVM", "RocksDB", "gRPC", "Prometheus"],
                ["TypeScript", "Jest", "SQLite", "GitHub Actions", "Node.js"],
            ],
            problem_archetypes=[
                "Full test suite takes 75 minutes to execute on every commit, delaying release cadence and increasing compute expenses.",
                "Intermittent non-deterministic test failures (flaky tests) cause 30% of CI builds to fail spuriously.",
                "Engineers lack visibility into which test cases actually exercise modified code paths in large PRs.",
                "Long regression suites block urgent hotfix deployments during production incidents.",
            ],
            solution_archetypes=[
                "Train an ML-based test selection model that predicts failure probability and runs only relevant test subsets for code diffs.",
                "Deploy an automatic flakiness detection engine using Bayesian inference to quarantine unreliable tests and file Jira tickets.",
                "Implement compiler-level call-graph tracing to map every unit and integration test directly to specific source code AST nodes.",
                "Create a prioritized test executor that runs historically high-failure tests first to provide sub-2-minute feedback.",
            ],
            target_audiences=[
                "CI/CD release engineers and all PR contributors",
                "QA automation teams",
                "Platform reliability leads",
            ],
            tags=["ci-cd", "testing", "ml-selection", "flakiness", "dev-prod"],
        ),
        SubDomainMeta(
            name="Release Automation & Trunk-Based CI",
            epics=[
                "Automated Canary Analysis",
                "Trunk Merge Train Coordinator",
                "Automated Rollback Safeguard",
                "Feature Flag Synchronizer",
            ],
            tech_stacks=[
                ["Go", "Kubernetes", "Prometheus", "Envoy", "ArgoCD"],
                ["Rust", "Kafka", "PostgreSQL", "gRPC", "OpenPolicyAgent"],
                ["Python", "FastAPI", "Datadog API", "GitHub API", "Redis"],
                ["Go", "Spinnaker", "Istio", "NATS", "Elasticsearch"],
            ],
            problem_archetypes=[
                "High-velocity monorepos suffer merge conflicts and broken main branches due to race conditions between concurrent approvals.",
                "Manual production canary evaluation is error-prone, delaying progressive delivery by several hours.",
                "Failed deployments require manual engineering intervention, leading to extended MTTR during regressions.",
                "Feature flags lack automated lifecycle tracking, accumulating stale flags that trigger unexpected edge-case interactions.",
            ],
            solution_archetypes=[
                "Build an asynchronous merge train queue that speculatively compiles and tests grouped PR batches against HEAD.",
                "Deploy an automated canary analyzer comparing real-time Prometheus error rates and latency quantiles against control baselines.",
                "Implement automated circuit-breaker rollbacks driven by eBPF network metrics and application exception spikes.",
                "Create a centralized feature flag sync daemon with automated PR generation for dead flag cleanup.",
            ],
            target_audiences=[
                "Service owners and DevOps release managers",
                "On-call site reliability engineers",
                "Monorepo core contributors",
            ],
            tags=["ci-cd", "canary", "merge-train", "argo", "reliability"],
        ),
    ],
    "Data Infrastructure & Analytics": [
        SubDomainMeta(
            name="Real-time Stream Processing",
            epics=[
                "Low-Latency Event Router",
                "Streaming State Store Optimization",
                "Exactly-Once Transaction Coordinator",
                "Dynamic Stream Filter & Aggregator",
            ],
            tech_stacks=[
                ["Rust", "Apache Kafka", "RocksDB", "gRPC", "Apache Arrow"],
                ["Java", "Apache Flink", "Apache Iceberg", "Prometheus", "Kubernetes"],
                ["Go", "Redpanda", "DuckDB", "NATS JetStream", "Grafana"],
                ["C++", "Apache Pulsar", "LevelDB", "gRPC", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Kafka consumer groups experience high GC pause spikes and lag accumulation when processing 500k+ events/sec during peak traffic.",
                "Stateful stream aggregations exhaust memory during multi-day tumbling window calculations.",
                "Downstream analytic pipelines receive duplicate financial records during transient node failovers.",
                "Event schemas drift silently across producers, breaking downstream real-time streaming jobs.",
            ],
            solution_archetypes=[
                "Implement a zero-copy streaming pipeline using Rust and Apache Arrow with SIMD-vectorized filtering.",
                "Deploy an optimized state management backend replacing RocksDB with NVMe-backed io_uring local storage.",
                "Build a two-phase commit exactly-once sink coordinator bridging Kafka streaming with Apache Iceberg tables.",
                "Deploy a real-time schema registry validation proxy rejecting non-conforming Protobuf and Avro payloads at the gateway.",
            ],
            target_audiences=[
                "Data engineering teams and streaming analytics consumers",
                "Financial transaction and ledger teams",
                "Real-time recommendation and fraud detection pipelines",
            ],
            tags=["streaming", "kafka", "flink", "arrow", "realtime", "data-infra"],
        ),
        SubDomainMeta(
            name="Data Lakehouse & Compaction",
            epics=[
                "Automated Iceberg Table Optimizer",
                "Tiered Parquet Storage Manager",
                "Columnar Data Compaction Worker",
                "Zero-Copy Clone Lake Engine",
            ],
            tech_stacks=[
                ["Rust", "Apache Iceberg", "Apache Arrow", "AWS S3", "DataFusion"],
                ["Python", "PyArrow", "DuckDB", "Ray", "MinIO"],
                ["Java", "Apache Spark", "Delta Lake", "Trino", "Prometheus"],
                ["Go", "ClickHouse", "Ceph", "gRPC", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Small file problem in S3 lakehouse creates millions of 2MB Parquet files, degrading query latency by 10x.",
                "High storage costs incurred by retaining uncompressed, historical raw events across hot S3 tiers.",
                "Concurrent writes to Delta/Iceberg tables fail with optimistic concurrency conflict errors during high-ingestion periods.",
                "Auditors require point-in-time time-travel queries across petabyte-scale table partitions with tight SLA.",
            ],
            solution_archetypes=[
                "Develop an autonomous compaction daemon using DataFusion to bin-pack small Parquet files into optimal 512MB columnar chunks.",
                "Implement an automated lifecycle tiered storage manager migrating cold partitions to S3 Glacier with Zstd dictionary compression.",
                "Build a conflict-free partition write coordinator using distributed leasing and commit grouping.",
                "Deploy an accelerated metadata index caching Iceberg snapshot manifests in high-speed Redis / NVMe caches.",
            ],
            target_audiences=[
                "Data platform engineers and BI analysts (~300 users)",
                "Analytics infrastructure stakeholders",
                "Data science and quantitative modeling teams",
            ],
            tags=["lakehouse", "iceberg", "parquet", "storage", "arrow", "data-infra"],
        ),
        SubDomainMeta(
            name="Distributed Query Acceleration",
            epics=[
                "Distributed Caching Query Proxy",
                "Cost-Based Adaptive Query Planner",
                "Vectorized Expression Evaluator",
                "Materialized View Precomputation Engine",
            ],
            tech_stacks=[
                ["Rust", "DataFusion", "Apache Arrow", "gRPC", "Redis"],
                ["C++", "DuckDB", "Trino", "LLVM", "RocksDB"],
                ["Java", "Apache Calcite", "Presto", "HDFS", "Prometheus"],
                ["Go", "ClickHouse", "Apache Arrow Flight", "Kubernetes", "Grafana"],
            ],
            problem_archetypes=[
                "Interactive BI dashboard queries exceed 30 seconds on multi-billion row event tables.",
                "Ad-hoc analytical queries consume entire Trino cluster capacity, starving production ETL pipelines.",
                "Duplicate sub-queries across distinct user dashboards recompute identical joins and aggregations continuously.",
                "Large shuffle exchanges during distributed joins overwhelm cluster top-of-rack network switches.",
            ],
            solution_archetypes=[
                "Implement an Arrow Flight SQL caching proxy that serves sub-millisecond repeated queries from shared memory.",
                "Build an adaptive cost-based query optimizer that dynamically selects between hash joins and broadcast joins using runtime row counts.",
                "Create an automatic materialized view engine identifying common sub-tree expressions and updating incremental aggregates.",
                "Implement SIMD-accelerated filtering and late materialization for columnar scans over object stores.",
            ],
            target_audiences=[
                "Business intelligence and product analytics teams",
                "Executive reporting and dashboard users",
                "Core data infrastructure engineers",
            ],
            tags=["query-engine", "duckdb", "arrow-flight", "datafusion", "trino"],
        ),
        SubDomainMeta(
            name="Metadata & Schema Cataloging",
            epics=[
                "Unified Enterprise Data Catalog",
                "Automated Column-Level Lineage",
                "Data Quality Assertion Framework",
                "Semantic Metric Definition Layer",
            ],
            tech_stacks=[
                ["Python", "FastAPI", "Neo4j", "Elasticsearch", "PostgreSQL"],
                ["TypeScript", "React", "GraphQL", "Apache Kafka", "OpenMetadata"],
                ["Go", "gRPC", "CockroachDB", "Redis", "OpenLineage"],
                ["Rust", "SQLite", "Tantivy", "FastAPI", "Prometheus"],
            ],
            problem_archetypes=[
                "Data scientists and engineers waste hours finding authoritative datasets due to fragmented silos and duplicate tables.",
                "Upstream schema modifications silently break downstream dashboards without advance notification.",
                "Lack of column-level data lineage impairs regulatory GDPR/CCPA compliance audits.",
                "Different business units compute conflicting KPI metrics due to inconsistent SQL definitions.",
            ],
            solution_archetypes=[
                "Construct an enterprise data catalog extracting schemas, descriptions, and usage statistics via automated scrapers.",
                "Implement SQL AST parser to automatically extract and visualize end-to-end column-level lineage across all DAGs.",
                "Deploy an automated data quality framework running statistical assertions (nulls, ranges, uniqueness) on ingestion.",
                "Establish a centralized metric store (Cube/dbt semantic layer) standardizing business metric computations across all BI tools.",
            ],
            target_audiences=[
                "All data analysts, data scientists, and governance officers (~500 users)",
                "Compliance and risk management teams",
            ],
            tags=["catalog", "metadata", "lineage", "data-quality", "governance"],
        ),
    ],
    "AI/ML Platform & Inference": [
        SubDomainMeta(
            name="High-Throughput Model Serving",
            epics=[
                "Dynamic Continuous Batching Engine",
                "Speculative Decoding Accelerator",
                "Multi-Tenant GPU Memory Virtualizer",
                "Quantized Kernel Optimization",
            ],
            tech_stacks=[
                ["C++", "CUDA", "vLLM", "Triton Inference Server", "gRPC"],
                ["Rust", "Candle", "CUDA", "Tokio", "Prometheus"],
                ["Python", "PyTorch", "TensorRT-LLM", "FastAPI", "Ray Serve"],
                ["C++", "Metal", "ONNX Runtime", "gRPC", "Docker"],
            ],
            problem_archetypes=[
                "LLM inference costs are ballooning due to low GPU utilization and fixed request batching.",
                "Time-to-first-token (TTFT) and inter-token latency exceed SLAs for user-facing conversational AI features.",
                "Multi-model deployments fail due to GPU VRAM fragmentation and out-of-memory errors on concurrent requests.",
                "Cold model weight swapping takes over 45 seconds per GPU instance when serving multiple specialized LoRA adapters.",
            ],
            solution_archetypes=[
                "Deploy an optimized inference server with PagedAttention and dynamic continuous batching to 4x throughput.",
                "Implement speculative decoding with small draft models reducing latency by 45% for high-parameter LLMs.",
                "Create a multi-tenant VRAM manager dynamically scheduling LoRA adapter weights in GPU high-bandwidth memory.",
                "Implement custom INT4/FP8 AWQ tensor cores kernels maximizing FLOPs utilization on modern GPUs.",
            ],
            target_audiences=[
                "AI application engineers and product teams integrating LLMs",
                "Core ML platform and infrastructure teams",
                "End-users interacting with real-time AI agents",
            ],
            tags=["ml-platform", "inference", "vllm", "cuda", "gpu", "llm"],
        ),
        SubDomainMeta(
            name="Low-Latency Feature Store",
            epics=[
                "Real-Time Feature Ingestion Engine",
                "Point-in-Time Accurate Training Joiner",
                "Sub-Millisecond Feature Cache",
                "Automated Feature Drift Monitor",
            ],
            tech_stacks=[
                ["Go", "Redis Cluster", "Feast", "Apache Kafka", "gRPC"],
                ["Rust", "RocksDB", "Apache Arrow", "gRPC", "PostgreSQL"],
                ["Python", "DuckDB", "Ray", "ClickHouse", "Prometheus"],
                ["C++", "Aerospike", "Kafka", "Protobuf", "Grafana"],
            ],
            problem_archetypes=[
                "Online recommendation models encounter > 20ms feature lookup latency, breaching user request budgets.",
                "Training-serving skew causes model accuracy degradation in production due to inconsistent offline vs online feature transforms.",
                "Joining historical feature values for training datasets requires hours of slow Spark batch jobs.",
                "Feature distributions drift silently over time without alerting data science teams.",
            ],
            solution_archetypes=[
                "Deploy a distributed dual-storage feature store: low-latency in-memory cache for online lookups and columnar lake for offline training.",
                "Implement unified feature transformation pipelines in Rust compiled to both streaming engines and offline batch jobs.",
                "Create an optimized point-in-time ASOF join engine in DuckDB/Arrow executing training dataset joins in minutes.",
                "Implement automated Kolmogorov-Smirnov and PSI drift detection workers checking production feature streams.",
            ],
            target_audiences=[
                "ML engineers building ranking and recommendation models",
                "Fraud detection and risk modeling teams",
                "Data science research groups",
            ],
            tags=["feature-store", "ml-platform", "low-latency", "redis", "feast"],
        ),
        SubDomainMeta(
            name="Distributed Training Orchestration",
            epics=[
                "Elastic Gang Scheduler for PyTorch",
                "Zero-Bubble Pipeline Parallelism",
                "Checkpoint Resumption Accelerator",
                "Automated Fault-Tolerant Worker Recovery",
            ],
            tech_stacks=[
                ["Python", "PyTorch FSDP", "Kubernetes", "KubeFlow", "Slurm"],
                ["Rust", "RDMA", "InfiniBand", "gRPC", "Etcd"],
                ["C++", "NCCL", "DeepSpeed", "CUDA", "Prometheus"],
                ["Go", "Kubernetes Operator", "AWS EFA", "Docker", "Grafana"],
            ],
            problem_archetypes=[
                "Multi-node GPU training jobs crash when a single worker encounters transient hardware or network dropouts.",
                "Saving multi-gigabyte model checkpoints every hour stalls training computation for 15 minutes per epoch.",
                "Fragmented GPU clusters leave unassigned GPUs idle due to lack of gang scheduling for distributed jobs.",
                "Inter-node gradient all-reduce communication saturates cluster switches, causing poor multi-node scaling efficiency.",
            ],
            solution_archetypes=[
                "Deploy an elastic gang scheduler with automatic spot-instance preemption recovery and instant warm checkpoint restores.",
                "Implement asynchronous non-blocking checkpoint streaming directly to distributed NVMe/S3 pools via RDMA.",
                "Optimize NCCL communication topologies and overlap gradient synchronization with backward pass computations.",
                "Build an automated health-check daemon isolating failing GPU nodes before scheduling large 128-GPU runs.",
            ],
            target_audiences=[
                "Foundation model training teams",
                "Computer vision and NLP researchers",
                "Compute infrastructure platform leads",
            ],
            tags=["training", "pytorch", "gpu", "fsdp", "distributed", "ml-platform"],
        ),
        SubDomainMeta(
            name="LLM Evals & Hallucination Guardrails",
            epics=[
                "Deterministic Semantic Guardrail Proxy",
                "Automated Multi-Turn LLM Benchmark Suite",
                "Synthetic Evaluation Dataset Generator",
                "Latent Space Uncertainty Detector",
            ],
            tech_stacks=[
                ["Python", "FastAPI", "Pydantic", "OpenAI API", "Redis"],
                ["Rust", "HuggingFace Tokenizers", "ONNX", "gRPC", "PostgreSQL"],
                ["TypeScript", "Next.js", "TailwindCSS", "ClickHouse", "Docker"],
                ["Go", "Triton", "OpenTelemetry", "Prometheus", "Kafka"],
            ],
            problem_archetypes=[
                "Customer-facing LLM agents occasionally generate hallucinations or reveal internal prompt instructions.",
                "Engineering teams lack systematic offline benchmark datasets to evaluate whether model prompt revisions improve or degrade accuracy.",
                "Validating compliance and safety filters introduces unacceptable 500ms latency overheads to user queries.",
                "Difficulty in quantifying multi-turn conversation quality and agent tool invocation correctness.",
            ],
            solution_archetypes=[
                "Deploy a high-speed Rust guardrail proxy evaluating regex, token probabilities, and embedding cosine similarity in < 5ms.",
                "Construct an automated continuous evaluation harness running LLM-as-a-judge tests across 2,000 synthetic test scenarios on every PR.",
                "Implement an epistemic uncertainty estimator analyzing output token entropy to trigger fallback human review.",
                "Build an interactive observability dashboard tracking prompt regressions, hallucination rates, and task completion metrics.",
            ],
            target_audiences=[
                "GenAI product teams and prompt engineers",
                "Trust & Safety officers",
                "Customer support automation leads",
            ],
            tags=["llm-evals", "guardrails", "genai", "ml-platform", "benchmarking"],
        ),
    ],
    "Security & Zero-Trust Architecture": [
        SubDomainMeta(
            name="Identity Federation & SPIFFE/SPIRE",
            epics=[
                "Workload Identity Attestation Agent",
                "Zero-Trust Mutual TLS Mesh Interop",
                "Short-Lived Ephemeral Token Broker",
                "Cross-Cloud Federated IAM Bridge",
            ],
            tech_stacks=[
                ["Go", "SPIRE / SPIFFE", "Kubernetes", "gRPC", "Vault"],
                ["Rust", "OpenSSL", "eBPF", "Envoy", "Prometheus"],
                ["C++", "Tls", "Linux Security Modules", "Docker", "WireGuard"],
                ["Python", "FastAPI", "OAuth2 / OIDC", "PostgreSQL", "Redis"],
            ],
            problem_archetypes=[
                "Microservices authenticate using long-lived static API keys stored in environment variables, creating severe credential leak risks.",
                "Cross-cloud communications between AWS and GCP require complex IAM role synchronizations with high management overhead.",
                "Lack of cryptographic workload provenance allows compromised containers to impersonate sensitive billing services.",
                "Manual certificate rotations cause intermittent outages when internal mTLS certificates expire unexpectedly.",
            ],
            solution_archetypes=[
                "Deploy SPIRE server and node agents issuing short-lived SVID X.509 certificates with automated 1-hour renewals.",
                "Integrate workload identity attestation with Envoy sidecars to enforce cryptographically verified mTLS authorization policies.",
                "Build an ephemeral token broker exchanging SPIFFE identities for scoped cloud provider STS access tokens.",
                "Implement automated certificate lifecycle management with continuous telemetry alerting on approaching expirations.",
            ],
            target_audiences=[
                "Enterprise security architecture and SecOps teams",
                "Microservice backend engineering squads",
                "Infrastructure compliance auditors",
            ],
            tags=["security", "spiffe", "spire", "zero-trust", "mtls", "iam"],
        ),
        SubDomainMeta(
            name="Automated Secret Rotation & Vaulting",
            epics=[
                "Dynamic Database Credential Generator",
                "Encrypted Envelope Key Custody Service",
                "Zero-Knowledge Developer Secrets CLI",
                "Automated Secret Leak Scanner in CI",
            ],
            tech_stacks=[
                ["Go", "HashiCorp Vault", "Kubernetes Operator", "PostgreSQL", "AWS KMS"],
                ["Rust", "Sodium / Ring", "SQLite", "gRPC", "OpenTelemetry"],
                ["Python", "GitPython", "Tree-sitter", "GitHub App", "FastAPI"],
                ["TypeScript", "Node.js", "WebCrypto", "Tailscale", "Redis"],
            ],
            problem_archetypes=[
                "Engineers accidentally commit production credentials or API tokens into git repositories.",
                "Database passwords remain unchanged for months because rotating credentials requires coordinated service restarts.",
                "Developer workstations store unencrypted production staging keys in local `.env` files.",
                "Key management service KMS costs explode due to un-cached granular envelope decryption requests.",
            ],
            solution_archetypes=[
                "Deploy a dynamic secret vault generating short-lived, single-use database credentials with automatic lease revocations.",
                "Implement a pre-commit and CI Git secret scanning engine with entropy analysis and AST parsing to block leaked tokens in PRs.",
                "Build a zero-knowledge CLI injecting secrets into container memory at runtime without writing keys to disk.",
                "Design a local envelope encryption cache with authenticated memory pinning and cryptographic key shredding.",
            ],
            target_audiences=[
                "All software developers and DevOps teams",
                "Database administrators and Security Operations",
                "Compliance officers",
            ],
            tags=["secrets", "vault", "security", "kms", "devops"],
        ),
        SubDomainMeta(
            name="Supply Chain & SBOM Attestation",
            epics=[
                "Cryptographic Artifact Signing (Sigstore/Cosign)",
                "Automated Software Bill of Materials Generator",
                "Transitive Dependency Vulnerability Interceptor",
                "Hermetic Binary Provenance Attestor",
            ],
            tech_stacks=[
                ["Go", "Cosign / Sigstore", "SLSA Framework", "Tekton", "Rekor"],
                ["Rust", "Syft / Grype", "SQLite", "Docker", "gRPC"],
                ["Python", "FastAPI", "PostgreSQL", "Kafka", "GitHub Actions"],
                ["TypeScript", "Node.js", "GraphQL", "Elasticsearch", "Docker"],
            ],
            problem_archetypes=[
                "Production Kubernetes clusters lack verification of whether running container images actually originated from trusted CI pipelines.",
                "Open-source packages in dependency trees introduce zero-day vulnerabilities (e.g. log4j style) that take weeks to discover across 1,000 repos.",
                "Auditors require cryptographic proof that binaries were built hermetically without intermediate tampering.",
                "Security teams struggle to inventory which microservices contain vulnerable libraries across legacy deployments.",
            ],
            solution_archetypes=[
                "Integrate Sigstore keyless signing and Rekor transparency logs into all CI pipelines, blocking unsigned images via admission controllers.",
                "Deploy automated SBOM generation (SPDX/CycloneDX) for every release artifact with instant graph vulnerability querying.",
                "Implement an automated dependency PR bot with regression test verification to auto-patch vulnerable packages within 24 hours.",
                "Enforce SLSA Level 3 build provenance attestation across all production container registries.",
            ],
            target_audiences=[
                "DevSecOps and application security leads",
                "Compliance and vendor-risk auditors",
                "Release engineering teams",
            ],
            tags=["sbom", "sigstore", "supply-chain", "security", "slsa"],
        ),
        SubDomainMeta(
            name="Runtime Intrusion Detection & eBPF",
            epics=[
                "eBPF Kernel Anomaly Sensor",
                "Zero-Overhead Network Packet Inspector",
                "Automated Container Threat Isolation",
                "Syscall Profiling & Behavioral Baseline",
            ],
            tech_stacks=[
                ["C", "Rust (Aya)", "eBPF", "Linux Kernel", "gRPC"],
                ["Go", "Tetragon / Cilium", "Kubernetes", "Prometheus", "Kafka"],
                ["C++", "Falco", "Fluentbit", "Elasticsearch", "Grafana"],
                ["Python", "FastAPI", "ClickHouse", "PostgreSQL", "Grafana"],
            ],
            problem_archetypes=[
                "Traditional host security agents introduce 15% CPU overhead and crash user workloads during kernel driver updates.",
                "Zero-day reverse-shell exploits go undetected in production containers until data exfiltration has occurred.",
                "Security teams lack visibility into raw process executions, socket openings, and file writes within ephemeral pods.",
                "Alert fatigue caused by thousands of noisy, un-correlated security log events per day.",
            ],
            solution_archetypes=[
                "Deploy an ultra-lightweight eBPF sensor capturing raw syscalls (execve, connect, openat) with < 1% CPU overhead.",
                "Build a behavioral baselining engine that automatically whitelists standard container process trees and alerts on anomalous execution.",
                "Implement automated container network isolation killing compromised pods immediately upon detecting unauthorized outbound socket connections.",
                "Correlate low-level kernel events with high-level Kubernetes metadata to provide actionable incident forensics.",
            ],
            target_audiences=[
                "Security Operations Center (SOC) and Incident Response teams",
                "Cloud infrastructure and Kubernetes platform engineers",
            ],
            tags=["ebpf", "runtime-security", "intrusion-detection", "linux", "cilium"],
        ),
    ],
    "Core Backend & Microservices": [
        SubDomainMeta(
            name="High-Performance RPC & Service Mesh",
            epics=[
                "Zero-Copy Protobuf Serializer",
                "Adaptive Envoy Sidecar Proxy",
                "Connection Pooling & Multiplexing Engine",
                "Cross-Cluster Service Mesh Router",
            ],
            tech_stacks=[
                ["Rust", "gRPC", "Tokio", "Tonic", "FlatBuffers"],
                ["C++", "Envoy", "Istio", "xDS API", "Prometheus"],
                ["Go", "Connect-RPC", "HTTP/2", "NATS", "OpenTelemetry"],
                ["Java", "Netty", "gRPC-Java", "Micrometer", "Kubernetes"],
            ],
            problem_archetypes=[
                "High JSON serialization overhead accounts for 35% of CPU time in inter-service RPC communications.",
                "Envoy sidecars introduce 4ms p99 latency overhead and substantial memory bloat across 5,000 pod replicas.",
                "Sudden traffic surges cause connection storms that exhaust backend service file descriptors and TCP sockets.",
                "Cross-region microservice calls suffer cascading timeouts when intermediate network links experience packet loss.",
            ],
            solution_archetypes=[
                "Migrate high-volume RPC endpoints to zero-copy Protobuf/FlatBuffers serialization with shared memory buffers.",
                "Deploy ambient service mesh architectures using node-level eBPF routing to eliminate per-pod sidecar overheads.",
                "Implement intelligent client-side load balancing with outlier detection, exponential backoff, and jittered retries.",
                "Build dynamic circuit-breakers that proactively drop non-critical secondary requests during upstream latency spikes.",
            ],
            target_audiences=[
                "Backend platform engineers and microservice owners (~800 devs)",
                "Network infrastructure and reliability leads",
            ],
            tags=["grpc", "rpc", "service-mesh", "envoy", "backend", "performance"],
        ),
        SubDomainMeta(
            name="Event-Driven Messaging & Brokerage",
            epics=[
                "High-Throughput Distributed Message Broker",
                "Ordered Event Partition Balancer",
                "Dead Letter Queue Management & Auto-Replay",
                "Schema-Validated Event Publisher",
            ],
            tech_stacks=[
                ["Rust", "Kafka Protocol", "io_uring", "Raft", "RocksDB"],
                ["Go", "NATS JetStream", "gRPC", "Prometheus", "PostgreSQL"],
                ["Java", "Apache Pulsar", "BookKeeper", "ZooKeeper", "Grafana"],
                ["Python", "Celery", "RabbitMQ", "Redis", "FastAPI"],
            ],
            problem_archetypes=[
                "Message brokers crash under backpressure when consumer services experience prolonged database lock contention.",
                "Unbalanced partition keys cause hot partitions where a single consumer instance processes 80% of cluster event volume.",
                "Poison-pill messages block consumer queues, requiring manual manual database intervention to skip invalid payloads.",
                "Lack of distributed trace propagation across async message boundaries obscures end-to-end transaction latency.",
            ],
            solution_archetypes=[
                "Build a modern high-performance message broker using io_uring and Raft consensus achieving sub-millisecond p99 latencies.",
                "Implement adaptive consistent hashing that dynamically splits hot partition keys into virtual sub-partitions.",
                "Deploy an automated dead-letter queue orchestrator with error classification and scheduled replay policies.",
                "Inject OpenTelemetry W3C trace context headers into all message envelopes to guarantee continuous distributed tracing.",
            ],
            target_audiences=[
                "Distributed systems engineers and asynchronous workflow owners",
                "Payments, billing, and order processing teams",
            ],
            tags=["messaging", "kafka", "nats", "event-driven", "backend"],
        ),
        SubDomainMeta(
            name="Distributed Transaction Sagas",
            epics=[
                "Asynchronous Saga Orchestrator",
                "Two-Phase Commit Compensating Action Engine",
                "Idempotency Key Guard & Storage",
                "Distributed Lock Manager with Fencing Tokens",
            ],
            tech_stacks=[
                ["Go", "Temporal.io", "PostgreSQL", "gRPC", "Redis"],
                ["Rust", "Tokio", "CockroachDB", "Kafka", "Prometheus"],
                ["Java", "Spring Boot", "Cadence", "MySQL", "OpenTelemetry"],
                ["TypeScript", "Node.js", "Redis", "BullMQ", "Fastify"],
            ],
            problem_archetypes=[
                "Multi-step checkout workflows leave user accounts in inconsistent states when third-party payment gateways fail intermittently.",
                "Race conditions during concurrent balance deductions lead to double-spending vulnerabilities.",
                "Custom saga state machines written in application code lack durability, losing progress when worker nodes restart.",
                "Distributed locks without fencing tokens result in split-brain data corruption during GC pauses.",
            ],
            solution_archetypes=[
                "Adopt Temporal durable execution workflows to guarantee deterministic multi-step saga execution and automated rollbacks.",
                "Deploy an enterprise idempotency gateway storing cryptographic request hashes with atomic database transactions.",
                "Implement distributed locking with monotonically increasing fencing tokens validated at the database storage layer.",
                "Create a centralized workflow dashboard tracking compensation metrics and un-recovered transaction anomalies.",
            ],
            target_audiences=[
                "Financial transaction, commerce, and billing engineering teams",
                "Core backend architecture teams",
            ],
            tags=["saga", "distributed-systems", "temporal", "idempotency", "transactions"],
        ),
        SubDomainMeta(
            name="Adaptive Rate Limiting & Shedding",
            epics=[
                "Global Token Bucket Cluster",
                "Predictive Concurrency Limit Controller",
                "Prioritized Load Shedding Gateway",
                "Client Tier Quota Enforcer",
            ],
            tech_stacks=[
                ["Go", "Redis", "Envoy", "gRPC", "Prometheus"],
                ["Rust", "Tokio", "eBPF", "RocksDB", "Grafana"],
                ["C++", "Varnish", "Nginx", "Lua", "OpenTelemetry"],
                ["Python", "FastAPI", "Redis", "PostgreSQL", "Docker"],
            ],
            problem_archetypes=[
                "Scraper bots and runaway internal batch jobs overwhelm public APIs, causing denial of service for mobile end-users.",
                "Traditional static rate limits either reject legitimate spiky traffic or fail to protect backends from cascading brownouts.",
                "Microservices collapse simultaneously when traffic increases by 200% during seasonal flash sales.",
                "Tier 1 enterprise customers experience throttling due to noisy neighbor tenants on shared infrastructure.",
            ],
            solution_archetypes=[
                "Deploy a distributed token-bucket rate limiter with local sliding-window caching and Redis synchronization.",
                "Implement gradient-based adaptive concurrency limits (Vegas algorithm) that throttle requests when p90 response latency rises.",
                "Build a load-shedding gateway prioritizing authenticated revenue-generating traffic and dropping telemetry requests during overload.",
                "Create a multi-tier customer quota manager with bursting allowances and real-time usage webhook alerts.",
            ],
            target_audiences=[
                "API gateway teams and site reliability engineers",
                "Customer integration and partner engineering teams",
            ],
            tags=["rate-limiting", "load-shedding", "resilience", "gateway", "traffic"],
        ),
    ],
    "Cloud Infrastructure & Reliability": [
        SubDomainMeta(
            name="Kubernetes Fleet Management",
            epics=[
                "Multi-Cluster GitOps Synchronizer",
                "Node Auto-Provisioner (Karpenter)",
                "Cluster Upgrade Automation Controller",
                "Resource Oversubscription & Rightsizing Engine",
            ],
            tech_stacks=[
                ["Go", "Kubernetes Operator SDK", "ArgoCD", "Helm", "Terraform"],
                ["Go", "Karpenter", "AWS EKS", "Prometheus", "Grafana"],
                ["Rust", "Kube-rs", "Etcd", "gRPC", "OpenTelemetry"],
                ["Python", "FastAPI", "Kubernetes API", "PostgreSQL", "Docker"],
            ],
            problem_archetypes=[
                "Managing 40+ independent Kubernetes clusters across regions causes configuration drift and security patch delays.",
                "Cluster autoscaling via standard cluster-autoscaler takes 5+ minutes to provision new worker nodes during traffic spikes.",
                "Engineering teams over-provision CPU and memory requests by 300%, wasting millions in monthly cloud infrastructure spend.",
                "In-place Kubernetes control plane and node pool upgrades frequently cause downtime for misconfigured stateful workloads.",
            ],
            solution_archetypes=[
                "Deploy unified multi-cluster GitOps orchestration with ArgoCD ApplicationSets and automated drift remediation.",
                "Migrate compute scaling to Karpenter for sub-45-second direct EC2 node provisioning tailored to pod requirements.",
                "Implement a dynamic pod rightsizing controller (VPA + machine learning) that automatically adjusts requests based on 30-day usage.",
                "Construct an automated progressive cluster upgrade pipeline with automated workload drainage and validation checks.",
            ],
            target_audiences=[
                "Core cloud infrastructure and platform engineering teams",
                "FinOps and cloud budget managers",
                "All service owners deploying on Kubernetes",
            ],
            tags=["kubernetes", "k8s", "gitops", "karpenter", "cloud-infra", "finops"],
        ),
        SubDomainMeta(
            name="Multi-Region Active-Active Failover",
            epics=[
                "Anycast Global Traffic Director",
                "Cross-Region State Synchronization Engine",
                "Automated Region Evacuation Drill Coordinator",
                "Global Consistency Conflict Resolver",
            ],
            tech_stacks=[
                ["Go", "Cloudflare Workers", "Route53", "CockroachDB", "Envoy"],
                ["Rust", "Kafka MirrorMaker 2", "AWS DynamoDB Global Tables", "gRPC", "Prometheus"],
                ["C++", "BGP Anycast", "Spanner", "NATS", "Grafana"],
                ["Python", "FastAPI", "Terraform", "Datadog API", "Docker"],
            ],
            problem_archetypes=[
                "Cloud provider availability zone and regional outages cause prolonged downtime for single-region applications.",
                "Cross-region database replication lag introduces stale read anomalies for international users.",
                "Executing emergency traffic failovers requires manual DNS edits, taking over 45 minutes to propagate globally.",
                "Concurrent writes to different geographical regions create split-brain data conflicts during network partitions.",
            ],
            solution_archetypes=[
                "Deploy an Anycast global routing layer dynamically shifting traffic based on real-time health and latency telemetry.",
                "Implement an active-active architecture using globally distributed databases (Spanner/CockroachDB) with Raft consensus.",
                "Build an automated region evacuation framework capable of draining an entire cloud region in < 2 minutes with zero data loss.",
                "Design Conflict-Free Replicated Data Types (CRDTs) for shared global shopping carts and user profiles.",
            ],
            target_audiences=[
                "Chief Technology Officers, SRE leads, and enterprise risk management",
                "Global high-availability service teams",
            ],
            tags=["multi-region", "active-active", "high-availability", "dns", "failover"],
        ),
        SubDomainMeta(
            name="Chaos Engineering & Resilience Verification",
            epics=[
                "Automated Fault Injection Pipeline",
                "Network Latency & Jitter Simulator",
                "GameDay Experiment Orchestrator",
                "Steady-State Hypothesis Validator",
            ],
            tech_stacks=[
                ["Go", "Chaos Mesh", "Kubernetes", "Prometheus", "Slack API"],
                ["Rust", "eBPF", "Linux TC (Traffic Control)", "gRPC", "OpenTelemetry"],
                ["Python", "FastAPI", "Grafana API", "Docker", "PostgreSQL"],
                ["TypeScript", "React", "Node.js", "Argo Workflows", "Redis"],
            ],
            problem_archetypes=[
                "Production outages repeatedly expose untested failure modes (e.g., cascading timeouts, threadpool exhaustion, missing fallback logic).",
                "Disaster recovery runbooks are rarely tested and fail during real-world incidents.",
                "Engineering teams lack confidence in whether new retry configurations cause thundering herd storms.",
                "Manual chaos testing is slow and rarely integrated into automated continuous deployment pipelines.",
            ],
            solution_archetypes=[
                "Integrate continuous chaos experiments into staging pipelines, injecting pod kills, packet drops, and CPU spikes automatically.",
                "Build an eBPF network chaos injector simulating cross-datacenter latency and TLS handshake failures.",
                "Create a GameDay orchestration platform measuring service steady-state SLOs during controlled failure scenarios.",
                "Deploy an automated rollback system terminating chaos experiments instantly if error budgets are breached.",
            ],
            target_audiences=[
                "Site reliability engineers, QA teams, and resilience champions",
                "Mission-critical tier 1 backend teams",
            ],
            tags=["chaos-engineering", "resilience", "fault-injection", "sre", "reliability"],
        ),
        SubDomainMeta(
            name="Zero-Downtime Database Migration",
            epics=[
                "Online Schema Change Engine (gh-ost/pt-osc)",
                "Dual-Writing Replication Proxy",
                "Automated Data Verification & Backfill Validator",
                "Partition Archival & Compaction Coordinator",
            ],
            tech_stacks=[
                ["Go", "MySQL / PostgreSQL", "Kafka", "Debezium", "Prometheus"],
                ["Rust", "RocksDB", "gRPC", "Apache Arrow", "Grafana"],
                ["Python", "SQLAlchemy", "Alembic", "Redis", "ClickHouse"],
                ["Java", "Flyway", "Spanner", "Docker", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Altering 2TB MySQL/PostgreSQL table schemas locks production tables, causing query timeouts and application downtime.",
                "Migrating mission-critical databases from legacy on-prem instances to cloud RDS requires 8-hour maintenance windows.",
                "Data backfill scripts corrupt secondary index data due to missing foreign key and uniqueness constraints.",
                "Lack of automated dual-run verification makes verifying data consistency between old and new databases risky.",
            ],
            solution_archetypes=[
                "Implement an asynchronous online schema change runner using triggerless binlog parsing to alter schemas with 0 downtime.",
                "Deploy a dual-writing proxy with Debezium change data capture (CDC) streaming records to validate shadow databases.",
                "Build a high-speed distributed data comparison engine hashing rows in parallel to detect data discrepancies before cutover.",
                "Automate zero-downtime DNS and connection string cutovers with instant rollback capabilities.",
            ],
            target_audiences=[
                "Database administrators and data platform engineers",
                "Backend product engineers managing large relational datasets",
            ],
            tags=["database", "migration", "zero-downtime", "cdc", "postgres", "mysql"],
        ),
    ],
    "Frontend & Design Systems": [
        SubDomainMeta(
            name="Micro-Frontend Module Federation",
            epics=[
                "Dynamic Module Federation Runtime",
                "Shared Dependency Version Arbiter",
                "Isolated Sandbox Styling Enforcer",
                "Micro-Frontend Orchestrator Shell",
            ],
            tech_stacks=[
                ["TypeScript", "Webpack Module Federation", "React", "Node.js", "Vite"],
                ["Rust", "SWC", "Wasm", "TypeScript", "ESBuild"],
                ["JavaScript", "Single-SPA", "TailwindCSS", "Docker", "S3"],
                ["TypeScript", "Next.js", "Web Components", "GraphQL", "Playwright"],
            ],
            problem_archetypes=[
                "Independent frontend feature teams block each other during monolithic web application deployments.",
                "Different micro-apps bundle duplicate copies of React and large libraries, bloating page bundle sizes to > 8MB.",
                "CSS class name collisions between micro-frontends cause severe visual bugs in shared shell layouts.",
                "Uncaught runtime exceptions in one sub-app crash the entire web application interface.",
            ],
            solution_archetypes=[
                "Implement Module Federation with dynamic remote module loading and automatic shared singleton dependency negotiation.",
                "Build an isolated micro-frontend container shell with shadow DOM boundaries and CSS-in-JS scoping.",
                "Deploy a shared dependency arbiter in the build toolchain alerting on semver incompatibilities before merge.",
                "Wrap all micro-frontend mount points in React error boundaries with graceful fallback UI placeholders.",
            ],
            target_audiences=[
                "Frontend engineering teams (~350 devs) across 15 business units",
                "Web platform and core experience engineers",
            ],
            tags=["frontend", "module-federation", "micro-frontend", "react", "webpack"],
        ),
        SubDomainMeta(
            name="Design System Tokens & Accessibility",
            epics=[
                "Multi-Platform Design Token Pipeline",
                "Automated WCAG 2.1 AAA Accessibility Auditor",
                "Headless Accessible Component Library",
                "Theme Switching & High-Contrast Mode Generator",
            ],
            tech_stacks=[
                ["TypeScript", "Figma REST API", "Style Dictionary", "React", "Axe-core"],
                ["Rust", "Tree-sitter", "PostCSS", "Node.js", "GitHub Actions"],
                ["TypeScript", "Radix UI", "TailwindCSS", "Storybook", "Jest"],
                ["Dart", "Flutter", "SwiftUI", "Jetpack Compose", "Kotlin"],
            ],
            problem_archetypes=[
                "Design tokens defined in Figma drift from production CSS/iOS/Android codebases due to manual copy-pasting.",
                "Web application fails accessibility compliance audits, exposing the organization to legal and usability risks.",
                "Custom UI components lack proper ARIA labels, focus traps, and keyboard navigation support.",
                "Supporting dark mode and high-contrast accessibility themes requires extensive custom CSS overrides.",
            ],
            solution_archetypes=[
                "Build an automated design token pipeline syncing Figma token variables directly to TypeScript, CSS, Swift, and Kotlin.",
                "Integrate Axe-core automated accessibility audits into CI, failing PRs on color contrast or ARIA attribute violations.",
                "Develop a headless, fully accessible component primitive library adhering strictly to WAI-ARIA authoring guidelines.",
                "Implement dynamic CSS custom property theming supporting instant theme changes and user OS preferences.",
            ],
            target_audiences=[
                "UI/UX designers, accessibility specialists, and frontend engineers",
                "All end-users including users with visual or motor disabilities",
            ],
            tags=["design-system", "accessibility", "a11y", "figma", "tokens", "frontend"],
        ),
        SubDomainMeta(
            name="Real-time Collaborative Canvas & CRDTs",
            epics=[
                "Conflict-Free Collaborative Document Engine",
                "High-Performance WebGL Canvas Renderer",
                "Low-Latency Presence & Cursor Synchronizer",
                "Offline-First Local Storage Sync Manager",
            ],
            tech_stacks=[
                ["Rust", "Wasm", "Yjs / Automerge", "TypeScript", "WebSockets"],
                ["TypeScript", "WebGL", "PixiJS", "Node.js", "Redis"],
                ["Go", "WebRTC", "Protobuf", "Pion", "PostgreSQL"],
                ["C++", "WebGPU", "Dawn", "IndexedDB", "FastAPI"],
            ],
            problem_archetypes=[
                "Simultaneous edits by multiple users cause data loss and overwrite conflicts in collaborative document editors.",
                "Rendering 50,000 interactive canvas shapes in the browser causes severe frame drops (< 15 FPS) and sluggish interaction.",
                "Real-time cursor positions and live presence indicators lag by over 300ms, creating a disjointed multiplayer experience.",
                "Users working offline lose their edits when reconnecting to the internet due to simplistic last-write-wins synchronization.",
            ],
            solution_archetypes=[
                "Implement a high-speed CRDT document model in Rust compiled to WebAssembly for conflict-free multi-user synchronization.",
                "Build a hardware-accelerated WebGL/WebGPU 2D rendering pipeline maintaining steady 60 FPS with 100k+ vector objects.",
                "Deploy an ephemeral WebRTC / WebSocket broadcast service synchronizing live cursors with < 30ms latency.",
                "Create an offline-first IndexedDB persistence layer that automatically merges changes upon network restoration.",
            ],
            target_audiences=[
                "End-users of collaborative design, whiteboard, and document tools",
                "Rich web application frontend developers",
            ],
            tags=["crdt", "realtime", "wasm", "webgl", "collaboration", "frontend"],
        ),
        SubDomainMeta(
            name="Edge SSR & Streaming Hydration",
            epics=[
                "Edge Worker SSR Renderer",
                "Selective Progressive Hydration Engine",
                "Smart Asset Preloading & Speculation Rules",
                "Edge HTML Streaming & Sub-Resource Integrity",
            ],
            tech_stacks=[
                ["TypeScript", "Cloudflare Workers", "React 19", "V8 Isolates", "Rust"],
                ["Go", "Fastly Compute@Edge", "Wasm", "HTML Rewriter", "Redis"],
                ["JavaScript", "Deno", "Fresh", "Preact", "TailwindCSS"],
                ["TypeScript", "Next.js App Router", "Node.js", "Vercel", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Largest Contentful Paint (LCP) and Time to Interactive (TTI) are degraded by slow monolithic server-side rendering in distant origins.",
                "Full-page JavaScript hydration blocks the main thread for 1.5 seconds on mobile devices with low-power CPUs.",
                "E-commerce product detail pages experience high bounce rates due to delayed hero image and pricing rendering.",
                "Cold starts on serverless edge functions cause sporadic 800ms latency spikes for international users.",
            ],
            solution_archetypes=[
                "Deploy React 19 streaming server components onto globally distributed Cloudflare V8 Workers close to users.",
                "Implement selective progressive hydration that hydrates interactive components only when they scroll into the viewport.",
                "Inject speculative pre-rendering and asset prefetch headers dynamically based on ML click-prediction models.",
                "Utilize edge HTML streaming to deliver the first byte and initial visual shell in < 50ms.",
            ],
            target_audiences=[
                "E-commerce, content, and high-traffic consumer web visitors",
                "Frontend performance engineers and SEO leads",
            ],
            tags=["ssr", "edge", "hydration", "performance", "cwv", "frontend"],
        ),
    ],
    "Search & Discovery": [
        SubDomainMeta(
            name="Hybrid Sparse-Dense Retrieval",
            epics=[
                "Unified BM25 + Vector Fusion Ranker",
                "Real-Time Dense Vector Index Sharder",
                "Learned Sparse Representation Inverted Index",
                "Multi-Modal Query Embedding Search",
            ],
            tech_stacks=[
                ["Rust", "HNSW / Qdrant", "Tantivy", "gRPC", "PyTorch"],
                ["C++", "Faiss", "Lucene", "RocksDB", "gRPC"],
                ["Go", "Milvus", "Elasticsearch", "Apache Arrow", "Prometheus"],
                ["Python", "FastAPI", "ONNX Runtime", "ClickHouse", "Redis"],
            ],
            problem_archetypes=[
                "Traditional keyword search fails on semantic queries with synonym variations, while pure dense search misses exact SKU and part numbers.",
                "Dense vector indexing RAM requirements exceed infrastructure budgets when scaling to 100M+ documents.",
                "Adding new documents to the vector index requires long re-indexing pauses that delay real-time catalog updates.",
                "Reciprocal Rank Fusion (RRF) score normalization yields suboptimal relevance ranking across heterogeneous document types.",
            ],
            solution_archetypes=[
                "Construct a hybrid search engine combining Tantivy BM25 sparse scoring with HNSW dense vector retrieval via learned rank fusion.",
                "Implement scalar and product quantization (PQ/SQ8) in Rust, reducing vector index memory footprint by 75% with < 2% recall loss.",
                "Deploy a real-time memory-mapped vector segment builder allowing continuous instant index updates without search interruption.",
                "Integrate cross-encoder re-ranking for the top 100 retrieved candidates to maximize NDCG@10.",
            ],
            target_audiences=[
                "E-commerce searchers, internal knowledge base users, and enterprise discovery teams",
                "Search relevance engineers and data retrieval scientists",
            ],
            tags=["search", "vector-search", "bm25", "hybrid-search", "hnsw", "tantivy"],
        ),
        SubDomainMeta(
            name="Semantic Re-ranking & Cross-Encoders",
            epics=[
                "Ultra-Low Latency Cross-Encoder Serving",
                "Transformer Distillation for Edge Re-Ranking",
                "Personalized Real-Time Feature Re-Scorer",
                "Diversity & Fair Ranking Post-Processor",
            ],
            tech_stacks=[
                ["C++", "TensorRT", "ONNX Runtime", "CUDA", "gRPC"],
                ["Rust", "Candle", "FastAPI", "Redis", "Prometheus"],
                ["Python", "PyTorch", "FlashAttention-2", "FastAPI", "Ray"],
                ["Go", "Triton Inference Server", "PostgreSQL", "Kafka", "Grafana"],
            ],
            problem_archetypes=[
                "Cross-encoder re-ranking introduces 80ms latency, making it impossible to evaluate more than 20 candidate documents within SLA.",
                "Search results suffer from redundancy where the top 5 items are nearly identical variations of the same product.",
                "Re-ranking models fail to incorporate real-time user session context and recent click feedback.",
                "High model GPU memory footprint prevents deploying dedicated re-rankers across regional edge locations.",
            ],
            solution_archetypes=[
                "Deploy an optimized FlashAttention-2 TensorRT cross-encoder pipeline evaluating 100 candidates in < 8ms.",
                "Implement Maximal Marginal Relevance (MMR) and determinantal point processes (DPP) to diversify top-ranked results.",
                "Incorporate real-time user session embeddings into the re-scoring stage via online vector dot-products.",
                "Distill large cross-encoders into compact 4-layer MiniLM models with INT8 quantization for CPU deployment.",
            ],
            target_audiences=[
                "Search product leads, ranking algorithm engineers, and end-users",
            ],
            tags=["reranking", "cross-encoder", "relevance", "search", "ml-platform"],
        ),
        SubDomainMeta(
            name="Entity Resolution & Knowledge Graph",
            epics=[
                "Graph Neural Network Entity Linker",
                "Distributed Graph Traversal Engine",
                "Semantic Triplet Extraction Pipeline",
                "Real-Time Entity Disambiguation Service",
            ],
            tech_stacks=[
                ["Rust", "Neo4j / Memgraph", "Apache Arrow", "gRPC", "PyTorch"],
                ["Go", "Dgraph", "GraphQL", "Redis", "Elasticsearch"],
                ["Python", "PyG (PyTorch Geometric)", "Spacy", "FastAPI", "PostgreSQL"],
                ["Java", "Apache TinkerPop", "JanusGraph", "Cassandra", "Prometheus"],
            ],
            problem_archetypes=[
                "Entities (people, companies, products) are duplicated across databases under varying names, aliases, and typos.",
                "Multi-hop knowledge graph queries exceed database timeouts when traversing high-degree celebrity nodes.",
                "Unstructured text documents contain valuable relationships that are not indexed in relational tables.",
                "Entity linking models misclassify ambiguous acronyms and polysemous terms without deep context.",
            ],
            solution_archetypes=[
                "Train a Graph Neural Network (GNN) entity resolution model mapping heterogeneous aliases into unified canonical entity IDs.",
                "Deploy a high-performance in-memory graph traversal engine with bounded depth-first search and parallel edge expansion.",
                "Build an automated LLM triplet extraction pipeline generating structured subject-predicate-object knowledge graphs from documentation.",
                "Implement contextual disambiguation leveraging surrounding document embeddings and co-occurrence statistics.",
            ],
            target_audiences=[
                "Enterprise knowledge graph teams, intelligence analysts, and data discovery users",
            ],
            tags=["knowledge-graph", "entity-resolution", "gnn", "graph", "search"],
        ),
        SubDomainMeta(
            name="Query Intent Parsing & Expansion",
            epics=[
                "Real-Time Query Intent Classifier",
                "Zero-Shot Acronym & Synonym Expander",
                "Typo Correction & Auto-Suggest Engine",
                "Semantic Query Rewriter for Complex Filters",
            ],
            tech_stacks=[
                ["Rust", "SymSpell", "Tantivy", "HuggingFace Tokenizers", "gRPC"],
                ["Go", "FastAPI", "Redis", "Elasticsearch", "Prometheus"],
                ["Python", "ONNX Runtime", "FastAPI", "PostgreSQL", "Docker"],
                ["C++", "Trie / FST", "LevelDB", "gRPC", "OpenTelemetry"],
            ],
            problem_archetypes=[
                "Search users enter misspelled or colloquial queries resulting in zero search results ('zero-results' rate > 12%).",
                "Engineers use internal project codenames and jargon that standard search tokenizers fail to understand.",
                "Queries with implicit attribute filters (e.g. 'high priority bugs last week') fail on naive keyword matching.",
                "Auto-suggest dropdowns exhibit noticeable typing lag (> 150ms), degrading search UX.",
            ],
            solution_archetypes=[
                "Build a sub-millisecond SymSpell typo correction and finite-state transducer (FST) auto-suggest engine in Rust.",
                "Deploy a lightweight transformer model classifying query intent into transactional, informational, or navigational modes.",
                "Implement an LLM-assisted offline synonym and acronym graph automatically expanding enterprise queries.",
                "Construct a semantic query parser converting natural language requests into structured SQL/Elasticsearch filter clauses.",
            ],
            target_audiences=[
                "All search end-users and internal employees using enterprise search",
            ],
            tags=["query-understanding", "nlp", "search", "auto-suggest", "symspell"],
        ),
    ],
}
