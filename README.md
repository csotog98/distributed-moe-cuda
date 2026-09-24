# Distributed MoE Runtime

A compact C++20/CUDA prototype for a distributed Mixture-of-Experts inference pipeline with explicit routing, microbatching, communication planning, expert sharding, and multi-rank execution modeling.

The project aims to model the core mechanics of a real MoE runtime before moving into a heavier production stack with NCCL, CUDA kernels, and large-scale serving infrastructure.

## Status

This software is currently in a functional prototype stage.

What is already implemented and validated:

- deterministic token routing and expert scoring
- top-k expert selection
- rank dispatch planning
- microbatch grouping
- transfer buffers and transfer batches
- destination-ordered transfer packing with stable offsets
- rank-aware distributed transfer metadata and receive counts
- CPU execution and merge over rank-aware distributed transfer plans
- host-to-device and device-to-host `TransferBatch` exchange through NCCL
- communication-plan modeling
- all-to-all exchange planning
- expert sharding model
- multi-rank execution context and merged outputs
- CUDA/NCCL communicator validation across two ranks
- real two-GPU peer-to-peer NCCL exchange validation on RTX 5060 Ti hardware

What is still to be proven in the full runtime:

- overlapped NCCL communication and expert execution across multiple ranks
- real expert-parallel execution with weight shards
- full distributed routing and expert execution pass using rank-local batches
- full performance and throughput benchmarking

This is a serious engineering prototype, not a finished production inference engine.

## Why this project exists

The goal is to explore and structure the runtime logic behind a distributed MoE system:

- how tokens are routed to experts
- how expert ownership is mapped to ranks
- how work is grouped into microbatches
- how payloads move across devices
- how execution is coordinated across distributed ranks
- how the final output is reconstructed

The code is intentionally explicit and readable rather than hidden behind heavy abstractions, so the runtime data flow is easy to inspect and extend.

## Architecture overview

The project is organized around a few core ideas:

### Token routing
A token is scored against experts and assigned to a preferred expert. The router produces a ranked list of candidates and maps them to the destination rank.

### Rank dispatch
Tokens are partitioned by the rank that should own their expert execution. This is the first step toward distributed expert parallelism.

### Microbatching
Work is grouped into small batches to match typical distributed inference patterns and make communication and execution easier to schedule.

### Transfers and buffers
Payloads are packed into transfer buffers with explicit send/receive counts and offsets so the communication layer can serialize and reconstruct the relevant hidden states.
The CUDA backend can now move a packed `TransferBatch` through device buffers and NCCL, then return the received values to host memory.
The router also builds rank-aware send and receive metadata from per-rank token batches.

### Communication planning
The system models send and receive stages and an all-to-all exchange plan. This is the abstraction that later connects to NCCL collectives.

### Sharded experts
Each expert can be modeled as a shard owned by a rank, with the associated token indices tracked explicitly.

### Multi-rank execution plan
The runtime declares the local execution state for each rank, including local token indices, packed input vectors, and local output values.

## Repository layout

```text
.
├── CMakeLists.txt
├── README.md
├── include/
│   ├── cuda_backend.hpp
│   └── token_router.hpp
├── src/
│   ├── cuda_backend.cu
│   ├── main.cpp
│   └── token_router.cpp
├── tests/
│   ├── cuda_smoke.cu
│   └── test_token_router.cpp
└── build/
```

## Build and run

### Linux / Arch

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/moe_demo
```

The demo prints both the CPU routing pipeline and a two-rank distributed transfer example, including per-rank send/receive counts and merged expert outputs.

### Requirements

For the CPU-only path:

- C++20 compiler
- CMake

For the CUDA/NCCL path:

- NVIDIA driver and `nvidia-smi`
- CUDA toolkit / `nvcc`
- NCCL headers and library

On Arch Linux, the NCCL dependency is commonly installed with:

```bash
sudo pacman -S nccl
```

## Current validation status

The project has been validated on the current machine through:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2 && ./build/moe_tests && ./build/moe_demo && ./build/moe_cuda_smoke 0 1
```

This produces a successful build and passing CPU and CUDA smoke tests. The low-level two-rank NCCL exchange has also been run on a remote host with two RTX 5060 Ti GPUs. The new high-level `TransferBatch` exchange path is currently validated locally in one-rank mode and is ready for the next two-GPU run.

## Important reality check

This project is not yet a full multi-GPU inference server.

The current state is best described as:

- a working distributed MoE runtime model
- a valid CPU baseline
- a CUDA/NCCL transport layer validated with a real two-rank peer exchange
- a strong starting point for true multi-GPU deployment

To reach the next stage, the project needs:

- full all-to-all token exchange driven by rank-aware routing metadata
- optimized expert execution and batching in CUDA

## Next milestones

The next realistic milestones are:

1. connect rank-aware batches to the multi-rank CUDA runtime
2. execute local experts on received hidden states
3. reconstruct and validate the distributed MoE output
4. benchmark overlapped communication and expert execution
5. production-oriented cleanup and scaling analysis

## License

This project is shared for research and engineering exploration.

## Repository intent

This code is meant to be readable, structured, and extensible. It is especially useful as a reference implementation for understanding the internal flow of a distributed MoE inference runtime without immediately depending on a large production framework.
