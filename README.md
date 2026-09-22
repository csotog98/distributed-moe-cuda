# Distributed MoE inference engine

Small C++20/CUDA scaffold for a sharded Mixture-of-Experts block. The CPU path is always buildable and provides deterministic token routing. When `nvcc`, the CUDA toolkit, and NCCL are installed, CMake adds the asynchronous NCCL transport in `src/cuda_backend.cu`.

## Build on Arch Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/moe_demo
```

The current machine can build the CPU path. CUDA/NCCL support requires:

- NVIDIA driver and a working `nvidia-smi`.
- CUDA toolkit (`nvcc`, `cuda` package or the NVIDIA-supported equivalent).
- NCCL headers and library (`nccl.h`, `libnccl.so`), normally from the `nccl` package or NVIDIA's repository.

On Arch Linux, install the missing communication library with:

```bash
sudo pacman -S nccl
```

The default CMake configuration targets compute capability 7.5, which matches an RTX 2060. Remove `build/` after changing CUDA installations so CMake does not retain an old compiler path.

The transport deliberately uses paired `ncclSend`/`ncclRecv` calls inside one `ncclGroupStart`/`ncclGroupEnd` region. A production deployment still needs one NCCL communicator per process/rank, expert GEMM kernels, pinned host staging buffers, and a CUDA-event pipeline around each microbatch; those are the next integration layer rather than silently being simulated by the CPU demo.
