#pragma once

#include <cstddef>

namespace moe {

struct CudaTransportConfig {
    int rank{};
    int total_gpus{1};
    std::size_t hidden_size{};
    void* nccl_communicator{};
};

class PinnedHostBuffer {
public:
    explicit PinnedHostBuffer(std::size_t bytes);
    ~PinnedHostBuffer();

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    void* data() noexcept;
    const void* data() const noexcept;
    std::size_t size() const noexcept;

private:
    struct Impl;
    Impl* impl_{};
};

class CudaNcclTransport {
public:
    explicit CudaNcclTransport(CudaTransportConfig config);
    ~CudaNcclTransport();

    CudaNcclTransport(const CudaNcclTransport&) = delete;
    CudaNcclTransport& operator=(const CudaNcclTransport&) = delete;

    void exchange_async(const float* send_buffer,
                        std::size_t send_size,
                        const int* send_counts,
                        std::size_t send_counts_size,
                        const int* receive_counts,
                        std::size_t receive_counts_size,
                        float* receive_buffer,
                        std::size_t receive_size);
    void synchronize();

private:
    struct Impl;
    Impl* impl_{};
};

}  // namespace moe
