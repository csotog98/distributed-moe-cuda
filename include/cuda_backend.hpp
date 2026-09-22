#pragma once

#include <cstddef>
#include <span>

namespace moe {

struct CudaTransportConfig {
    int rank{};
    int world_size{1};
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

    void exchange_async(std::span<const float> send_buffer,
                        std::span<const int> send_counts,
                        std::span<const int> receive_counts,
                        std::span<float> receive_buffer);
    void synchronize();

private:
    struct Impl;
    Impl* impl_{};
};

}  // namespace moe
