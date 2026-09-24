#include "cuda_backend.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace moe {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_nccl(ncclResult_t status, const char* operation) {
    if (status != ncclSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + ncclGetErrorString(status));
    }
}

}  // namespace

struct PinnedHostBuffer::Impl {
    void* data{};
    std::size_t bytes{};
};

PinnedHostBuffer::PinnedHostBuffer(std::size_t bytes) : impl_(new Impl{nullptr, bytes}) {
    if (bytes != 0) {
        check_cuda(cudaMallocHost(&impl_->data, bytes), "cudaMallocHost");
    }
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (impl_ != nullptr) {
        cudaFreeHost(impl_->data);
        delete impl_;
    }
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : impl_(std::exchange(other.impl_, nullptr)) {}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        if (impl_ != nullptr) {
            cudaFreeHost(impl_->data);
            delete impl_;
        }
        impl_ = std::exchange(other.impl_, nullptr);
    }
    return *this;
}

void* PinnedHostBuffer::data() noexcept { return impl_ == nullptr ? nullptr : impl_->data; }
const void* PinnedHostBuffer::data() const noexcept { return impl_ == nullptr ? nullptr : impl_->data; }
std::size_t PinnedHostBuffer::size() const noexcept { return impl_ == nullptr ? 0 : impl_->bytes; }

struct CudaNcclTransport::Impl {
    CudaTransportConfig config;
    ncclComm_t communicator{};
    cudaStream_t communication_stream{};
    cudaEvent_t transfer_complete{};
};

CudaNcclTransport::CudaNcclTransport(CudaTransportConfig config)
    : impl_(new Impl{config, static_cast<ncclComm_t>(config.nccl_communicator), nullptr, nullptr}) {
    if (impl_->communicator == nullptr) {
        delete impl_;
        impl_ = nullptr;
        throw std::invalid_argument("CudaNcclTransport requires an initialized NCCL communicator");
    }
    check_cuda(cudaSetDevice(config.rank), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&impl_->communication_stream, cudaStreamNonBlocking), "cudaStreamCreate");
    check_cuda(cudaEventCreateWithFlags(&impl_->transfer_complete, cudaEventDisableTiming), "cudaEventCreate");
}

CudaNcclTransport::~CudaNcclTransport() {
    if (impl_ != nullptr) {
        cudaEventDestroy(impl_->transfer_complete);
        cudaStreamDestroy(impl_->communication_stream);
        delete impl_;
    }
}

void CudaNcclTransport::exchange_async(const float* send_buffer,
                                       std::size_t send_size,
                                       const int* send_counts,
                                       std::size_t send_counts_size,
                                       const int* receive_counts,
                                       std::size_t receive_counts_size,
                                       float* receive_buffer,
                                       std::size_t receive_size) {
    exchange_async(send_buffer, send_size, send_counts, send_counts_size,
                   receive_counts, receive_counts_size,
                   nullptr, nullptr,
                   receive_buffer, receive_size);
}

void CudaNcclTransport::exchange_async(const float* send_buffer,
                                       std::size_t send_size,
                                       const int* send_counts,
                                       std::size_t send_counts_size,
                                       const int* receive_counts,
                                       std::size_t receive_counts_size,
                                       const std::size_t* send_offsets,
                                       const std::size_t* receive_offsets,
                                       float* receive_buffer,
                                       std::size_t receive_size) {
    if (send_counts_size != static_cast<std::size_t>(impl_->config.total_gpus) ||
        receive_counts_size != static_cast<std::size_t>(impl_->config.total_gpus)) {
        throw std::invalid_argument("send_counts and receive_counts must contain one count per rank");
    }

    std::size_t send_offset = 0;
    std::size_t receive_offset = 0;
    check_nccl(ncclGroupStart(), "ncclGroupStart");
    for (int peer = 0; peer < impl_->config.total_gpus; ++peer) {
        const std::size_t send_elements = static_cast<std::size_t>(send_counts[peer]);
        const std::size_t receive_elements = static_cast<std::size_t>(receive_counts[peer]);
        const std::size_t send_peer_offset = send_offsets == nullptr ? send_offset : send_offsets[peer];
        const std::size_t receive_peer_offset = receive_offsets == nullptr ? receive_offset : receive_offsets[peer];

        if (send_peer_offset + send_elements > send_size ||
            receive_peer_offset + receive_elements > receive_size) {
            ncclGroupEnd();
            throw std::invalid_argument("NCCL count exceeds a supplied buffer");
        }
        if (peer == impl_->config.rank) {
            if (send_elements != receive_elements) {
                ncclGroupEnd();
                throw std::invalid_argument("local exchange requires matching send and receive counts");
            }
            if (send_elements != 0) {
                check_cuda(cudaMemcpyAsync(receive_buffer + receive_peer_offset,
                                           send_buffer + send_peer_offset,
                                           send_elements * sizeof(float), cudaMemcpyDeviceToDevice,
                                           impl_->communication_stream), "cudaMemcpyAsync local exchange");
            }
        } else if (send_elements != 0) {
            check_nccl(ncclSend(send_buffer + send_peer_offset, send_elements, ncclFloat32, peer,
                                impl_->communicator, impl_->communication_stream), "ncclSend");
        }
        if (peer != impl_->config.rank && receive_elements != 0) {
            check_nccl(ncclRecv(receive_buffer + receive_peer_offset, receive_elements, ncclFloat32, peer,
                                impl_->communicator, impl_->communication_stream), "ncclRecv");
        }
        send_offset += send_elements;
        receive_offset += receive_elements;
    }
    check_nccl(ncclGroupEnd(), "ncclGroupEnd");
    check_cuda(cudaEventRecord(impl_->transfer_complete, impl_->communication_stream), "cudaEventRecord");
}

void CudaNcclTransport::exchange_transfer_batch(TransferBatch& batch) {
    const std::size_t expected_receive_size = [&batch] {
        std::size_t size = 0;
        for (const int count : batch.receive_counts) {
            if (count < 0) {
                throw std::invalid_argument("receive counts cannot be negative");
            }
            size += static_cast<std::size_t>(count);
        }
        return size;
    }();

    if (batch.send_counts.size() != static_cast<std::size_t>(impl_->config.total_gpus) ||
        batch.receive_counts.size() != static_cast<std::size_t>(impl_->config.total_gpus) ||
        batch.send_offsets.size() != static_cast<std::size_t>(impl_->config.total_gpus) ||
        batch.receive_offsets.size() != static_cast<std::size_t>(impl_->config.total_gpus)) {
        throw std::invalid_argument("TransferBatch metadata must contain one entry per rank");
    }
    if (batch.send_buffer.size() != [&batch] {
            std::size_t size = 0;
            for (const int count : batch.send_counts) {
                if (count < 0) {
                    throw std::invalid_argument("send counts cannot be negative");
                }
                size += static_cast<std::size_t>(count);
            }
            return size;
        }()) {
        throw std::invalid_argument("TransferBatch send buffer size does not match send counts");
    }

    float* send_device = nullptr;
    float* receive_device = nullptr;
    try {
        if (!batch.send_buffer.empty()) {
            check_cuda(cudaMalloc(&send_device, batch.send_buffer.size() * sizeof(float)), "cudaMalloc(send batch)");
            check_cuda(cudaMemcpy(send_device, batch.send_buffer.data(), batch.send_buffer.size() * sizeof(float),
                                  cudaMemcpyHostToDevice), "cudaMemcpy H2D batch");
        }
        if (expected_receive_size != 0) {
            check_cuda(cudaMalloc(&receive_device, expected_receive_size * sizeof(float)), "cudaMalloc(receive batch)");
        }

        exchange_async(send_device, batch.send_buffer.size(),
                       batch.send_counts.data(), batch.send_counts.size(),
                       batch.receive_counts.data(), batch.receive_counts.size(),
                       batch.send_offsets.data(), batch.receive_offsets.data(),
                       receive_device, expected_receive_size);
        synchronize();

        batch.receive_buffer.resize(expected_receive_size);
        if (!batch.receive_buffer.empty()) {
            check_cuda(cudaMemcpy(batch.receive_buffer.data(), receive_device,
                                  batch.receive_buffer.size() * sizeof(float), cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H batch");
        }
    } catch (...) {
        cudaFree(receive_device);
        cudaFree(send_device);
        throw;
    }
    cudaFree(receive_device);
    cudaFree(send_device);
}

void CudaNcclTransport::synchronize() {
    check_cuda(cudaEventSynchronize(impl_->transfer_complete), "cudaEventSynchronize");
}

}  // namespace moe
