#include "cuda_backend.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

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

int main() {
    try {
        constexpr int rank = 0;
        constexpr int world_size = 1;
        constexpr std::size_t elements = 4;

        check_cuda(cudaSetDevice(rank), "cudaSetDevice");

        ncclUniqueId unique_id{};
        check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
        ncclComm_t communicator{};
        check_nccl(ncclCommInitRank(&communicator, world_size, unique_id, rank), "ncclCommInitRank");

        float* send_device = nullptr;
        float* receive_device = nullptr;
        check_cuda(cudaMalloc(&send_device, elements * sizeof(float)), "cudaMalloc(send)");
        check_cuda(cudaMalloc(&receive_device, elements * sizeof(float)), "cudaMalloc(receive)");

        const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
        check_cuda(cudaMemcpy(send_device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        {
            moe::CudaNcclTransport transport({rank, world_size, 1, communicator});
            const int element_count = static_cast<int>(elements);
            transport.exchange_async(send_device, elements,
                                     &element_count, 1,
                                     &element_count, 1,
                                     receive_device, elements);
            transport.synchronize();
        }

        std::vector<float> output(elements);
        check_cuda(cudaMemcpy(output.data(), receive_device, output.size() * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
        check_cuda(cudaFree(send_device), "cudaFree(send)");
        check_cuda(cudaFree(receive_device), "cudaFree(receive)");
        check_nccl(ncclCommDestroy(communicator), "ncclCommDestroy");

        if (output != input) {
            std::cerr << "received:";
            for (const float value : output) {
                std::cerr << ' ' << value;
            }
            std::cerr << '\n';
            throw std::runtime_error("NCCL smoke test returned unexpected data");
        }
        std::cout << "NCCL CUDA smoke test passed on rank 0\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "NCCL CUDA smoke test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}