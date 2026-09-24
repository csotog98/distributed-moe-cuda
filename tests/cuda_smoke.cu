#include "cuda_backend.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
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

bool save_unique_id_to_file(const std::string& path, const ncclUniqueId& unique_id) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        return false;
    }
    output.write(reinterpret_cast<const char*>(&unique_id), sizeof(unique_id));
    output.flush();
    return output.good();
}

bool load_unique_id_from_file(const std::string& path, ncclUniqueId& unique_id) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        return false;
    }
    input.read(reinterpret_cast<char*>(&unique_id), sizeof(unique_id));
    return input.good() || input.eof();
}

int main(int argc, char** argv) {
    try {
        int rank = 0;
        int total_gpus = 1;
        std::string nccl_id_file;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--rank" && i + 1 < argc) {
                rank = std::stoi(argv[++i]);
            } else if (arg == "--total-gpus" && i + 1 < argc) {
                total_gpus = std::stoi(argv[++i]);
            } else if (arg == "--nccl-id-file" && i + 1 < argc) {
                nccl_id_file = argv[++i];
            } else if (arg == "--help") {
                std::cout << "Usage: moe_cuda_smoke [--rank N] [--total-gpus N] [--nccl-id-file path]\n";
                return EXIT_SUCCESS;
            }
        }

        const char* env_path = std::getenv("MOE_NCCL_ID_FILE");
        if (!nccl_id_file.empty() && env_path != nullptr && std::string(env_path) != "") {
            nccl_id_file = env_path;
        }
        if (nccl_id_file.empty()) {
            nccl_id_file = std::getenv("MOE_NCCL_ID_FILE") ? std::string(std::getenv("MOE_NCCL_ID_FILE")) : "";
        }

        if (total_gpus <= 0) {
            throw std::invalid_argument("total_gpus must be positive");
        }
        if (rank < 0 || rank >= total_gpus) {
            throw std::invalid_argument("rank must be in [0, total_gpus)");
        }

        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) {
            std::cout << "CUDA smoke test skipped: no CUDA-capable device detected\n";
            return EXIT_SUCCESS;
        }
        if (total_gpus > device_count) {
            std::cout << "CUDA smoke test skipped: requested " << total_gpus
                      << " ranks but only " << device_count << " CUDA devices are available\n";
            return EXIT_SUCCESS;
        }

        constexpr std::size_t elements = 4;
        check_cuda(cudaSetDevice(rank), "cudaSetDevice");

        ncclUniqueId unique_id{};
        if (!nccl_id_file.empty()) {
            if (rank == 0) {
                check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
                if (!save_unique_id_to_file(nccl_id_file, unique_id)) {
                    throw std::runtime_error("failed to write shared NCCL unique id to file: " + nccl_id_file);
                }
            } else {
                for (int attempt = 0; attempt < 200; ++attempt) {
                    if (load_unique_id_from_file(nccl_id_file, unique_id)) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!load_unique_id_from_file(nccl_id_file, unique_id)) {
                    throw std::runtime_error("timed out waiting for shared NCCL unique id at: " + nccl_id_file);
                }
            }
        } else {
            check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
        }

        ncclComm_t communicator{};
        check_nccl(ncclCommInitRank(&communicator, total_gpus, unique_id, rank), "ncclCommInitRank");

        float* send_device = nullptr;
        float* receive_device = nullptr;
        check_cuda(cudaMalloc(&send_device, elements * sizeof(float)), "cudaMalloc(send)");
        check_cuda(cudaMalloc(&receive_device, elements * sizeof(float)), "cudaMalloc(receive)");

        const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
        check_cuda(cudaMemcpy(send_device, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");

        std::vector<int> send_counts(static_cast<std::size_t>(total_gpus), 0);
        std::vector<int> receive_counts(static_cast<std::size_t>(total_gpus), 0);
        std::vector<std::size_t> send_offsets(static_cast<std::size_t>(total_gpus), 0);
        std::vector<std::size_t> receive_offsets(static_cast<std::size_t>(total_gpus), 0);
        send_counts[rank] = static_cast<int>(elements);
        receive_counts[rank] = static_cast<int>(elements);

        {
            moe::CudaNcclTransport transport({rank, total_gpus, 1, communicator});
            transport.exchange_async(send_device, elements,
                                     send_counts.data(), send_counts.size(),
                                     receive_counts.data(), receive_counts.size(),
                                     send_offsets.data(), receive_offsets.data(),
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

        std::cout << "NCCL CUDA smoke test passed on rank " << rank << " with " << total_gpus << " ranks\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "NCCL CUDA smoke test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}