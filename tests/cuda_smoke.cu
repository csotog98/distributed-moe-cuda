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

            if (arg == "--rank" || arg == "-r") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value after --rank");
                }
                rank = std::stoi(argv[++i]);
            } else if (arg == "--total-gpus" || arg == "-g") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value after --total-gpus");
                }
                total_gpus = std::stoi(argv[++i]);
            } else if (arg == "--nccl-id-file" || arg == "-f") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("missing value after --nccl-id-file");
                }
                nccl_id_file = argv[++i];
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: moe_cuda_smoke [rank] [total_gpus] [nccl_id_file] | [--rank N] [--total-gpus N] [--nccl-id-file path]\n";
                return EXIT_SUCCESS;
            } else if (arg.rfind("--", 0) == 0) {
                throw std::invalid_argument("unknown option: " + arg);
            } else if (i == 1) {
                rank = std::stoi(arg);
            } else if (i == 2) {
                total_gpus = std::stoi(arg);
            } else if (i == 3) {
                nccl_id_file = arg;
            } else {
                throw std::invalid_argument("unexpected extra argument: " + arg);
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
        std::cout << "[rank " << rank << "] before cudaSetDevice\n" << std::flush;
        check_cuda(cudaSetDevice(rank), "cudaSetDevice");
        std::cout << "[rank " << rank << "] after cudaSetDevice\n" << std::flush;

        ncclUniqueId unique_id{};
        if (!nccl_id_file.empty()) {
            if (rank == 0) {
                std::cout << "[rank " << rank << "] before ncclGetUniqueId\n" << std::flush;
                check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
                std::cout << "[rank " << rank << "] after ncclGetUniqueId, writing shared id\n" << std::flush;
                if (!save_unique_id_to_file(nccl_id_file, unique_id)) {
                    throw std::runtime_error("failed to write shared NCCL unique id to file: " + nccl_id_file);
                }
            } else {
                std::cout << "[rank " << rank << "] waiting for shared nccl id file: " << nccl_id_file << "\n" << std::flush;
                for (int attempt = 0; attempt < 200; ++attempt) {
                    if (load_unique_id_from_file(nccl_id_file, unique_id)) {
                        std::cout << "[rank " << rank << "] read shared id on attempt " << attempt << "\n" << std::flush;
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                if (!load_unique_id_from_file(nccl_id_file, unique_id)) {
                    throw std::runtime_error("timed out waiting for shared NCCL unique id at: " + nccl_id_file);
                }
            }
        } else {
            std::cout << "[rank " << rank << "] before ncclGetUniqueId\n" << std::flush;
            check_nccl(ncclGetUniqueId(&unique_id), "ncclGetUniqueId");
            std::cout << "[rank " << rank << "] after ncclGetUniqueId\n" << std::flush;
        }

        ncclComm_t communicator{};
        std::cout << "[rank " << rank << "] before ncclCommInitRank\n" << std::flush;
        check_nccl(ncclCommInitRank(&communicator, total_gpus, unique_id, rank), "ncclCommInitRank");
        std::cout << "[rank " << rank << "] after ncclCommInitRank\n" << std::flush;

        const std::vector<float> input{
            static_cast<float>(rank * 10 + 1),
            static_cast<float>(rank * 10 + 2),
            static_cast<float>(rank * 10 + 3),
            static_cast<float>(rank * 10 + 4),
        };
        moe::TransferBatch transfer_batch;
        transfer_batch.send_counts.resize(static_cast<std::size_t>(total_gpus), 0);
        transfer_batch.receive_counts.resize(static_cast<std::size_t>(total_gpus), 0);
        transfer_batch.send_offsets.resize(static_cast<std::size_t>(total_gpus), 0);
        transfer_batch.receive_offsets.resize(static_cast<std::size_t>(total_gpus), 0);
        const int peer = (rank + 1) % total_gpus;
        transfer_batch.send_counts[peer] = static_cast<int>(elements);
        transfer_batch.receive_counts[peer] = static_cast<int>(elements);
        transfer_batch.send_buffer = input;

        {
            std::cout << "[rank " << rank << "] before transport.exchange_transfer_batch\n" << std::flush;
            moe::CudaNcclTransport transport({rank, total_gpus, 1, communicator});
            transport.exchange_transfer_batch(transfer_batch);
            std::cout << "[rank " << rank << "] after transport.exchange_transfer_batch\n" << std::flush;
        }

        const std::vector<float>& output = transfer_batch.receive_buffer;
        check_nccl(ncclCommDestroy(communicator), "ncclCommDestroy");

        const std::vector<float> expected_output{
            static_cast<float>(peer * 10 + 1),
            static_cast<float>(peer * 10 + 2),
            static_cast<float>(peer * 10 + 3),
            static_cast<float>(peer * 10 + 4),
        };
        if (output != expected_output) {
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