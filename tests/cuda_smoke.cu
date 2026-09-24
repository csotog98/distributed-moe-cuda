#include "cuda_backend.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <chrono>
#include <cmath>
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

        const std::vector<float> rank_zero_input{1.0F, 2.0F, 3.0F, 4.0F};
        const std::vector<float> rank_one_input{-1.0F, -2.0F, -3.0F, -4.0F};
        const std::vector<std::vector<moe::Token>> tokens_by_rank = total_gpus == 1
            ? std::vector<std::vector<moe::Token>>{{{100, rank_zero_input}}}
            : std::vector<std::vector<moe::Token>>{{{100, rank_zero_input}}, {{101, rank_one_input}}};
        moe::TokenRouter router(4, static_cast<std::size_t>(total_gpus));
        const moe::DistributedTransferPlan distributed_plan =
            router.build_distributed_transfer_plan(tokens_by_rank, elements);
        moe::TransferBatch transfer_batch = distributed_plan.per_rank[static_cast<std::size_t>(rank)];
        const std::vector<float>& expected_output = total_gpus == 1
            ? rank_zero_input
            : (rank == 0 ? rank_one_input : rank_zero_input);

        {
            std::cout << "[rank " << rank << "] before transport.exchange_transfer_batch\n" << std::flush;
            moe::CudaNcclTransport transport({rank, total_gpus, 1, communicator});
            transport.exchange_transfer_batch(transfer_batch);
            std::cout << "[rank " << rank << "] after transport.exchange_transfer_batch\n" << std::flush;
            if (transfer_batch.receive_buffer != expected_output) {
                std::cerr << "[rank " << rank << "] received hidden state:";
                for (const float value : transfer_batch.receive_buffer) {
                    std::cerr << ' ' << value;
                }
                std::cerr << "\n";
                throw std::runtime_error("NCCL hidden-state exchange returned unexpected data");
            }

            moe::TransferBatch return_batch;
            return_batch.send_counts.resize(static_cast<std::size_t>(total_gpus), 0);
            return_batch.receive_counts.resize(static_cast<std::size_t>(total_gpus), 0);
            return_batch.send_offsets.resize(static_cast<std::size_t>(total_gpus), 0);
            return_batch.receive_offsets.resize(static_cast<std::size_t>(total_gpus), 0);

            for (const auto& routed : distributed_plan.received_tokens_by_rank[static_cast<std::size_t>(rank)]) {
                ++return_batch.send_counts[static_cast<std::size_t>(routed.source_rank)];
            }
            for (std::size_t source_rank = 0; source_rank < static_cast<std::size_t>(total_gpus); ++source_rank) {
                return_batch.receive_counts[source_rank] =
                    static_cast<int>(distributed_plan.per_rank[source_rank].send_counts[static_cast<std::size_t>(rank)] /
                                     static_cast<int>(elements));
            }

            std::size_t send_size = 0;
            std::size_t receive_size = 0;
            for (std::size_t peer = 0; peer < static_cast<std::size_t>(total_gpus); ++peer) {
                return_batch.send_offsets[peer] = send_size;
                return_batch.receive_offsets[peer] = receive_size;
                send_size += static_cast<std::size_t>(return_batch.send_counts[peer]);
                receive_size += static_cast<std::size_t>(return_batch.receive_counts[peer]);
            }
            return_batch.send_buffer.resize(send_size, 0.0F);
            return_batch.receive_buffer.resize(receive_size, 0.0F);

            std::vector<std::size_t> output_cursors = return_batch.send_offsets;
            moe::CudaExpertExecutor cuda_expert(elements, 4);
            std::size_t received_offset = 0;
            for (const auto& routed : distributed_plan.received_tokens_by_rank[static_cast<std::size_t>(rank)]) {
                const std::vector<float> hidden(
                    transfer_batch.receive_buffer.begin() + static_cast<std::ptrdiff_t>(received_offset),
                    transfer_batch.receive_buffer.begin() + static_cast<std::ptrdiff_t>(received_offset + elements));
                const float output = cuda_expert.forward(hidden, routed.expert_id).front();
                received_offset += elements;
                const std::size_t destination = static_cast<std::size_t>(routed.source_rank);
                return_batch.send_buffer[output_cursors[destination]++] = output;
            }

            std::cout << "[rank " << rank << "] before return exchange_transfer_batch\n" << std::flush;
            transport.exchange_transfer_batch(return_batch);
            std::cout << "[rank " << rank << "] after return exchange_transfer_batch\n" << std::flush;

            moe::ExpertLayer expected_layer(elements, 4);
            std::vector<float> expected_return;
            for (const auto& token : tokens_by_rank[static_cast<std::size_t>(rank)]) {
                    const auto expert_id = router.route_token_topk(token, 1).front().expert_id;
                    expected_return.push_back(expected_layer.forward(token, expert_id).front());
            }
            if (return_batch.receive_buffer.size() != expected_return.size()) {
                throw std::runtime_error("NCCL return exchange returned an unexpected number of outputs");
            }
            for (std::size_t index = 0; index < expected_return.size(); ++index) {
                if (std::abs(return_batch.receive_buffer[index] - expected_return[index]) > 1.0e-4F) {
                    std::cerr << "[rank " << rank << "] output index " << index
                              << " CUDA=" << return_batch.receive_buffer[index]
                              << " CPU=" << expected_return[index]
                              << " difference=" << (return_batch.receive_buffer[index] - expected_return[index])
                              << "\n";
                    throw std::runtime_error("CUDA expert output differed from the CPU reference");
                }
            }
        }

        const std::vector<float>& output = transfer_batch.receive_buffer;
        check_nccl(ncclCommDestroy(communicator), "ncclCommDestroy");

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