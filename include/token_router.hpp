#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace moe {

struct Token {
    std::uint32_t id{};
    std::vector<float> hidden;
};

struct RoutedToken {
    std::uint32_t token_id{};
    std::size_t token_index{};
    std::uint32_t expert_id{};
    int destination_rank{};
    int source_rank{};
};

struct RoutingPlan {
    std::vector<RoutedToken> sends;
    std::vector<std::vector<std::size_t>> by_rank;
};

struct RankDispatch {
    int rank{};
    std::vector<std::size_t> token_indices;
};

struct Microbatch {
    int rank{};
    std::size_t batch_index{};
    std::vector<std::size_t> token_indices;
};

struct BatchPlan {
    std::vector<Microbatch> microbatches;
};

struct TransferPayload {
    std::uint32_t token_id{};
    std::size_t token_index{};
    std::vector<float> hidden;
    float expert_output{};
};

struct TransferBuffer {
    int rank{};
    std::vector<TransferPayload> payloads;
};

struct TransferBatch {
    std::vector<int> send_counts;
    std::vector<int> receive_counts;
    std::vector<std::size_t> send_offsets;
    std::vector<std::size_t> receive_offsets;
    std::vector<float> send_buffer;
    std::vector<float> receive_buffer;
};

struct DistributedTransferPlan {
    std::vector<TransferBatch> per_rank;
    std::vector<std::vector<RoutedToken>> routed_tokens_by_destination;
    std::vector<std::vector<RoutedToken>> received_tokens_by_rank;
};

struct DistributedTransferExecution {
    std::vector<std::vector<float>> expert_outputs_by_rank;
    std::vector<std::vector<float>> returned_outputs_by_rank;
    std::vector<std::vector<float>> merged_outputs_by_rank;
};

struct ExpertPreference {
    std::uint32_t expert_id{};
    float score{};
};

struct ExpertExecutionResult {
    std::uint32_t token_id{};
    std::size_t token_index{};
    std::uint32_t expert_id{};
    int destination_rank{};
    float output{};
};

struct RankExecutionPass {
    int rank{};
    std::vector<ExpertExecutionResult> outputs;
};

struct DistributedExecutionPass {
    std::vector<RankExecutionPass> by_rank;
    std::vector<float> merged_outputs;
};

struct CommunicationStage {
    int source_rank{};
    int destination_rank{};
    std::vector<std::size_t> token_indices;
    std::vector<std::size_t> offsets;
};

struct CommunicationPlan {
    std::vector<CommunicationStage> send_stages;
    std::vector<CommunicationStage> receive_stages;
};

struct RankExchangeStage {
    int source_rank{};
    int destination_rank{};
    std::vector<std::size_t> token_indices;
    std::vector<float> payload;
};

struct AllToAllExchangePlan {
    std::vector<RankExchangeStage> steps;
    std::vector<std::vector<std::size_t>> tokens_per_rank;
};

struct ShardedExpert {
    std::uint32_t expert_id{};
    int owner_rank{};
    std::vector<std::size_t> token_indices;
};

struct ExpertShardPlan {
    std::vector<ShardedExpert> shards;
    std::vector<std::vector<std::size_t>> tokens_by_rank;
};

struct RankExecutionContext {
    int rank{};
    std::vector<std::size_t> local_token_indices;
    std::vector<float> local_inputs;
    std::vector<float> local_outputs;
};

struct MultiRankExecutionPlan {
    std::vector<RankExecutionContext> contexts;
    std::vector<float> merged_outputs;
};

struct RankTransferState {
    int rank{};
    std::vector<std::size_t> token_indices;
    std::vector<float> send_buffer;
    std::vector<float> receive_buffer;
    std::vector<std::size_t> send_offsets;
    std::vector<std::size_t> receive_offsets;
};

struct DistributedRuntimePlan {
    std::vector<RankTransferState> per_rank;
    std::vector<float> merged_outputs;
};

struct PipelineSnapshot {
    RoutingPlan route;
    std::vector<RankDispatch> dispatch;
    BatchPlan microbatches;
    std::vector<TransferBuffer> transfer_buffers;
    CommunicationPlan communication;
    DistributedExecutionPass execution;
    AllToAllExchangePlan exchange;
    DistributedRuntimePlan runtime;
    std::vector<float> final_outputs;
};

struct ExpertExecutionPass {
    std::vector<ExpertExecutionResult> outputs;
    std::vector<float> merged_outputs;
};

class ExpertLayer {
public:
    ExpertLayer(std::size_t hidden_size, std::size_t expert_count);

    std::size_t hidden_size() const noexcept { return hidden_size_; }
    std::size_t expert_count() const noexcept { return expert_count_; }

    std::vector<float> forward(const Token& token, std::size_t expert_id) const;
    std::vector<float> forward(const std::vector<float>& hidden, std::size_t expert_id) const;
    std::vector<float> forward_batch(const Token* tokens,
                                    const std::vector<std::size_t>& token_indices,
                                    std::size_t expert_id) const;

private:
    std::size_t hidden_size_{};
    std::size_t expert_count_{};
    std::vector<std::vector<float>> weights_;
    std::vector<float> biases_;
};

class TokenRouter {
public:
    TokenRouter(std::size_t experts, std::size_t total_gpus);

    RoutingPlan route(const Token* tokens, std::size_t token_count) const;
    std::vector<RankDispatch> build_rank_dispatch(const Token* tokens, std::size_t token_count, std::size_t top_k = 1) const;
    BatchPlan build_microbatches(const Token* tokens, std::size_t token_count, std::size_t microbatch_size = 2) const;
    std::vector<TransferBuffer> build_transfer_buffers(const Token* tokens,
                                                     std::size_t token_count,
                                                     std::size_t microbatch_size = 2) const;
    TransferBatch build_transfer_batch(const Token* tokens,
                                     std::size_t token_count,
                                     std::size_t hidden_size,
                                     std::size_t microbatch_size = 2) const;
    DistributedTransferPlan build_distributed_transfer_plan(
        const std::vector<std::vector<Token>>& tokens_by_rank,
        std::size_t hidden_size) const;
    DistributedTransferExecution execute_distributed_transfer_plan(
        const std::vector<std::vector<Token>>& tokens_by_rank,
        const DistributedTransferPlan& plan,
        const ExpertLayer& expert_layer,
        std::size_t top_k = 1) const;
    std::vector<ExpertPreference> route_token_topk(const Token& token, std::size_t top_k) const;
    ExpertExecutionPass execute_remote_expert_pass(const Token* tokens,
                                                 std::size_t token_count,
                                                 const ExpertLayer& expert_layer,
                                                 std::size_t top_k = 1) const;
    DistributedExecutionPass execute_distributed_pass(const Token* tokens,
                                                    std::size_t token_count,
                                                    const ExpertLayer& expert_layer,
                                                    std::size_t top_k = 1) const;
    CommunicationPlan build_communication_plan(const Token* tokens,
                                              std::size_t token_count,
                                              std::size_t top_k,
                                              int local_rank) const;
    AllToAllExchangePlan build_all_to_all_exchange(const Token* tokens,
                                                  std::size_t token_count,
                                                  const ExpertLayer& expert_layer,
                                                  std::size_t top_k,
                                                  int local_rank) const;
    ExpertShardPlan build_expert_shard_plan(const Token* tokens,
                                          std::size_t token_count,
                                          const ExpertLayer& expert_layer,
                                          std::size_t top_k = 1) const;
    MultiRankExecutionPlan build_multi_rank_execution_plan(const Token* tokens,
                                                         std::size_t token_count,
                                                         const ExpertLayer& expert_layer,
                                                         std::size_t top_k = 1) const;
    DistributedRuntimePlan build_distributed_runtime_plan(const Token* tokens,
                                                        std::size_t token_count,
                                                        const ExpertLayer& expert_layer,
                                                        std::size_t top_k = 1) const;
    PipelineSnapshot run_pipeline(const Token* tokens,
                                 std::size_t token_count,
                                 const ExpertLayer& expert_layer,
                                 std::size_t microbatch_size = 2,
                                 std::size_t top_k = 1,
                                 int local_rank = 0) const;
    std::size_t expert_count() const noexcept { return expert_count_; }
    std::size_t total_gpus() const noexcept { return total_gpus_; }

private:
    std::size_t expert_count_;
    std::size_t total_gpus_;
    float score_for_expert(const Token& token, std::size_t expert_id) const;
};

}  // namespace moe