#include "token_router.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace moe {
namespace {

float compute_hidden_energy(const std::vector<float>& hidden) {
    float energy = 0.0F;
    for (const float value : hidden) {
        energy += value * value;
    }
    return energy;
}

}  // namespace

ExpertLayer::ExpertLayer(std::size_t hidden_size, std::size_t expert_count)
    : hidden_size_(hidden_size), expert_count_(expert_count) {
    if (hidden_size_ == 0 || expert_count_ == 0) {
        throw std::invalid_argument("hidden_size and expert_count must be positive");
    }

    weights_.resize(expert_count_);
    biases_.resize(expert_count_, 0.0F);

    for (std::size_t expert = 0; expert < expert_count_; ++expert) {
        weights_[expert].resize(hidden_size_);
        for (std::size_t index = 0; index < hidden_size_; ++index) {
            weights_[expert][index] = 0.1F * static_cast<float>(expert + 1) +
                                     0.01F * static_cast<float>(index + 1);
        }
        biases_[expert] = 0.05F * static_cast<float>(expert + 1);
    }
}

std::vector<float> ExpertLayer::forward(const Token& token, std::size_t expert_id) const {
    return forward(token.hidden, expert_id);
}

std::vector<float> ExpertLayer::forward(const std::vector<float>& hidden, std::size_t expert_id) const {
    if (expert_id >= expert_count_) {
        throw std::out_of_range("expert_id is out of range");
    }
    if (hidden.size() != hidden_size_) {
        throw std::invalid_argument("hidden size does not match expert layer hidden_size");
    }

    std::vector<float> output(1, biases_[expert_id]);
    for (std::size_t index = 0; index < hidden_size_; ++index) {
        output[0] += hidden[index] * weights_[expert_id][index];
    }
    return output;
}

std::vector<float> ExpertLayer::forward_batch(const Token* tokens,
                                            const std::vector<std::size_t>& token_indices,
                                            std::size_t expert_id) const {
    if (token_indices.empty()) {
        return {};
    }
    if (tokens == nullptr) {
        throw std::invalid_argument("tokens pointer cannot be null when token_indices is non-empty");
    }
    if (expert_id >= expert_count_) {
        throw std::out_of_range("expert_id is out of range");
    }

    std::vector<float> outputs;
    outputs.reserve(token_indices.size());
    for (const std::size_t token_index : token_indices) {
        if (token_index >= static_cast<std::size_t>(-1)) {
            // the pointer-based API is intentionally simple; runtime validation is handled by the caller
        }
        if (token_index >= 1'000'000'000ULL) {
            throw std::out_of_range("token_index is out of range for the current batch");
        }
        outputs.push_back(forward(tokens[token_index], expert_id)[0]);
    }
    return outputs;
}

TokenRouter::TokenRouter(std::size_t experts, std::size_t total_gpus)
    : expert_count_(experts), total_gpus_(total_gpus) {
    if (expert_count_ == 0 || total_gpus_ == 0) {
        throw std::invalid_argument("experts and total_gpus must be positive");
    }
}

float TokenRouter::score_for_expert(const Token& token, std::size_t expert_id) const {
    if (expert_id >= expert_count_) {
        throw std::out_of_range("expert_id is out of range");
    }

    const float hidden_energy = compute_hidden_energy(token.hidden);
    float score = 0.0F;
    for (std::size_t index = 0; index < token.hidden.size(); ++index) {
        const float weight = 0.25F * static_cast<float>(expert_id + 1) +
                             0.5F * static_cast<float>(index + 1);
        score += token.hidden[index] * weight;
    }
    score += 0.1F * static_cast<float>(expert_id + 1);
    score += 0.01F * hidden_energy;
    return score;
}

std::vector<ExpertPreference> TokenRouter::route_token_topk(const Token& token, std::size_t top_k) const {
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    std::vector<ExpertPreference> candidates;
    candidates.reserve(expert_count_);
    for (std::size_t expert_id = 0; expert_id < expert_count_; ++expert_id) {
        candidates.push_back({static_cast<std::uint32_t>(expert_id), score_for_expert(token, expert_id)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const ExpertPreference& lhs, const ExpertPreference& rhs) {
        return lhs.score > rhs.score;
    });

    const std::size_t effective_top_k = std::min(top_k, candidates.size());
    candidates.resize(effective_top_k);
    return candidates;
}

std::vector<RankDispatch> TokenRouter::build_rank_dispatch(const Token* tokens, std::size_t token_count, std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and less than or equal to expert_count");
    }

    std::vector<RankDispatch> dispatch(total_gpus_);
    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        dispatch[rank].rank = static_cast<int>(rank);
    }

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination = static_cast<int>(expert % total_gpus_);
        dispatch[static_cast<std::size_t>(destination)].token_indices.push_back(token_index);
    }
    return dispatch;
}

BatchPlan TokenRouter::build_microbatches(const Token* tokens, std::size_t token_count, std::size_t microbatch_size) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (microbatch_size == 0) {
        throw std::invalid_argument("microbatch_size must be positive");
    }

    BatchPlan plan;
    const auto dispatch = build_rank_dispatch(tokens, token_count, 1);
    for (const auto& entry : dispatch) {
        if (entry.token_indices.empty()) {
            continue;
        }

        std::size_t batch_index = 0;
        for (std::size_t offset = 0; offset < entry.token_indices.size(); offset += microbatch_size) {
            const std::size_t end = std::min(offset + microbatch_size, entry.token_indices.size());
            std::vector<std::size_t> batch_tokens;
            batch_tokens.reserve(end - offset);
            for (std::size_t i = offset; i < end; ++i) {
                batch_tokens.push_back(entry.token_indices[i]);
            }
            plan.microbatches.push_back(Microbatch{entry.rank, batch_index, batch_tokens});
            ++batch_index;
        }
    }
    return plan;
}

std::vector<TransferBuffer> TokenRouter::build_transfer_buffers(const Token* tokens,
                                                             std::size_t token_count,
                                                             std::size_t microbatch_size) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (microbatch_size == 0) {
        throw std::invalid_argument("microbatch_size must be positive");
    }

    const auto microbatches = build_microbatches(tokens, token_count, microbatch_size);
    std::vector<TransferBuffer> transfer_buffers(total_gpus_);
    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        transfer_buffers[rank].rank = static_cast<int>(rank);
    }

    for (const auto& microbatch : microbatches.microbatches) {
        TransferBuffer& buffer = transfer_buffers[static_cast<std::size_t>(microbatch.rank)];
        for (const auto token_index : microbatch.token_indices) {
            const Token& token = tokens[token_index];
            const auto top_expert = route_token_topk(token, 1).front();
            const float expert_output = static_cast<float>(top_expert.expert_id + 1) * 0.1F +
                                       static_cast<float>(token.hidden.size()) * 0.05F;
            buffer.payloads.push_back(TransferPayload{
                token.id,
                token_index,
                token.hidden,
                expert_output,
            });
        }
    }
    return transfer_buffers;
}

TransferBatch TokenRouter::build_transfer_batch(const Token* tokens,
                                              std::size_t token_count,
                                              std::size_t hidden_size,
                                              std::size_t microbatch_size) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (microbatch_size == 0) {
        throw std::invalid_argument("microbatch_size must be positive");
    }
    if (hidden_size == 0) {
        throw std::invalid_argument("hidden_size must be positive");
    }

    TransferBatch batch;
    batch.send_counts.resize(total_gpus_, 0);
    batch.receive_counts.resize(total_gpus_, 0);
    batch.send_offsets.resize(total_gpus_, 0);
    batch.receive_offsets.resize(total_gpus_, 0);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != hidden_size) {
            throw std::invalid_argument("token hidden size does not match the expected hidden_size");
        }

        const auto top_experts = route_token_topk(token, 1);
        const std::size_t destination = static_cast<std::size_t>(top_experts.front().expert_id % total_gpus_);
        batch.send_counts[destination] += static_cast<int>(hidden_size);
        batch.receive_counts[destination] += static_cast<int>(hidden_size);
    }

    std::size_t send_offset = 0;
    std::size_t receive_offset = 0;
    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        batch.send_offsets[rank] = send_offset;
        batch.receive_offsets[rank] = receive_offset;
        send_offset += static_cast<std::size_t>(batch.send_counts[rank]);
        receive_offset += static_cast<std::size_t>(batch.receive_counts[rank]);
    }

    batch.send_buffer.resize(send_offset, 0.0F);
    batch.receive_buffer.resize(receive_offset, 0.0F);

    std::vector<std::size_t> send_cursors = batch.send_offsets;

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const auto top_experts = route_token_topk(token, 1);
        const std::size_t destination = static_cast<std::size_t>(top_experts.front().expert_id % total_gpus_);
        const std::size_t offset = send_cursors[destination];
        std::copy(token.hidden.begin(), token.hidden.end(), batch.send_buffer.begin() + static_cast<std::ptrdiff_t>(offset));
        send_cursors[destination] += token.hidden.size();
    }
    return batch;
}

DistributedTransferPlan TokenRouter::build_distributed_transfer_plan(
    const std::vector<std::vector<Token>>& tokens_by_rank,
    std::size_t hidden_size) const {
    if (tokens_by_rank.size() != total_gpus_) {
        throw std::invalid_argument("tokens_by_rank must contain one batch per rank");
    }
    if (hidden_size == 0) {
        throw std::invalid_argument("hidden_size must be positive");
    }

    std::vector<std::vector<int>> send_counts(total_gpus_, std::vector<int>(total_gpus_, 0));
    for (std::size_t source_rank = 0; source_rank < total_gpus_; ++source_rank) {
        for (const Token& token : tokens_by_rank[source_rank]) {
            if (token.hidden.size() != hidden_size) {
                throw std::invalid_argument("token hidden size does not match the expected hidden_size");
            }
            const std::size_t destination = static_cast<std::size_t>(
                route_token_topk(token, 1).front().expert_id % total_gpus_);
            send_counts[source_rank][destination] += static_cast<int>(hidden_size);
        }
    }

    DistributedTransferPlan plan;
    plan.per_rank.resize(total_gpus_);
    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        TransferBatch& batch = plan.per_rank[rank];
        batch.send_counts = send_counts[rank];
        batch.receive_counts.resize(total_gpus_, 0);
        batch.send_offsets.resize(total_gpus_, 0);
        batch.receive_offsets.resize(total_gpus_, 0);

        std::size_t send_size = 0;
        std::size_t receive_size = 0;
        for (std::size_t peer = 0; peer < total_gpus_; ++peer) {
            batch.receive_counts[peer] = send_counts[peer][rank];
            batch.send_offsets[peer] = send_size;
            batch.receive_offsets[peer] = receive_size;
            send_size += static_cast<std::size_t>(batch.send_counts[peer]);
            receive_size += static_cast<std::size_t>(batch.receive_counts[peer]);
        }
        batch.send_buffer.resize(send_size, 0.0F);
        batch.receive_buffer.resize(receive_size, 0.0F);

        std::vector<std::size_t> cursors = batch.send_offsets;
        for (const Token& token : tokens_by_rank[rank]) {
            const std::size_t destination = static_cast<std::size_t>(
                route_token_topk(token, 1).front().expert_id % total_gpus_);
            const std::size_t offset = cursors[destination];
            std::copy(token.hidden.begin(), token.hidden.end(),
                      batch.send_buffer.begin() + static_cast<std::ptrdiff_t>(offset));
            cursors[destination] += token.hidden.size();
        }
    }
    return plan;
}

ExpertExecutionPass TokenRouter::execute_remote_expert_pass(const Token* tokens,
                                                          std::size_t token_count,
                                                          const ExpertLayer& expert_layer,
                                                          std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    ExpertExecutionPass execution;
    execution.outputs.reserve(token_count);
    execution.merged_outputs.resize(token_count, 0.0F);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination_rank = static_cast<int>(expert_id % total_gpus_);
        const auto output = expert_layer.forward(token, expert_id);
        const float value = output.empty() ? 0.0F : output.front();

        execution.outputs.push_back(ExpertExecutionResult{
            token.id,
            token_index,
            static_cast<std::uint32_t>(expert_id),
            destination_rank,
            value,
        });

        execution.merged_outputs[token_index] = value;
    }

    return execution;
}

DistributedExecutionPass TokenRouter::execute_distributed_pass(const Token* tokens,
                                                             std::size_t token_count,
                                                             const ExpertLayer& expert_layer,
                                                             std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    DistributedExecutionPass distributed;
    distributed.by_rank.resize(total_gpus_);
    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        distributed.by_rank[rank].rank = static_cast<int>(rank);
    }
    distributed.merged_outputs.resize(token_count, 0.0F);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination_rank = static_cast<int>(expert_id % total_gpus_);
        const auto output = expert_layer.forward(token, expert_id);
        const float value = output.empty() ? 0.0F : output.front();

        ExpertExecutionResult result{
            token.id,
            token_index,
            static_cast<std::uint32_t>(expert_id),
            destination_rank,
            value,
        };

        distributed.by_rank[static_cast<std::size_t>(destination_rank)].outputs.push_back(result);
        distributed.merged_outputs[token_index] = value;
    }

    return distributed;
}

CommunicationPlan TokenRouter::build_communication_plan(const Token* tokens,
                                                      std::size_t token_count,
                                                      std::size_t top_k,
                                                      int local_rank) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }
    if (local_rank < 0 || static_cast<std::size_t>(local_rank) >= total_gpus_) {
        throw std::invalid_argument("local_rank is out of range");
    }

    CommunicationPlan plan;
    std::vector<std::size_t> send_offsets(total_gpus_, 0);
    std::vector<std::size_t> receive_offsets(total_gpus_, 0);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination_rank = static_cast<int>(expert_id % total_gpus_);

        if (destination_rank == local_rank) {
            plan.receive_stages.push_back(CommunicationStage{
                static_cast<int>(local_rank),
                destination_rank,
                {token_index},
                {receive_offsets[static_cast<std::size_t>(destination_rank)]},
            });
            ++receive_offsets[static_cast<std::size_t>(destination_rank)];
        } else {
            plan.send_stages.push_back(CommunicationStage{
                local_rank,
                destination_rank,
                {token_index},
                {send_offsets[static_cast<std::size_t>(destination_rank)]},
            });
            ++send_offsets[static_cast<std::size_t>(destination_rank)];
        }
    }

    return plan;
}

AllToAllExchangePlan TokenRouter::build_all_to_all_exchange(const Token* tokens,
                                                          std::size_t token_count,
                                                          const ExpertLayer& expert_layer,
                                                          std::size_t top_k,
                                                          int local_rank) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }
    if (local_rank < 0 || static_cast<std::size_t>(local_rank) >= total_gpus_) {
        throw std::invalid_argument("local_rank is out of range");
    }

    AllToAllExchangePlan plan;
    plan.tokens_per_rank.resize(total_gpus_);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination_rank = static_cast<int>(expert_id % total_gpus_);
        plan.tokens_per_rank[static_cast<std::size_t>(destination_rank)].push_back(token_index);

        RankExchangeStage stage{
            local_rank,
            destination_rank,
            {token_index},
            token.hidden,
        };
        plan.steps.push_back(stage);
    }

    return plan;
}

ExpertShardPlan TokenRouter::build_expert_shard_plan(const Token* tokens,
                                                  std::size_t token_count,
                                                  const ExpertLayer& expert_layer,
                                                  std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    ExpertShardPlan plan;
    plan.tokens_by_rank.resize(total_gpus_);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int owner_rank = static_cast<int>(expert_id % total_gpus_);
        auto shard_it = std::find_if(plan.shards.begin(), plan.shards.end(), [expert_id, owner_rank](const ShardedExpert& shard) {
            return shard.expert_id == expert_id && shard.owner_rank == owner_rank;
        });
        if (shard_it == plan.shards.end()) {
            plan.shards.push_back(ShardedExpert{static_cast<std::uint32_t>(expert_id), owner_rank, {token_index}});
        } else {
            shard_it->token_indices.push_back(token_index);
        }
        plan.tokens_by_rank[static_cast<std::size_t>(owner_rank)].push_back(token_index);
    }

    return plan;
}

MultiRankExecutionPlan TokenRouter::build_multi_rank_execution_plan(const Token* tokens,
                                                                   std::size_t token_count,
                                                                   const ExpertLayer& expert_layer,
                                                                   std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    MultiRankExecutionPlan plan;
    plan.contexts.resize(total_gpus_);
    plan.merged_outputs.resize(token_count, 0.0F);

    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        plan.contexts[rank].rank = static_cast<int>(rank);
    }

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int owner_rank = static_cast<int>(expert_id % total_gpus_);
        auto& context = plan.contexts[static_cast<std::size_t>(owner_rank)];
        context.local_token_indices.push_back(token_index);
        context.local_inputs.insert(context.local_inputs.end(), token.hidden.begin(), token.hidden.end());

        const auto output = expert_layer.forward(token, expert_id);
        context.local_outputs.push_back(output.empty() ? 0.0F : output.front());
        plan.merged_outputs[token_index] = output.empty() ? 0.0F : output.front();
    }

    return plan;
}

DistributedRuntimePlan TokenRouter::build_distributed_runtime_plan(const Token* tokens,
                                                                std::size_t token_count,
                                                                const ExpertLayer& expert_layer,
                                                                std::size_t top_k) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }

    DistributedRuntimePlan plan;
    plan.per_rank.resize(total_gpus_);
    plan.merged_outputs.resize(token_count, 0.0F);

    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        plan.per_rank[rank].rank = static_cast<int>(rank);
        plan.per_rank[rank].send_offsets.resize(total_gpus_, 0);
        plan.per_rank[rank].receive_offsets.resize(total_gpus_, 0);
    }

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        if (token.hidden.size() != expert_layer.hidden_size()) {
            throw std::invalid_argument("token hidden size does not match expert layer hidden_size");
        }

        const auto top_experts = route_token_topk(token, top_k);
        const std::size_t expert_id = static_cast<std::size_t>(top_experts.front().expert_id);
        const int owner_rank = static_cast<int>(expert_id % total_gpus_);
        auto& rank_state = plan.per_rank[static_cast<std::size_t>(owner_rank)];
        rank_state.token_indices.push_back(token_index);

        const auto output = expert_layer.forward(token, expert_id);
        const float expert_value = output.empty() ? 0.0F : output.front();
        rank_state.send_buffer.insert(rank_state.send_buffer.end(), token.hidden.begin(), token.hidden.end());
        rank_state.receive_buffer.insert(rank_state.receive_buffer.end(), token.hidden.begin(), token.hidden.end());
        rank_state.send_offsets[static_cast<std::size_t>(owner_rank)] += token.hidden.size();
        rank_state.receive_offsets[static_cast<std::size_t>(owner_rank)] += token.hidden.size();
        plan.merged_outputs[token_index] = expert_value;
    }

    for (std::size_t rank = 0; rank < total_gpus_; ++rank) {
        auto& state = plan.per_rank[rank];
        state.send_offsets.assign(total_gpus_, 0);
        state.receive_offsets.assign(total_gpus_, 0);
        for (std::size_t token_index = 0; token_index < state.token_indices.size(); ++token_index) {
            const std::size_t owner = static_cast<std::size_t>(state.rank % total_gpus_);
            state.send_offsets[owner] += static_cast<std::size_t>(tokens[state.token_indices[token_index]].hidden.size());
            state.receive_offsets[owner] += static_cast<std::size_t>(tokens[state.token_indices[token_index]].hidden.size());
        }
    }

    return plan;
}

PipelineSnapshot TokenRouter::run_pipeline(const Token* tokens,
                                         std::size_t token_count,
                                         const ExpertLayer& expert_layer,
                                         std::size_t microbatch_size,
                                         std::size_t top_k,
                                         int local_rank) const {
    if (tokens == nullptr && token_count != 0) {
        throw std::invalid_argument("tokens pointer cannot be null when token_count is non-zero");
    }
    if (microbatch_size == 0) {
        throw std::invalid_argument("microbatch_size must be positive");
    }
    if (top_k == 0 || top_k > expert_count_) {
        throw std::invalid_argument("top_k must be positive and not exceed expert_count");
    }
    if (local_rank < 0 || static_cast<std::size_t>(local_rank) >= total_gpus_) {
        throw std::invalid_argument("local_rank is out of range");
    }

    PipelineSnapshot snapshot;
    snapshot.route = route(tokens, token_count);
    snapshot.dispatch = build_rank_dispatch(tokens, token_count, top_k);
    snapshot.microbatches = build_microbatches(tokens, token_count, microbatch_size);
    snapshot.transfer_buffers = build_transfer_buffers(tokens, token_count, microbatch_size);
    snapshot.communication = build_communication_plan(tokens, token_count, top_k, local_rank);
    snapshot.execution = execute_distributed_pass(tokens, token_count, expert_layer, top_k);
    snapshot.exchange = build_all_to_all_exchange(tokens, token_count, expert_layer, top_k, local_rank);
    snapshot.runtime = build_distributed_runtime_plan(tokens, token_count, expert_layer, top_k);
    snapshot.final_outputs = snapshot.execution.merged_outputs;
    return snapshot;
}

RoutingPlan TokenRouter::route(const Token* tokens, std::size_t token_count) const {
    RoutingPlan plan;
    plan.by_rank.resize(total_gpus_);
    plan.sends.reserve(token_count);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const auto top_experts = route_token_topk(token, 1);
        const std::size_t expert = static_cast<std::size_t>(top_experts.front().expert_id);
        const int destination = static_cast<int>(expert % total_gpus_);
        const std::size_t position = plan.sends.size();

        plan.sends.push_back(RoutedToken{
            token.id, token_index, static_cast<std::uint32_t>(expert), destination});
        plan.by_rank[static_cast<std::size_t>(destination)].push_back(position);
    }
    return plan;
}

}  // namespace moe