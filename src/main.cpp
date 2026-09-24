#include "token_router.hpp"

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t total_gpus = 2;
    const std::size_t expert_count = 4;
    const std::size_t hidden_size = 3;

    moe::TokenRouter router(expert_count, total_gpus);
    moe::ExpertLayer expert_layer(hidden_size, expert_count);

    const std::vector<moe::Token> batch{
        {0, {0.1F, 0.2F, 0.3F}},
        {1, {0.4F, 0.5F, 0.6F}},
        {2, {0.7F, 0.8F, 0.9F}},
        {3, {1.0F, 1.1F, 1.2F}},
        {4, {1.3F, 1.4F, 1.5F}},
    };

    const moe::RoutingPlan plan = router.route(batch.data(), batch.size());
    const auto dispatch = router.build_rank_dispatch(batch.data(), batch.size(), 2);
    const auto microbatches = router.build_microbatches(batch.data(), batch.size(), 2);
    const auto transfer_buffers = router.build_transfer_buffers(batch.data(), batch.size(), 2);
    const auto transfer_batch = router.build_transfer_batch(batch.data(), batch.size(), hidden_size, 2);
    std::cout << "routed " << plan.sends.size() << " tokens across "
              << total_gpus << " ranks\n";
    for (std::size_t rank = 0; rank < plan.by_rank.size(); ++rank) {
        std::cout << "rank " << rank << ": " << plan.by_rank[rank].size()
                  << " tokens\n";
    }

    std::cout << "rank dispatch plan:\n";
    for (const auto& entry : dispatch) {
        std::cout << " rank " << entry.rank << " ->";
        for (const auto index : entry.token_indices) {
            std::cout << " token " << batch[index].id;
        }
        std::cout << '\n';
    }

    std::cout << "microbatch plan:\n";
    for (const auto& batch_entry : microbatches.microbatches) {
        std::cout << " rank " << batch_entry.rank << " batch " << batch_entry.batch_index << " ->";
        for (const auto index : batch_entry.token_indices) {
            std::cout << " token " << batch[index].id;
        }
        std::cout << '\n';

        const std::size_t expert_id = static_cast<std::size_t>(
            router.route_token_topk(batch[batch_entry.token_indices.front()], 1).front().expert_id);
        const auto microbatch_output = expert_layer.forward_batch(batch.data(), batch_entry.token_indices, expert_id);
        std::cout << "   expert output values:";
        for (const float value : microbatch_output) {
            std::cout << " " << std::fixed << std::setprecision(4) << value;
        }
        std::cout << '\n';
    }

    std::cout << "transfer buffers:\n";
    for (const auto& buffer : transfer_buffers) {
        std::cout << " rank " << buffer.rank << " ->";
        for (const auto& payload : buffer.payloads) {
            std::cout << " token " << payload.token_id << " (output " << std::fixed << std::setprecision(4)
                      << payload.expert_output << ")";
        }
        std::cout << '\n';
    }

    std::cout << "transfer batch counts:";
    for (const auto value : transfer_batch.send_counts) {
        std::cout << " " << value;
    }
    std::cout << "\ntransfer batch flat buffer size: " << transfer_batch.send_buffer.size() << "\n";

    std::cout << "top-k expert selections:\n";
    for (const auto& token : batch) {
        const auto topk = router.route_token_topk(token, 2);
        std::cout << " token " << token.id << " ->";
        for (const auto& candidate : topk) {
            std::cout << " expert " << candidate.expert_id << " (score " << std::fixed << std::setprecision(4)
                      << candidate.score << ")";
        }
        std::cout << '\n';
    }

    std::cout << "expert outputs:\n";
    for (const auto& send : plan.sends) {
        const auto& token = batch[send.token_index];
        const auto output = expert_layer.forward(token, send.expert_id);
        std::cout << " token " << send.token_id << " -> expert " << send.expert_id
                  << " -> rank " << send.destination_rank
                  << " -> output " << std::fixed << std::setprecision(4) << output[0] << "\n";
    }

    const auto execution = router.execute_remote_expert_pass(batch.data(), batch.size(), expert_layer, 1);
    std::cout << "final remote execution pass:\n";
    for (std::size_t i = 0; i < execution.outputs.size(); ++i) {
        const auto& result = execution.outputs[i];
        std::cout << " token " << result.token_id << " -> expert " << result.expert_id
                  << " -> rank " << result.destination_rank
                  << " -> merged output " << std::fixed << std::setprecision(4) << execution.merged_outputs[i]
                  << "\n";
    }

    const std::vector<std::vector<moe::Token>> tokens_by_rank{
        {{100, {1.0F, 2.0F, 3.0F}}},
        {{101, {-1.0F, -2.0F, -3.0F}}},
    };
    const auto distributed_transfer = router.build_distributed_transfer_plan(tokens_by_rank, hidden_size);
    const auto distributed_execution = router.execute_distributed_transfer_plan(
        tokens_by_rank, distributed_transfer, expert_layer, 1);

    std::cout << "distributed transfer plan:\n";
    for (std::size_t rank = 0; rank < distributed_transfer.per_rank.size(); ++rank) {
        const auto& transfer = distributed_transfer.per_rank[rank];
        std::cout << " rank " << rank << " sends:";
        for (const int count : transfer.send_counts) {
            std::cout << " " << count;
        }
        std::cout << " receives:";
        for (const int count : transfer.receive_counts) {
            std::cout << " " << count;
        }
        std::cout << "\n";
    }

    std::cout << "distributed merged outputs:\n";
    for (std::size_t rank = 0; rank < distributed_execution.merged_outputs_by_rank.size(); ++rank) {
        for (std::size_t index = 0; index < distributed_execution.merged_outputs_by_rank[rank].size(); ++index) {
            std::cout << " rank " << rank << " token " << tokens_by_rank[rank][index].id
                      << " -> output " << std::fixed << std::setprecision(4)
                      << distributed_execution.merged_outputs_by_rank[rank][index] << "\n";
        }
    }

    return 0;
}
