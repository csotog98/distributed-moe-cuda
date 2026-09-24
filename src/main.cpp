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
    std::cout << "routed " << plan.sends.size() << " tokens across "
              << total_gpus << " ranks\n";
    for (std::size_t rank = 0; rank < plan.by_rank.size(); ++rank) {
        std::cout << "rank " << rank << ": " << plan.by_rank[rank].size()
                  << " tokens\n";
    }

    std::cout << "expert outputs:\n";
    for (const auto& send : plan.sends) {
        const auto& token = batch[send.token_index];
        const auto output = expert_layer.forward(token, send.expert_id);
        std::cout << " token " << send.token_id << " -> expert " << send.expert_id
                  << " -> rank " << send.destination_rank
                  << " -> output " << std::fixed << std::setprecision(4) << output[0] << "\n";
    }

    return 0;
}
