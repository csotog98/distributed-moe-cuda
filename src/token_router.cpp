#include "token_router.hpp"

#include <stdexcept>

namespace moe {

TokenRouter::TokenRouter(std::size_t experts, std::size_t total_gpus)
    : expert_count_(experts), total_gpus_(total_gpus) {
    if (expert_count_ == 0 || total_gpus_ == 0) {
        throw std::invalid_argument("experts and total_gpus must be positive");
    }
}

RoutingPlan TokenRouter::route(const Token* tokens, std::size_t token_count) const {
    RoutingPlan plan;
    plan.by_rank.resize(total_gpus_);
    plan.sends.reserve(token_count);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const std::size_t expert = token.id % expert_count_; // provisional way to choose an expert for a token
        const int destination = static_cast<int>(expert % total_gpus_);
        const std::size_t position = plan.sends.size();
        plan.sends.push_back(RoutedToken{
            token.id, token_index, static_cast<std::uint32_t>(expert), destination});
        plan.by_rank[static_cast<std::size_t>(destination)].push_back(position);
    }
    return plan;
}

}  // namespace moe