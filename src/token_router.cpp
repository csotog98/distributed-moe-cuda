#include "token_router.hpp"

#include <stdexcept>

namespace moe {

TokenRouter::TokenRouter(std::size_t experts, std::size_t world_size)
    : expert_count_(experts), world_size_(world_size) {
    if (expert_count_ == 0 || world_size_ == 0) {
        throw std::invalid_argument("experts and world_size must be positive");
    }
}

RoutingPlan TokenRouter::route(const Token* tokens, std::size_t token_count) const {
    RoutingPlan plan;
    plan.by_rank.resize(world_size_);
    plan.sends.reserve(token_count);

    for (std::size_t token_index = 0; token_index < token_count; ++token_index) {
        const Token& token = tokens[token_index];
        const std::size_t expert = token.id % expert_count_;
        const int destination = static_cast<int>(expert % world_size_);
        const std::size_t position = plan.sends.size();
        plan.sends.push_back(RoutedToken{
            token.id, token_index, static_cast<std::uint32_t>(expert), destination});
        plan.by_rank[static_cast<std::size_t>(destination)].push_back(position);
    }
    return plan;
}

}  // namespace moe