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
};

struct RoutingPlan {
    std::vector<RoutedToken> sends;
    std::vector<std::vector<std::size_t>> by_rank;
};

class TokenRouter {
public:
    TokenRouter(std::size_t experts, std::size_t total_gpus);

    RoutingPlan route(const Token* tokens, std::size_t token_count) const;
    std::size_t expert_count() const noexcept { return expert_count_; }
    std::size_t total_gpus() const noexcept { return total_gpus_; }

private:
    std::size_t expert_count_;
    std::size_t total_gpus_;
};

}  // namespace moe