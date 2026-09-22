#include "vector_index.hpp"

#include <iostream>
#include <vector>

int main() {
    constexpr std::size_t world_size = 2;
    moe::TokenRouter router(/*experts=*/4, world_size);
    const std::vector<moe::Token> batch{
        {0, {0.1F, 0.2F}}, {1, {0.3F, 0.4F}}, {2, {0.5F, 0.6F}},
        {3, {0.7F, 0.8F}}, {4, {0.9F, 1.0F}},
    };

    const moe::RoutingPlan plan = router.route(batch);
    std::cout << "routed " << plan.sends.size() << " tokens across "
              << world_size << " ranks\n";
    for (std::size_t rank = 0; rank < plan.by_rank.size(); ++rank) {
        std::cout << "rank " << rank << ": " << plan.by_rank[rank].size()
                  << " tokens\n";
    }
    return 0;
}
