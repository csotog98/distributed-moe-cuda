#include "token_router.hpp"

#include <cassert>
#include <vector>

int main() {
    moe::TokenRouter router(4, 2);
    const std::vector<moe::Token> tokens{{0, {1.0F}}, {1, {2.0F}}, {2, {3.0F}}, {3, {4.0F}}, {4, {5.0F}}};
    const moe::RoutingPlan plan = router.route(tokens.data(), tokens.size());

    assert(plan.sends.size() == tokens.size());
    assert(plan.sends[0].expert_id == 0 && plan.sends[0].destination_rank == 0);
    assert(plan.sends[0].token_id == tokens[0].id && plan.sends[0].token_index == 0);
    assert(plan.sends[1].expert_id == 1 && plan.sends[1].destination_rank == 1);
    assert(plan.sends[4].expert_id == 0 && plan.sends[4].destination_rank == 0);
    assert(plan.by_rank[0].size() == 3);
    assert(plan.by_rank[1].size() == 2);
}