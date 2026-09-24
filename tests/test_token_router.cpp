#include "token_router.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

int main() {
    moe::TokenRouter router(4, 2);

    const std::vector<moe::Token> tokens{{0, {1.0F}}, {1, {2.0F}}, {2, {3.0F}}, {3, {4.0F}}, {4, {5.0F}}, {5, {6.0F}}};
    const moe::RoutingPlan plan = router.route(tokens.data(), tokens.size());

    assert(plan.sends.size() == tokens.size());
    assert(plan.by_rank.size() == 2);
    assert(plan.by_rank[0].size() + plan.by_rank[1].size() == tokens.size());

    assert(plan.sends[0].token_id == tokens[0].id);
    assert(plan.sends[0].token_index == 0);
    assert(plan.sends[0].expert_id == 0);
    assert(plan.sends[0].destination_rank == 0);

    assert(plan.sends[1].expert_id == 1);
    assert(plan.sends[1].destination_rank == 1);

    assert(plan.sends[2].expert_id == 2);
    assert(plan.sends[2].destination_rank == 0);

    assert(plan.sends[3].expert_id == 3);
    assert(plan.sends[3].destination_rank == 1);

    assert(plan.sends[4].expert_id == 0);
    assert(plan.sends[4].destination_rank == 0);

    assert(plan.sends[5].expert_id == 1);
    assert(plan.sends[5].destination_rank == 1);

    assert(plan.by_rank[0].size() == 3);
    assert(plan.by_rank[1].size() == 3);

    const std::vector<moe::Token> same_id_but_different_hidden{{42, {1.0F, 0.0F, 0.0F}},
                                                             {42, {0.0F, 1.0F, 0.0F}},
                                                             {42, {0.0F, 0.0F, 1.0F}}};
    const moe::RoutingPlan hidden_sensitive = router.route(same_id_but_different_hidden.data(), same_id_but_different_hidden.size());
    assert(hidden_sensitive.sends.size() == 3);
    assert(hidden_sensitive.sends[0].expert_id != hidden_sensitive.sends[1].expert_id ||
           hidden_sensitive.sends[0].destination_rank != hidden_sensitive.sends[1].destination_rank);
    assert(hidden_sensitive.sends[1].expert_id != hidden_sensitive.sends[2].expert_id ||
           hidden_sensitive.sends[1].destination_rank != hidden_sensitive.sends[2].destination_rank);

    moe::ExpertLayer expert_layer(3, 4);
    const auto out0 = expert_layer.forward(same_id_but_different_hidden[0], 0);
    const auto out1 = expert_layer.forward(same_id_but_different_hidden[1], 1);
    assert(out0.size() == 1);
    assert(out1.size() == 1);
    assert(out0[0] != out1[0]);
}