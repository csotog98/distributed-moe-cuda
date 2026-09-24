#include "token_router.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

int main() {
    moe::TokenRouter router(4, 2);

    const std::vector<moe::Token> tokens{{0, {1.0F, 0.0F}}, {1, {0.0F, 1.0F}}, {2, {1.0F, 1.0F}}, {3, {2.0F, 0.0F}},
                                        {4, {0.0F, 2.0F}}, {5, {3.0F, 0.0F}}};
    const moe::RoutingPlan plan = router.route(tokens.data(), tokens.size());

    assert(plan.sends.size() == tokens.size());
    assert(plan.by_rank.size() == 2);
    assert(plan.by_rank[0].size() + plan.by_rank[1].size() == tokens.size());

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        assert(plan.sends[i].token_id == tokens[i].id);
        assert(plan.sends[i].token_index == i);
    }

    assert(plan.sends[0].expert_id == 0 || plan.sends[0].expert_id == 1 || plan.sends[0].expert_id == 2 || plan.sends[0].expert_id == 3);
    assert(plan.sends[0].destination_rank == static_cast<int>(plan.sends[0].expert_id % 2));

    const std::vector<moe::Token> same_id_but_different_hidden{{42, {1.0F, 0.0F, 0.0F}},
                                                             {42, {0.0F, 1.0F, 0.0F}},
                                                             {42, {0.0F, 0.0F, 1.0F}}};
    const auto topk0 = router.route_token_topk(same_id_but_different_hidden[0], 2);
    const auto topk1 = router.route_token_topk(same_id_but_different_hidden[1], 2);
    const auto topk2 = router.route_token_topk(same_id_but_different_hidden[2], 2);

    assert(topk0.size() == 2);
    assert(topk1.size() == 2);
    assert(topk2.size() == 2);
    assert(topk0[0].score >= topk0[1].score);
    assert(topk1[0].score >= topk1[1].score);
    assert(topk2[0].score >= topk2[1].score);
    assert(topk0[0].expert_id != topk0[1].expert_id);
    assert(topk1[0].expert_id != topk1[1].expert_id);
    assert(topk2[0].expert_id != topk2[1].expert_id);

    const auto dispatch = router.build_rank_dispatch(same_id_but_different_hidden.data(), same_id_but_different_hidden.size(), 2);
    assert(dispatch.size() == 2);
    assert(dispatch[0].rank == 0 || dispatch[0].rank == 1);
    assert(dispatch[1].rank == 0 || dispatch[1].rank == 1);
    assert(dispatch[0].token_indices.size() + dispatch[1].token_indices.size() == same_id_but_different_hidden.size());
    for (const auto& entry : dispatch) {
        for (const std::size_t index : entry.token_indices) {
            assert(index < same_id_but_different_hidden.size());
        }
    }

    const auto batch_plan = router.build_microbatches(same_id_but_different_hidden.data(), same_id_but_different_hidden.size(), 2);
    assert(!batch_plan.microbatches.empty());
    std::size_t total_packaged = 0;
    for (const auto& microbatch : batch_plan.microbatches) {
        assert(microbatch.token_indices.size() <= 2);
        total_packaged += microbatch.token_indices.size();
    }
    assert(total_packaged == same_id_but_different_hidden.size());

    const auto transfer_buffers = router.build_transfer_buffers(same_id_but_different_hidden.data(), same_id_but_different_hidden.size(), 2);
    std::size_t transferred = 0;
    for (const auto& transfer_buffer : transfer_buffers) {
        transferred += transfer_buffer.payloads.size();
        for (const auto& payload : transfer_buffer.payloads) {
            assert(payload.hidden.size() == same_id_but_different_hidden[0].hidden.size());
            assert(payload.expert_output > 0.0F);
        }
    }
    assert(transferred == same_id_but_different_hidden.size());

    const auto transfer_batch = router.build_transfer_batch(same_id_but_different_hidden.data(), same_id_but_different_hidden.size(), 3, 2);
    assert(transfer_batch.send_counts.size() == 2);
    assert(transfer_batch.receive_counts.size() == 2);
    assert(transfer_batch.send_offsets.size() == 2);
    assert(transfer_batch.receive_offsets.size() == 2);
    assert(transfer_batch.send_buffer.size() == transfer_batch.receive_buffer.size());
    assert(transfer_batch.send_buffer.size() == same_id_but_different_hidden.size() * 3);
    assert(transfer_batch.send_offsets[0] == 0);
    assert(transfer_batch.receive_offsets[0] == 0);

    moe::ExpertLayer expert_layer(3, 4);
    for (const auto& microbatch : batch_plan.microbatches) {
        const std::size_t expert_id = static_cast<std::size_t>(router.route_token_topk(
            same_id_but_different_hidden[microbatch.token_indices.front()], 1).front().expert_id);
        const auto outputs = expert_layer.forward_batch(same_id_but_different_hidden.data(), microbatch.token_indices, expert_id);
        assert(!outputs.empty());
        assert(outputs.size() == microbatch.token_indices.size());
    }

    const auto out0 = expert_layer.forward(same_id_but_different_hidden[0], topk0[0].expert_id);
    const auto out1 = expert_layer.forward(same_id_but_different_hidden[1], topk1[0].expert_id);
    assert(out0.size() == 1);
    assert(out1.size() == 1);
    assert(out0[0] != out1[0]);

    moe::ExpertLayer execution_layer(2, 4);
    const auto execution = router.execute_remote_expert_pass(tokens.data(), tokens.size(), execution_layer, 1);
    assert(execution.outputs.size() == tokens.size());
    assert(execution.merged_outputs.size() == tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        assert(execution.merged_outputs[i] > 0.0F);
        assert(execution.outputs[i].token_index == i);
    }

    const auto distributed = router.execute_distributed_pass(tokens.data(), tokens.size(), execution_layer, 1);
    assert(distributed.by_rank.size() == 2);
    std::size_t distributed_total = 0;
    for (const auto& rank_pass : distributed.by_rank) {
        distributed_total += rank_pass.outputs.size();
    }
    assert(distributed_total == tokens.size());
    assert(distributed.merged_outputs.size() == tokens.size());

    const auto communication = router.build_communication_plan(tokens.data(), tokens.size(), 2, 0);
    assert(!communication.send_stages.empty() || !communication.receive_stages.empty());
    assert(communication.send_stages.size() == communication.receive_stages.size() ||
           communication.send_stages.size() == 2 || communication.receive_stages.size() == 2);

    const auto pipeline = router.run_pipeline(tokens.data(), tokens.size(), execution_layer, 2, 1, 0);
    assert(pipeline.route.sends.size() == tokens.size());
    assert(pipeline.dispatch.size() == 2);
    assert(pipeline.microbatches.microbatches.size() >= 1);
    assert(pipeline.execution.merged_outputs.size() == tokens.size());
    assert(pipeline.final_outputs.size() == tokens.size());

    const auto exchange = router.build_all_to_all_exchange(tokens.data(), tokens.size(), execution_layer, 1, 0);
    assert(exchange.steps.size() == tokens.size());
    std::size_t exchange_total = 0;
    for (const auto& stage : exchange.tokens_per_rank) {
        exchange_total += stage.size();
    }
    assert(exchange_total == tokens.size());
    for (const auto& stage : exchange.steps) {
        assert(stage.source_rank == 0);
        assert(stage.destination_rank >= 0 && stage.destination_rank < 2);
        assert(!stage.payload.empty());
    }

    const auto shard_plan = router.build_expert_shard_plan(tokens.data(), tokens.size(), execution_layer, 1);
    std::size_t shard_total = 0;
    for (const auto& shard : shard_plan.shards) {
        shard_total += shard.token_indices.size();
    }
    assert(shard_total == tokens.size());
    assert(!shard_plan.shards.empty());
    for (const auto& rank_tokens : shard_plan.tokens_by_rank) {
        assert(rank_tokens.size() <= tokens.size());
    }

    const auto multi_rank = router.build_multi_rank_execution_plan(tokens.data(), tokens.size(), execution_layer, 1);
    std::size_t multi_rank_total = 0;
    for (const auto& context : multi_rank.contexts) {
        multi_rank_total += context.local_token_indices.size();
    }
    assert(multi_rank_total == tokens.size());
    assert(multi_rank.merged_outputs.size() == tokens.size());
    for (const auto& context : multi_rank.contexts) {
        assert(context.local_outputs.size() == context.local_token_indices.size());
    }

    const auto runtime = router.build_distributed_runtime_plan(tokens.data(), tokens.size(), execution_layer, 1);
    std::size_t runtime_total = 0;
    for (const auto& state : runtime.per_rank) {
        runtime_total += state.token_indices.size();
        assert(state.send_buffer.size() == state.receive_buffer.size());
        assert(state.rank >= 0 && state.rank < 2);
    }
    assert(runtime_total == tokens.size());
    assert(runtime.merged_outputs.size() == tokens.size());
    assert(pipeline.runtime.per_rank.size() == 2);
    assert(pipeline.runtime.merged_outputs.size() == tokens.size());
}