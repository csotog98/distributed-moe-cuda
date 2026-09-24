#include "token_router.hpp"

#include <cmath>
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
        const float energy = compute_hidden_energy(token.hidden);
        const std::size_t expert = static_cast<std::size_t>(std::floor(energy)) % expert_count_;
        const int destination = static_cast<int>(expert % total_gpus_);
        const std::size_t position = plan.sends.size();

        plan.sends.push_back(RoutedToken{
            token.id, token_index, static_cast<std::uint32_t>(expert), destination});
        plan.by_rank[static_cast<std::size_t>(destination)].push_back(position);
    }
    return plan;
}

}  // namespace moe