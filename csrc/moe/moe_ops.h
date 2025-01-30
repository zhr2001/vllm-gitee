#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void moe_permute_before_all2all(torch::Tensor topk_ids, int64_t num_experts,
                                torch::Tensor sorted_token_ids,
                                torch::Tensor token_cnts,
                                torch::Tensor reversed_indices);

void moe_align_block_size_during_all2all(torch::Tensor global_token_cnts,
                                         torch::Tensor global_token_cnts_cumsum,
                                         int64_t num_experts, int64_t ep_size,
                                         int64_t block_size,
                                         torch::Tensor token_block_expert_ids,
                                         torch::Tensor expert_input_token_ids);
