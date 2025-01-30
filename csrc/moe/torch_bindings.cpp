#include "core/registration.h"
#include "moe_ops.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor! input, Tensor output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  m.def(
      "moe_permute_before_all2all(Tensor topk_ids, int num_experts,"
      "                           Tensor sorted_token_ids, Tensor token_cnts,"
      "                           Tensor reversed_indices) -> ()");
  m.impl("moe_permute_before_all2all", torch::kCUDA,
         &moe_permute_before_all2all);

  m.def(
      "moe_align_block_size_during_all2all(Tensor global_token_cnts,"
      "                                    Tensor global_token_cnts_cumsum,"
      "                                    int num_experts, int ep_size,"
      "                                    int block_size,"
      "                                    Tensor token_block_expert_ids, "
      "                                    Tensor expert_input_token_ids) -> ()");
  m.impl("moe_align_block_size_during_all2all", torch::kCUDA,
         &moe_align_block_size_during_all2all);

#ifndef USE_ROCM
  m.def(
      "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
      "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
      "int b_q_type, SymInt size_m, "
      "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
      "topk, "
      "int moe_block_size, bool replicate_input, bool apply_weights)"
      " -> Tensor");
  // conditionally compiled so impl registration is in source file
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
