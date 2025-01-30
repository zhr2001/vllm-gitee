from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_moe_ep_group, get_moe_tp_group, get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def gather_along_first_dim(input_: torch.Tensor, output_splits=None) -> torch.Tensor:
    return get_tp_group().all_gather(input_, 0, output_split_sizes=output_splits)


def split_along_first_dim(
    input_: torch.Tensor,
) -> torch.Tensor:
    return get_tp_group().split_along_first_dim(input_)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


def moe_tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input_tensor across moe model parallel group."""
    return get_moe_tp_group().all_reduce(input_)


def moe_tensor_model_parallel_all_gather(input_: torch.Tensor) -> torch.Tensor:
    """Gather the input tensor across moe model parallel group."""
    return get_moe_tp_group().all_gather(input_)


def moe_split_along_first_dim(input_: torch.Tensor, split_sizes) -> torch.Tensor:
    return get_moe_tp_group().split_along_first_dim(input_, split_sizes)


def moe_gather_along_first_dim(
    input_: torch.Tensor, output_splits=None
) -> torch.Tensor:
    return get_moe_tp_group().all_gather(input_, 0, output_split_sizes=output_splits)


def moe_expert_model_parallel_all_to_all(
    input_: torch.Tensor, input_split_sizes: List[int], output_split_sizes: List[int]
) -> torch.Tensor:
    "All-to-All the input tensor across moe expert model parallel group."
    return get_moe_ep_group().all_to_all(input_, input_split_sizes, output_split_sizes)
