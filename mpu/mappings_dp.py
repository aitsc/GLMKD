import torch

from .initialize import get_data_parallel_group
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the the input tensor across data parallel group."""
    group = get_data_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=group)

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    group = get_data_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Split along last dimension.
    world_size = torch.distributed.get_world_size(group=group)
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = torch.distributed.get_rank(group=group)
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""
    group = get_data_parallel_group()

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank(group=group)
    world_size = torch.distributed.get_world_size(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


class _CopyToDataParallelRegion(torch.autograd.Function):
    """Pass the input to the data parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromDataParallelRegion(torch.autograd.Function):
    """All-redcue the input from the data parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToDataParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromDataParallelRegion(torch.autograd.Function):
    """Gather the input from data parallel region and concatinate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


# -----------------
# Helper functions.
# -----------------

def copy_to_data_parallel_region(input_):
    return _CopyToDataParallelRegion.apply(input_)

def reduce_from_data_parallel_region(input_):
    return _ReduceFromDataParallelRegion.apply(input_)

def scatter_to_data_parallel_region(input_):
    return _ScatterToDataParallelRegion.apply(input_)

def gather_from_data_parallel_region(input_):
    return _GatherFromDataParallelRegion.apply(input_)
