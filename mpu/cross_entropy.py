# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

from .initialize import get_model_parallel_group
from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):

        # Copy so the input remains unchanged.
        logits = vocab_parallel_logits.clone()
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_model_parallel_group())
        # Subtract the maximum value.  使得最大值为0, 全部负数
        logits.sub_(logits_max.unsqueeze(dim=-1))
        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)  # [16,512]; 得到exp分母
        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Get the partition's vocab indecies 获取当前进程logits的词编号范围,单进程就是[0,30592)
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_model_parallel_rank()
        world_size = get_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked). 盖住非当前进程能覆盖的词编号范围
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)  # 原版mask
        masked_target = target.clone() - vocab_start_index  # id偏移到当前进程的范围, 偏移的mask
        masked_target[target_mask] = 0  # [16,512]; 非当前进程范围的词编号全为0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = logits.view(-1, partition_vocab_size)  # 展平
        masked_target_1d = masked_target.view(-1)  # 展平
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]  # 从第二维度依次取出目标词logits
        predicted_logits = predicted_logits_1d.view_as(target)  # 还原
        predicted_logits[target_mask] = 0.0  # 非当前进程范围的词logits设置为0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits,  # 每个卡上的目标词合一, 得到真正完整的predicted_logits, 上一步设置为0的地方通常就不再是0
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits  # [16,512]

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # raise # 似乎没有影响,没走这个?
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as thier gradient.
        grad_input = softmax  # [16,512,3w/mp]
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (  # grad_input mask部分变0
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))  # 按概率将梯度分配到3w/mp个token上

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)
