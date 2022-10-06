import sys, os
sys.path.append(os.getcwd())

from fp16 import fp32_to_fp16, fp16_to_fp32
from tsc_base import obj_to_flat_aux, flat_aux_to_obj
import torch
import re


def aux_layer(args, layer, *inputs, **kwargs):
    # 解决外部调用内部layer精度不同的问题
    if args.fp16:
        return fp16_to_fp32(layer(*(fp32_to_fp16(inputs)), **kwargs))
    else:
        return layer(*inputs, **kwargs)


def get_checkpoint_forward_args(func, *args, **kwargs):
    # 生成 checkpoint 需要的 forward 方法和 args
    flat, aux = obj_to_flat_aux([args, kwargs])
    def custom_forward(*inputs):
        args, kwargs = flat_aux_to_obj(inputs, aux)
        return func(*args, **kwargs)
    return custom_forward, *flat


def distill_random_data(args, shuffle_objs: list, expand_objs: list, forward_repeat_num=0, cancel=False):
    # 随机打乱数据
    if cancel or not (hasattr(args, 'distill_random_data')\
        and hasattr(args, 'distill_random_data_n')\
        and hasattr(args, 'distill_random_data_method')) or not args.distill_random_data:
        return shuffle_objs, expand_objs
    batch_size_S = set()
    def shuffle_f(tensor: torch.Tensor, shuffle=False):
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if re.search(f'(^|,){forward_repeat_num}(,|$)', args.distill_random_data_n):
            batch_size_S.add(tensor.size(0))
            if shuffle:
                if args.distill_random_data_method == 'shuffle':
                    tensor_shuffle = tensor.clone().view(-1)
                    indexes = torch.randperm(tensor_shuffle.shape[0])
                    tensor_shuffle = tensor_shuffle[indexes].view(*tensor.shape)
                elif args.distill_random_data_method == 'sample':
                    tensor_shuffle = tensor.clone().random_(0, args.vocab_size)
                else:
                    raise NameError(f'未知的 distill_random_data_method 类型: {args.distill_random_data_method}')
            else:
                tensor_shuffle = tensor
            # 成对还是替换返回
            if args.distill_random_data == 'dual':
                return torch.cat([tensor, tensor_shuffle], dim=0)
            elif args.distill_random_data == 'replace':
                return tensor_shuffle
            else:
                raise NameError(f'未知的 distill_random_data 类型: {args.distill_random_data}')
        return tensor
    # start
    shuffled_objs, expanded_objs = [], []
    for objs, s_objs, shuffle in [
        (shuffle_objs, shuffled_objs, True),
        (expand_objs, expanded_objs, False)
    ]:
        for k, v in enumerate(objs):
            if not isinstance(v, (torch.Tensor, dict, list, tuple)):
                s_objs.append(v)
                continue
            if isinstance(v, dict):
                s_objs.append({})
                for k2, v2 in v.items():
                    s_objs[-1][k2] = shuffle_f(v2, shuffle)
            elif isinstance(v, (list, tuple)):
                s_objs.append([])
                for k2, v2 in enumerate(v):
                    s_objs[-1].append(shuffle_f(v2, shuffle))
                if isinstance(v, tuple):
                    s_objs[-1] = tuple(s_objs[-1])
            else:
                s_objs.append(shuffle_f(v, shuffle))
    assert len(batch_size_S) <= 1, f'发现了batch size不一样的tensor: {batch_size_S}'
    return shuffled_objs, expanded_objs
