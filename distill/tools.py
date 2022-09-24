import sys, os
sys.path.append(os.getcwd())

import mpu
from fp16 import fp32_to_fp16, fp16_to_fp32
from tsc_base import obj_to_flat_aux, flat_aux_to_obj


def all_mean_custom(tensor, keep_batch=False, reduce=False):
    # 平均张量, 可以保持 batch (第0个维度) 不被平均
    if keep_batch:
        if len(tensor.shape) > 1:
            ret = tensor.mean(list(range(1, len(tensor.shape))))
        else:
            ret = tensor
    else:
        ret = tensor.mean()
    if reduce:
        ret = mpu.reduce_from_model_parallel_region(ret) / mpu.get_model_parallel_world_size()
    return ret
    

def aux_layer(args, layer, *inputs, **kwargs):
    # 解决外部调用内部layer精度不同的问题
    if args.fp16:
        return fp16_to_fp32(layer(*(fp32_to_fp16(inputs)), **kwargs))
    else:
        return layer(*inputs, **kwargs)


def get_checkpoint_forward_args(func, *args, **kwargs):
    flat, aux = obj_to_flat_aux([args, kwargs])
    def custom_forward(*inputs):
        args, kwargs = flat_aux_to_obj(inputs, aux)
        return func(*args, **kwargs)
    return custom_forward, *flat
