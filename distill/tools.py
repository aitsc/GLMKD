import sys, os
sys.path.append(os.getcwd())

from fp16 import fp32_to_fp16, fp16_to_fp32
from tsc_base import obj_to_flat_aux, flat_aux_to_obj


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
