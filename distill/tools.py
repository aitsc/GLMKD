import sys, os
sys.path.append(os.getcwd())

import mpu


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
    