import torch.nn.functional as F
import mpu
from mpu import parallel_cross_entropy, parallel_info_entropy, parallel_relative_entropy


class CustomLoss:
    args = None
    
    @staticmethod
    def inputs_handling(*inputs, parallel='reduce/gather/', input_mask=None, **kwargs):
        # 前向 mask
        if input_mask is not None:
            inputs_ = []
            for i in inputs:
                if isinstance(input_mask, (int, float)):
                    if input_mask == 1:
                        inputs_.append(i)
                        continue
                elif len(i.shape) == len(input_mask.shape) + 1:
                    input_mask = input_mask.unsqueeze(dim=-1)
                elif len(i.shape) + 1 == len(input_mask.shape):
                    input_mask = input_mask.squeeze(dim=-1)
                inputs_.append(i * input_mask)
            inputs = inputs_
        # 模型并行
        inputs_ = []
        for i in inputs:
            if parallel == 'reduce' and mpu.get_model_parallel_world_size() > 1:
                i = mpu.reduce_from_model_parallel_region(i)
            inputs_.append(i)
        return inputs_

    @staticmethod
    def loss_handling(loss, keep_batch=False, parallel='reduce/gather/', mask=None, last_dim_avg=True, **kwargs):
        # loss mask
        if mask is not None:
            if isinstance(mask, (int, float)):
                if mask == 1:
                    mask = None
            elif len(loss.shape) == len(mask.shape) + 1:
                mask = mask.unsqueeze(dim=-1)
            elif len(loss.shape) + 1 == len(mask.shape):
                mask = mask.squeeze(dim=-1)
            if mask is not None:
                loss = loss * mask
        # 平均张量, 可以保持 batch (第0个维度) 不被平均
        if keep_batch:
            if len(loss.shape) > 1:
                ret = loss.mean(list(range(1, len(loss.shape))))
            else:
                ret = loss
        else:
            ret = loss.mean()
        # 模型并行
        if parallel == 'gather' and mpu.get_model_parallel_world_size() > 1:  # 最后一维的拼接
            ret = mpu.reduce_from_model_parallel_region(ret)
            if last_dim_avg:  # 适合平均型方法, 最后一维累加型不需要除法
                ret = ret / mpu.get_model_parallel_world_size()
        return ret

    @staticmethod
    def gather_inputs(*inputs, parallel='gather'):
        if parallel == 'gather':
            if mpu.get_model_parallel_world_size() > 1:
                for i in range(len(inputs)):
                    inputs[i] = mpu.gather_from_model_parallel_region(inputs[i])
            parallel = ''
        return inputs, parallel

    @classmethod
    def mse_loss(cls, input, target, **kwargs):
        input, target = cls.inputs_handling(input, target, **kwargs)
        loss = F.mse_loss(input, target, reduction='none')
        return cls.loss_handling(loss, **kwargs)

    @classmethod
    def kl_div(cls, input, target, parallel='', **kwargs):
        input, target = cls.inputs_handling(input, target, parallel=parallel, **kwargs)
        if getattr(cls.args, 'disable_parallel_entropy', 0):
            (input, target), parallel = cls.gather_inputs(input, target, parallel=parallel)
        if parallel == 'gather' and mpu.get_model_parallel_world_size() > 1:
            loss = parallel_relative_entropy(input, target)
        else:
            student_likelihood = F.log_softmax(input, dim=-1)
            targets_prob = F.softmax(target, dim=-1)
            loss = F.kl_div(student_likelihood, targets_prob, reduction="none").sum(-1)
        return cls.loss_handling(loss, **kwargs)

    @classmethod
    def cross_entropy(cls, input, target, parallel='', **kwargs):
        input, target = cls.inputs_handling(input, target, parallel=parallel, **kwargs)
        if getattr(cls.args, 'disable_parallel_entropy', 0):
            (input, target), parallel = cls.gather_inputs(input, target, parallel=parallel)
        if parallel == 'gather' and mpu.get_model_parallel_world_size() > 1:
            loss = parallel_cross_entropy(input, target)
        else:
            loss = (- F.softmax(target, dim=-1) * F.log_softmax(input, dim=-1)).sum(-1)
        return cls.loss_handling(loss, **kwargs)

    @classmethod
    def info_entropy(cls, input, parallel='', **kwargs):
        input = cls.inputs_handling(input, parallel=parallel, **kwargs)[0]
        if getattr(cls.args, 'disable_parallel_entropy', 0):
            (input,), parallel = cls.gather_inputs(input, parallel=parallel)
        if parallel == 'gather' and mpu.get_model_parallel_world_size() > 1:
            loss = parallel_info_entropy(input)
        else:
            loss = (- F.softmax(input, dim=-1) * F.log_softmax(input, dim=-1)).sum(-1)
        return cls.loss_handling(loss, **kwargs)

    @classmethod
    def cos_distance(cls, input1, input2, parallel='', **kwargs):
        input1, input2 = cls.inputs_handling(input1, input2, parallel=parallel, **kwargs)
        target = input1.new(input1.size(0)).fill_(1)
        # gather 模型并行存在分母问题有待实现
        (input1, input2), parallel = cls.gather_inputs(input1, input2, parallel=parallel)
        loss = F.cosine_embedding_loss(input1, input2, target, reduction='none')
        kwargs['last_dim_avg'] = False
        return cls.loss_handling(loss, parallel=parallel, **kwargs)
