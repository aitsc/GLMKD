import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from utils import print_rank_0
import mpu
from fp16 import fp32_to_fp16, fp16_to_fp32


class AvgTeacher(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.show_inter = True
        self.show_pre = True

    def record_and_show(self, student_model, op='init', t_no=-1, loss=0):
        # 初始化显示和记录的参数
        if op == 'init':
            self.inter_show_hooks = {}
            self.pre_loss_description = []
            student_model.show_inter = False
            student_model.show_pre = False
            self.summary_suffix_s = student_model.summary_suffix
        # 计算单个教师模型结果之前的准备
        elif op == 't_start':
            student_model.summary_suffix = self.summary_suffix_s + f'_t{t_no}'
        # 计算完教师模型结果之后的处理
        elif op == 't_end':
            # 记录loss联合方式
            self.inter_show_hooks.update(student_model.inter_show_hooks)
            if 'teacher' in self.inter_show_hooks:
                self.inter_show_hooks[f't_{t_no}'] = self.inter_show_hooks['teacher']
            if 'student' in self.inter_show_hooks:
                self.inter_show_hooks[f's_{t_no}'] = self.inter_show_hooks['student']
            self.pre_loss_description.append(f's-t_{t_no}: ' + student_model.pre_loss_description)
            # 记录 tensorboard
            student_model.add_summary(f'teacher_loss/teacher_{t_no}', loss)
            student_model.summary_suffix = self.summary_suffix_s  # 复原
        # 最终处理展示
        elif op == 'final_show':
            if self.show_pre:
                print_rank_0('\n'.join(self.pre_loss_description))
                self.show_pre = False
            if self.show_inter:
                if 'teacher' in self.inter_show_hooks:
                    del self.inter_show_hooks['teacher']
                if 'student' in self.inter_show_hooks:
                    del self.inter_show_hooks['student']
                print_rank_0(self.inter_show_hooks)
                self.show_inter = False
        else:
            raise NameError(f'未知的 op={op} !')

    def aux_layer(self, layer, *inputs, **kwargs):
        # 解决外部调用内部layer精度不同的问题
        if self.args.fp16:
            return fp16_to_fp32(layer(*(fp32_to_fp16(inputs)), **kwargs))
        else:
            return layer(*inputs, **kwargs)

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_out['loss'], loss_mask=loss_mask, labels=labels)
            loss += student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels)
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        return sum(loss_L) / len(loss_L)


class MT_BERT(AvgTeacher):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        if args.mt_hidden_size:  # W
            mt_hidden_size = [int(i) for i in args.mt_hidden_size.split(':')]
            for i, hidden_size in enumerate(mt_hidden_size):
                setattr(self, f'fit_dense_{i}', mpu.ColumnParallelLinear(args.hidden_size, hidden_size))

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            inter_vars = s_inter_vars.copy()  # 不修改 s_inter_vars
            # 学生中间层 W
            if self.args.mt_hidden_size and 'transformer' in s_hook and 'layers' in s_hook['transformer']:
                fit_dense = getattr(self, f'fit_dense_{i}')
                for v in s_hook['transformer']['layers'].values():
                    if 'layernorm_output' in v:
                        inter_vars[v['layernorm_output']] = self.aux_layer(fit_dense, inter_vars[v['layernorm_output']])
                if 'output' in s_hook['transformer']:
                    inter_vars[s_hook['transformer']['output']] = self.aux_layer(fit_dense, inter_vars[s_hook['transformer']['output']])
            # loss
            pre_loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_out['loss_batch'], loss_mask=loss_mask, labels=labels, keep_batch=True)
            pre_loss *= 1 / (1 + t_out['loss_batch'])  # 加权, 依赖参数 --mt_has_loss
            inter_loss = student_model.inter_loss(inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels)
            loss = pre_loss.mean() + inter_loss
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        return sum(loss_L) + s_out['loss']  # 这里加入了硬标签, pre_loss 不应再有硬标签参数


multi_teacher_model_D = {
    None: AvgTeacher,
    '': AvgTeacher,
    'tmkd': AvgTeacher,
    'mt_bert': MT_BERT,
}
