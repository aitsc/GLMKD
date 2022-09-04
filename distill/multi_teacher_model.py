import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from utils import print_rank_0


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
                self.inter_show_hooks[f'teacher_{t_no}'] = self.inter_show_hooks['teacher']
            self.pre_loss_description.append(f'teacher_{t_no}: ' + student_model.pre_loss_description)
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
                print_rank_0(self.inter_show_hooks)
                self.show_inter = False
        else:
            raise NameError(f'未知的 op={op} !')

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        logits, loss = s_out['logits'], s_out['loss']
        loss_L = []
        self.record_and_show(student_model)

        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            loss = student_model.pre_loss(logits, t_out['logits'], loss, loss_mask=loss_mask, labels=labels)
            loss += student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels)
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
            
        self.record_and_show(student_model, op='final_show')
        return sum(loss_L) / len(loss_L)


multi_teacher_model_D = {
    None: AvgTeacher,
    'tmkd': AvgTeacher,
}
