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

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        logits, loss = s_out['logits'], s_out['loss']
        loss_L = []
        inter_show_hooks = {}
        student_model.show_inter = False
        summary_suffix = student_model.summary_suffix

        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            student_model.summary_suffix = summary_suffix + f'_t{i}'
            loss = student_model.pre_loss(logits, t_out['logits'], loss, loss_mask=loss_mask, labels=labels)
            loss += student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels)
            loss_L.append(loss)
            inter_show_hooks.update(student_model.inter_show_hooks)
            if 'teacher' in inter_show_hooks:
                inter_show_hooks[f'teacher_{i}'] = inter_show_hooks['teacher']
            student_model.add_summary(f'teacher_loss/teacher_{i}', loss)
            
        student_model.summary_suffix = summary_suffix  # 复原
        if self.show_inter:
            del inter_show_hooks['teacher']
            print_rank_0(inter_show_hooks)
            self.show_inter = False
        return sum(loss_L) / len(loss_L)


multi_teacher_model_D = {
    None: AvgTeacher,
    'tmkd': AvgTeacher,
}
