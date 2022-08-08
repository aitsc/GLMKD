import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from pprint import pprint
from fp16 import fp16_to_fp32


class GLMStudent(torch.nn.Module):  # tinybert
    show_hook = True

    def __init__(self, language_model: GLMModel, args):
        super().__init__()
        self.model = language_model
        self.fit_dense = torch.nn.Linear(args.hidden_size, args.teacher_hidden_size)

    def forward(self, *inputs, **kwargs):
        logits, *mems = self.model(*inputs, **kwargs)
        if 'is_distill' in kwargs and kwargs['is_distill']:
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            inter_vars = logits
            for v in inter_vars['transformer']['layers'].values():
                v['layernorm_output'] = self.fit_dense(v['layernorm_output'])
            inter_vars['transformer']['output'] = self.fit_dense(inter_vars['transformer']['output'])
            return inter_vars, *mems 
        return logits, *mems

    @staticmethod
    def compute_loss(s_inter_vars, t_inter_vars, layers_per_block=2):
        get_layer_f = lambda iv, n: [i[1][n] for i in sorted(iv['transformer']['layers'].items())]
        outputs = 0.
        # attentions
        student_reps = get_layer_f(s_inter_vars, 'attention_scores')
        teacher_reps = get_layer_f(t_inter_vars, 'attention_scores')
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        mpu.reduce_from_model_parallel_region(outputs)
        # emb + hidden_states
        student_reps = get_layer_f(s_inter_vars, 'layernorm_output') + [s_inter_vars['transformer']['output']]
        teacher_reps = get_layer_f(t_inter_vars, 'layernorm_output') + [t_inter_vars['transformer']['output']]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        # show
        if GLMStudent.show_hook:
            print('GLMStudent.show_hook:')
            pprint({
                'student': GLMStudent.print_distill(s_inter_vars),
                'teacher': GLMStudent.print_distill(t_inter_vars),
            }, width=150)
            GLMStudent.show_hook = False
        return fp16_to_fp32(outputs)

    @staticmethod
    def print_distill(inter_vars):
        out_distill = {}
        if type(inter_vars) == dict:
            for k, v in list(inter_vars.items()):
                out = GLMStudent.print_distill(v)
                if out:
                    out_distill[k] = out
        elif hasattr(inter_vars, 'distill') and inter_vars.distill:
            return (list(inter_vars.shape), inter_vars.type())
        return out_distill
