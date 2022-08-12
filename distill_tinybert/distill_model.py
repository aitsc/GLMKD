import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from pprint import pprint
from fp16 import fp16_to_fp32
from mpu import hook_model, hook_return, hook_reduce


class GLMStudent(torch.nn.Module):  # tinybert
    show_hook = True

    def __init__(self, language_model: GLMModel, args):
        super().__init__()
        self.origin_model = language_model
        self.fit_dense = torch.nn.Linear(args.hidden_size, args.teacher_hidden_size)

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None:
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            for v in hook['transformer']['layers'].values():
                inter_vars[v['layernorm_output']] = self.fit_dense(inter_vars[v['layernorm_output']])
            inter_vars[hook['transformer']['output']] = self.fit_dense(inter_vars[hook['transformer']['output']])
        return hook_return(hook, inter_vars, outputs)

    @staticmethod
    def inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, layers_per_block=2):
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items())]
        outputs = 0.
        # attentions
        student_reps = get_layer_f('s', 'attention_scores')
        teacher_reps = get_layer_f('t', 'attention_scores')
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        mpu.reduce_from_model_parallel_region(outputs)
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        # show
        if GLMStudent.show_hook:
            pprint({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                    'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),}, width=150)
            GLMStudent.show_hook = False
        return fp16_to_fp32(outputs)

    @staticmethod
    def pre_loss(s_logits, t_logits, temperature=1.):
        student_likelihood = F.log_softmax(s_logits / temperature, dim=-1)
        targets_prob = F.softmax(t_logits / temperature, dim=-1)
        return (- targets_prob * student_likelihood).mean()

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super(GLMStudent, self).state_dict(destination, prefix, keep_vars)
        sd = {}
        for k, v in state_dict.items():
            if k.split('.', 1)[0] == 'origin_model':
                sd[k.split('.', 1)[1]] = v
            else:
                sd['student.' + k] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        sd = {}
        for k, v in state_dict.items():
            if k.split('.', 1)[0] == 'student':
                sd[k.split('.', 1)[1]] = v
            else:
                sd['origin_model.' + k] = v
        return super(GLMStudent, self).load_state_dict(sd, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        for k, v in super(GLMStudent, self).named_parameters(prefix=prefix, recurse=recurse):
            if k.split('.', 1)[0] == 'origin_model':
                yield k.split('.', 1)[1], v
            else:
                yield 'student.' + k, v
