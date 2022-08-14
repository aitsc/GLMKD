import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce
from utils import print_rank_0


class GLMStudent(torch.nn.Module):
    def __init__(self, language_model: GLMModel, **kwargs):
        super().__init__()
        self.origin_model = language_model

    def forward(self, *inputs, **kwargs):
        return self.origin_model(*inputs, **kwargs)

    @classmethod
    def inter_loss(cls, s_inter_vars, t_inter_vars, s_hook, t_hook, args, **kwargs):
        return 0.

    @classmethod
    def pre_loss(cls, s_logits, t_logits, loss, args, **kwargs):
        return 0.

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
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
        return super().load_state_dict(sd, strict=strict)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        for k, v in super().named_parameters(prefix=prefix, recurse=recurse):
            if k.split('.', 1)[0] == 'origin_model':
                yield k.split('.', 1)[1], v
            else:
                yield 'student.' + k, v


class TinyBERT(GLMStudent):
    show_hook = True
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model)
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

    @classmethod
    def inter_loss(cls, s_inter_vars, t_inter_vars, s_hook, t_hook, args, **kwargs):
        loss_ = 0.
        if args.finetune and (args.tinybert_ft_pre or args.tinybert_ft_hard):
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items())]

        layers_per_block = int(len(t_hook['transformer']['layers']) / len(s_hook['transformer']['layers']))
        # attentions
        if not args.tinybert_inter_final:
            student_reps = get_layer_f('s', 'attention_scores')
            teacher_reps = get_layer_f('t', 'attention_scores')
            new_teacher_reps = [teacher_reps[(i + 1) * layers_per_block - 1] for i in range(len(student_reps))]
            for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                student_rep.distill = teacher_rep.distill = True
                loss_ += F.mse_loss(student_rep, teacher_rep)
            mpu.reduce_from_model_parallel_region(loss_)
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        if args.tinybert_inter_final:
            student_reps, new_teacher_reps = [student_reps[-1]], [new_teacher_reps[-1]]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            loss_ += F.mse_loss(student_rep, teacher_rep)
        # show
        if cls.show_hook:
            print_rank_0({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                          'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),})
            cls.show_hook = False
        return loss_

    @classmethod
    def pre_loss(cls, s_logits, t_logits, loss, args, temperature=1., **kwargs):
        loss_ = 0.
        if args.finetune:
            if args.tinybert_ft_pre:
                student_likelihood = F.log_softmax(s_logits / temperature, dim=-1)
                targets_prob = F.softmax(t_logits / temperature, dim=-1)
                loss_ += (- targets_prob * student_likelihood).mean()
            if args.tinybert_ft_hard:
                loss_ += loss
        return loss_


student_model_D = {
    None: None,
    'tinybert': TinyBERT,
}
