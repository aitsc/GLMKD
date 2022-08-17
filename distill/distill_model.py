import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce
from utils import print_rank_0
import math


class GLMStudent(torch.nn.Module):
    def __init__(self, language_model: GLMModel, args, **kwargs):
        super().__init__()
        self.origin_model = language_model
        self.args = args
        self.pre_loss_description = ''
        self.show_pre = True

    def get_teacher_hook(self, **kwargs):
        return {}

    def get_student_hook(self, **kwargs):
        return {}

    def forward(self, *inputs, **kwargs):
        return self.origin_model(*inputs, **kwargs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        return 0.

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_ = 0.
        self.pre_loss_description = 'pre_loss: 0'
        if self.args.finetune:
            if self.args.distill_ft_soft:
                self.pre_loss_description += ' + distill_ft_soft'
                student_likelihood = F.log_softmax(s_logits / self.args.distill_temperature, dim=-1)
                targets_prob = F.softmax(t_logits / self.args.distill_temperature, dim=-1)
                loss_ += (- targets_prob * student_likelihood).mean()
            if self.args.distill_ft_hard:
                self.pre_loss_description += ' + distill_ft_hard'
                loss_ += loss
        else:
            if self.args.distill_pt_soft:
                self.pre_loss_description += ' + distill_pt_soft'
                student_likelihood = F.log_softmax(s_logits / self.args.distill_temperature, dim=-1)
                targets_prob = F.softmax(t_logits / self.args.distill_temperature, dim=-1)
                loss_ += (- targets_prob * student_likelihood).mean()
            if self.args.distill_pt_hard:
                self.pre_loss_description += ' + distill_pt_hard'
                loss_ += loss
        # show
        if self.show_pre:
            print_rank_0(self.pre_loss_description)
            self.show_pre = False
        return loss_

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


def unpacking_student_model(model):
    while True:
        if hasattr(model, 'origin_model') and hasattr(model, 'get_teacher_hook'):
            return model
        if hasattr(model, 'module'):
            model = model.module
        elif hasattr(model, 'model'):
            model = model.model
        else:
            return None


class TinyBERT(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args)
        self.fit_dense = torch.nn.Linear(args.hidden_size, args.teacher_hidden_size)
        self.show_inter = True

    def get_teacher_hook(self, **kwargs):
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        return {'transformer': {
            'layers': {} if self.args.tinybert_inter_final else {
                **{i: {'layernorm_output': None} for i in layers[:-1]},
                **{i - 1: {'attention_scores': None} for i in layers[1:]},
            },
            'output': None,
        }}

    def get_student_hook(self, **kwargs):
        return {'transformer': {
            'layers': {} if self.args.tinybert_inter_final else {i: {
                'layernorm_output': None, 'attention_scores': None,
            } for i in range(self.args.num_layers)},
            'output': None,
        }}

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None:
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            for v in hook['transformer']['layers'].values():
                inter_vars[v['layernorm_output']] = self.fit_dense(inter_vars[v['layernorm_output']])
            inter_vars[hook['transformer']['output']] = self.fit_dense(inter_vars[hook['transformer']['output']])
        return hook_return(hook, inter_vars, outputs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if self.args.finetune and (self.args.distill_ft_soft or self.args.distill_ft_hard):
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        # attentions
        if not self.args.tinybert_inter_final:
            student_reps = get_layer_f('s', 'attention_scores')
            teacher_reps = get_layer_f('t', 'attention_scores')
            for student_rep, teacher_rep in zip(student_reps, teacher_reps):
                student_rep.distill = teacher_rep.distill = True
                loss_ += F.mse_loss(student_rep, teacher_rep)
            mpu.reduce_from_model_parallel_region(loss_)
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            loss_ += F.mse_loss(student_rep, teacher_rep)
        # show
        if self.show_inter:
            print_rank_0({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                          'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),})
            self.show_inter = False
        return loss_


class MiniLMv2(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args)
        self.show_inter = True

    def get_teacher_hook(self, **kwargs):
        return {'transformer': {'layers': {self.args.minilmv2_teacher_layer - 1: {
                'mixed_query_layer': None, 'mixed_key_layer': None, 'mixed_value_layer': None
        }}}}

    def get_student_hook(self, **kwargs):
        return {'transformer': {'layers': {self.args.num_layers - 1: {
                'mixed_query_layer': None, 'mixed_key_layer': None, 'mixed_value_layer': None
        }}}}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        s_qkv, t_qkv = [], []
        for i in ['mixed_query_layer', 'mixed_key_layer', 'mixed_value_layer']:
            s_qkv.append(s_inter_vars[s_hook['transformer']['layers'][self.args.num_layers - 1][i]])
            t_qkv.append(t_inter_vars[t_hook['transformer']['layers'][self.args.minilmv2_teacher_layer - 1][i]])
        n_heads = int(self.args.minilmv2_relation_heads / mpu.get_model_parallel_world_size())
        # q k v
        for s_rep, t_rep in zip(s_qkv, t_qkv):
            s_rep.distill = t_rep.distill = True
            s_rep = s_rep.view(*s_rep.size()[:-1], n_heads, -1).permute(0, 2, 1, 3)
            s_rep = torch.matmul(s_rep, s_rep.transpose(-1,-2)) / math.sqrt(s_rep.size(-1))
            t_rep = t_rep.view(*t_rep.size()[:-1], n_heads, -1).permute(0, 2, 1, 3)
            t_rep = torch.matmul(t_rep, t_rep.transpose(-1,-2)) / math.sqrt(t_rep.size(-1))
            kl_loss = F.kl_div(F.log_softmax(s_rep, dim=-1), F.softmax(t_rep, dim=-1), reduction="sum")
            kl_loss = kl_loss / t_rep.size(0) / t_rep.size(1)
            mpu.gather_from_model_parallel_region(kl_loss)
            loss_ += kl_loss.mean()
        # show
        if self.show_inter:
            print_rank_0({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                          'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),})
            self.show_inter = False
        return loss_


student_model_D = {
    None: None,
    'tinybert': TinyBERT,
    'minilmv2': MiniLMv2,
}
