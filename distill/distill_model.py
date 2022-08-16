import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce
from utils import print_rank_0


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
                ce_loss = (- targets_prob * student_likelihood).mean()
                mpu.gather_from_model_parallel_region(ce_loss)  # 确保是 parallel_output
                loss_ += ce_loss.mean()
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
    # 拆包 model 直到遇到 GLMStudent, 用于使用 GLMStudent 内部的函数
    while True:
        if hasattr(model, 'origin_model') and hasattr(model, 'get_teacher_hook'):
            return model
        if hasattr(model, 'module'):
            model = model.module
        elif hasattr(model, 'model'):
            model = model.model
        else:
            return None


def find_model_inter_var(model, name):
    # 找到模型内部的某个参数, 找不到会一直拆包
    name_L = name.split('.')
    while True:
        has_find, m = False, model
        for i, n in enumerate(name_L):
            if hasattr(m, n):
                m = getattr(m, n)
                has_find = True if i == len(name_L) - 1 else False
            else:
                break
        if has_find:
            return m
        if hasattr(model, 'module'):
            model = model.module
        elif hasattr(model, 'model'):
            model = model.model
        elif hasattr(model, 'origin_model'):
            model = model.origin_model
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


class ERDistill(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args)
        self.show_inter = True
        self.erdistill_ft_logits = self.args.erdistill_ft_logits and self.args.finetune

    def get_teacher_hook(self, **kwargs):
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers, layers_per_block))
        return {'transformer': {'layers': {} if not self.args.erdistill_inter else {
            i: {'layernorm_output': None}
        }  for i in layers},
        **({'logits_parallel': None} if self.erdistill_ft_logits else {})}

    def get_student_hook(self, **kwargs):
        return {'transformer': {'layers': {} if not self.args.erdistill_inter else {
            i: {'layernorm_output': None}
        }  for i in range(self.args.num_layers)}
        **({'logits_parallel': None} if self.erdistill_ft_logits else {})}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=None, **kwargs):
        loss_ = 0.
        if self.args.finetune and (self.args.distill_ft_soft or self.args.distill_ft_hard):
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [
                inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]
            ] + ([hook['logits_parallel']] if 'logits_parallel' in hook else [])

        t_emb_w = find_model_inter_var(t_model, 'word_embeddings.weight')
        s_emb_w = find_model_inter_var(self, 'word_embeddings.weight')
        # ER
        student_reps = get_layer_f('s', 'layernorm_output')
        teacher_reps = get_layer_f('t', 'layernorm_output')
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            mpu.copy_to_model_parallel_region(student_rep)
            mpu.copy_to_model_parallel_region(teacher_rep)
            s_logits = F.linear(student_rep, s_emb_w)
            t_logits = F.linear(teacher_reps, t_emb_w)
            student_likelihood = F.log_softmax(s_logits / self.args.distill_temperature, dim=-1)
            targets_prob = F.softmax(t_logits / self.args.distill_temperature, dim=-1)
            ce_loss = (- targets_prob * student_likelihood).mean()
            mpu.gather_from_model_parallel_region(ce_loss)
            loss_ += ce_loss.mean()
        if 'logits_parallel' in s_hook:
            s_logits = s_hook['logits_parallel']
            t_logits = t_hook['logits_parallel']
            s_logits.distill = t_logits.distill = True
            student_likelihood = F.log_softmax(s_logits / self.args.distill_temperature, dim=-1)
            targets_prob = F.softmax(t_logits / self.args.distill_temperature, dim=-1)
            ce_loss = (- targets_prob * student_likelihood).mean()
            mpu.gather_from_model_parallel_region(ce_loss)
            loss_ += ce_loss.mean()
        # show
        if self.show_inter:
            print_rank_0({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                          'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),})
            self.show_inter = False
        return loss_


student_model_D = {
    None: None,
    'tinybert': TinyBERT,
    'erdistill': ERDistill,
}
