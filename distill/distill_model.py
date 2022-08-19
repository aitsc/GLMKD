import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce
from utils import print_rank_0
import math
from tsc_base import merge_dict


class GLMStudent(torch.nn.Module):
    def __init__(self, language_model: GLMModel, args, show_pre=True, show_inter=True, summary_loss=False, **kwargs):
        super().__init__()
        self.origin_model = language_model
        self.args = args
        self.pre_loss_description = ''
        self.show_pre = show_pre
        self.show_inter = show_inter
        self.summary_writer = None
        self.summary_loss = summary_loss

    def get_teacher_hook(self, **kwargs):
        return {}

    def get_student_hook(self, **kwargs):
        return {}

    def forward(self, *inputs, **kwargs):
        return self.origin_model(*inputs, **kwargs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        # show
        if self.show_inter:
            print_rank_0({'student': hook_reduce(s_hook, s_inter_vars, filter=None),
                          'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),})
            self.show_inter = False
        return 0.

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, **kwargs):
        loss_ = 0.
        self.pre_loss_description = 'pre_loss: 0'
        T = self.args.distill_temperature
        if self.args.finetune:
            if self.args.distill_ft_soft:
                self.pre_loss_description += ' + distill_ft_soft(T%s)'%T
                student_likelihood = F.log_softmax(s_logits / T, dim=-1)
                targets_prob = F.softmax(t_logits / T, dim=-1)
                l = (- targets_prob * student_likelihood).mean()
                self.add_summary('KD_pre_loss/ft_soft', l)
                loss_ += l
            if self.args.distill_ft_hard:
                self.pre_loss_description += ' + distill_ft_hard'
                self.add_summary('KD_pre_loss/ft_hard', loss)
                loss_ += loss
        else:
            if self.args.distill_pt_soft:
                self.pre_loss_description += ' + distill_pt_soft(T%s)'%T
                loss_mask = 1. if loss_mask is None else loss_mask.view(*loss_mask.size(), 1)
                s_logits = (s_logits * loss_mask / T).view(-1, s_logits.size(-1))
                t_logits = (t_logits * loss_mask / T).view(-1, t_logits.size(-1))
                kl_loss = F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction="batchmean")
                l += mpu.gather_from_model_parallel_region(kl_loss).mean() * T ** 2
                self.add_summary('KD_pre_loss/pt_soft', l)
                loss_ += l
            if self.args.distill_pt_hard:
                self.pre_loss_description += ' + distill_pt_hard'
                self.add_summary('KD_pre_loss/pt_hard', loss)
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

    def add_summary(self, name, value):
        if self.summary_writer is None or not self.summary_loss:
            return False
        if self.args.iteration % self.args.log_interval == 0:
            value = value.item() if hasattr(value, 'item') else value
            self.summary_writer.add_scalar(name, value, self.args.iteration)
            return True
        else:
            return False


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
        super().__init__(language_model, args, **kwargs)
        self.fit_dense = torch.nn.Linear(args.hidden_size, args.teacher_hidden_size)

    def get_teacher_hook(self, **kwargs):
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        return {} if self.args.tinybert_wo_inter else {'transformer': {
            'layers': {} if self.args.tinybert_inter_final else {
                **{i: {'layernorm_output': None} for i in layers[:-1]},
                **{i - 1: {'attention_scores': None} for i in layers[1:]},
            },
            'output': None,
        }}

    def get_student_hook(self, **kwargs):
        return {} if self.args.tinybert_wo_inter else {'transformer': {
            'layers': {} if self.args.tinybert_inter_final else {i: {
                'layernorm_output': None, 'attention_scores': None,
            } for i in range(self.args.num_layers)},
            'output': None,
        }}

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None and not self.args.tinybert_wo_inter:
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            for v in hook['transformer']['layers'].values():
                inter_vars[v['layernorm_output']] = self.fit_dense(inter_vars[v['layernorm_output']])
            inter_vars[hook['transformer']['output']] = self.fit_dense(inter_vars[hook['transformer']['output']])
        return hook_return(hook, inter_vars, outputs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if self.args.tinybert_wo_inter or len(s_inter_vars) == 0:
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
            loss_ += mpu.reduce_from_model_parallel_region(loss_)
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            student_rep.distill = teacher_rep.distill = True
            loss_ += F.mse_loss(student_rep, teacher_rep)
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_


class MiniLMv2(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

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
        if len(s_inter_vars) == 0:
            return loss_
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
            kl_loss = kl_loss / t_rep.size(0) / t_rep.size(1) / t_rep.size(2)
            loss_ += mpu.gather_from_model_parallel_region(kl_loss).mean()
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_


class MiniLM(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        return {'transformer': {'layers': {self.args.teacher_num_layers - 1: {
                'attention_probs': None, 'value_layer': None
        }}}}

    def get_student_hook(self, **kwargs):
        return {'transformer': {'layers': {self.args.num_layers - 1: {
                'attention_probs': None, 'value_layer': None
        }}}}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        s_a = s_inter_vars[s_hook['transformer']['layers'][self.args.num_layers - 1]['attention_probs']]
        s_v = s_inter_vars[s_hook['transformer']['layers'][self.args.num_layers - 1]['value_layer']]
        t_a = t_inter_vars[t_hook['transformer']['layers'][self.args.teacher_num_layers - 1]['attention_probs']]
        t_v = t_inter_vars[t_hook['transformer']['layers'][self.args.teacher_num_layers - 1]['value_layer']]
        s_a.distill = s_v.distill = t_a.distill = t_v.distill = True
        s_v2 = torch.matmul(s_v, s_v.transpose(-1,-2)) / math.sqrt(s_v.size(-1))
        t_v2 = torch.matmul(t_v, t_v.transpose(-1,-2)) / math.sqrt(t_v.size(-1))
        kl_loss = F.kl_div(F.log_softmax(s_a, dim=-1), F.softmax(t_a, dim=-1), reduction="sum")
        kl_loss += F.kl_div(F.log_softmax(s_v2, dim=-1), F.softmax(t_v2, dim=-1), reduction="sum")
        kl_loss = kl_loss / s_a.size(0) / s_a.size(1) / s_a.size(2)
        loss_ += mpu.gather_from_model_parallel_region(kl_loss).mean()
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_


class DistilBERT(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        return {'transformer': {'output': None}}

    def get_student_hook(self, **kwargs):
        return {'transformer': {'output': None}}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, loss_mask=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        if self.args.distilbert_alpha_cos <= 0.:
            return loss_
        s_o = s_inter_vars[s_hook['transformer']['output']]
        t_o = t_inter_vars[t_hook['transformer']['output']]
        assert s_o.size() == t_o.size(), f'{s_o.size()} == {t_o.size()}'
        s_o.distill = t_o.distill = True
        loss_mask = loss_mask.view(*loss_mask.size(), 1)
        s_o = (s_o * loss_mask).view(-1, s_o.size(-1))
        t_o = (t_o * loss_mask).view(-1, t_o.size(-1))
        target = s_o.new(s_o.size(0)).fill_(1)
        loss_ += F.cosine_embedding_loss(s_o, t_o, target) * self.args.distilbert_alpha_cos
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, **kwargs):
        loss_ = 0.
        if self.args.finetune:
            return super().pre_loss(s_logits, t_logits, loss)
        self.pre_loss_description = 'pre_loss: 0'
        if self.args.distilbert_alpha_ce > 0:
            self.pre_loss_description += ' + distilbert_alpha_ce'
            loss_mask = loss_mask.view(*loss_mask.size(), 1)
            T = self.args.distill_temperature
            s_logits = (s_logits * loss_mask / T).view(-1, s_logits.size(-1))
            t_logits = (t_logits * loss_mask / T).view(-1, t_logits.size(-1))
            kl_loss = F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction="batchmean")
            kl_loss = mpu.gather_from_model_parallel_region(kl_loss)
            loss_ += kl_loss.mean() * T ** 2 * self.args.distilbert_alpha_ce
        if self.args.distilbert_alpha_mlm > 0:
            self.pre_loss_description += ' + distilbert_alpha_mlm'
            loss_ += loss * self.args.distilbert_alpha_mlm
        # show
        if self.show_pre:
            print_rank_0(self.pre_loss_description)
            self.show_pre = False
        return loss_


class ERDistill(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        self.summary_loss = True

    def get_inter_hook(self, layers, st='s'):
        # erdistill_inter: all / one / two / 1plus /
        all = [(i, {'layernorm_output': None}) for i in layers]
        if self.args.erdistill_inter in {'one', '', None}:
            all = []
        elif self.args.erdistill_inter in {'two'}:
            all = all[-1:]
        elif self.args.erdistill_inter in {'1plus'}:
            all = all[-1:] if st=='s' else []
        return {'transformer': {'layers': dict(all), 
            **({'output': None} if self.args.erdistill_inter else {})}}

    def get_teacher_hook(self, **kwargs):
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers, layers_per_block))
        return self.get_inter_hook(layers, st='t')

    def get_student_hook(self, **kwargs):
        layers = tuple(i for i in self.args.num_layers)
        return self.get_inter_hook(layers, st='s')

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[s_hook['transformer']['output']]]
        # 局部相似
        t_reps = []
        for t_rep in teacher_reps:
            t_rep.distill = True
            t_rep = torch.matmul(t_rep, t_rep.transpose(-1,-2)) / math.sqrt(t_rep.size(-1))
            t_rep = t_rep if self.args.erdistill_inter_mse else F.softmax(t_rep, dim=-1)
            t_reps.append(t_rep)
        if self.args.erdistill_inter in {'1plus'}:
            t_reps += t_reps
        for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
            s_rep.distill = True
            s_rep = torch.matmul(s_rep, s_rep.transpose(-1,-2)) / math.sqrt(s_rep.size(-1))
            if self.args.erdistill_inter_mse:
                l = F.mse_loss(s_rep, t_rep)
            else:
                l = F.kl_div(F.log_softmax(s_rep, dim=-1), F.softmax(t_rep, dim=-1), reduction="sum")
                l = l / t_rep.size(0) / t_rep.size(1) / t_rep.size(2)
            super().add_summary(f'inter_loss/local.{i}', l)
            loss_ += l
        # 全局相似
        t_emb_w = find_model_inter_var(t_model, 'word_embeddings.weight')
        s_emb_w = find_model_inter_var(self, 'word_embeddings.weight')
        t_reps = []
        for i, t_rep in enumerate(teacher_reps):
            t_rep = mpu.copy_to_model_parallel_region(t_rep)
            t_rep = F.linear(t_rep, t_emb_w)
            t_rep = t_rep if self.args.erdistill_inter_mse else F.softmax(t_rep, dim=-1)
            t_reps.append(t_rep)
        if self.args.erdistill_inter in {'1plus'}:
            t_reps += t_reps
        for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
            s_rep = mpu.copy_to_model_parallel_region(s_rep)
            s_logits = F.linear(s_rep, s_emb_w)
            if self.args.erdistill_inter_mse:
                l = F.mse_loss(s_rep, t_rep)
            else:
                l = F.kl_div(F.log_softmax(s_logits, dim=-1), t_rep, reduction="sum")
                l = l / t_rep.size(0) / t_rep.size(1) / t_rep.size(2)
            l = mpu.gather_from_model_parallel_region(l).mean()
            super().add_summary(f'inter_loss/global.{i}', l)
            loss_ += l
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_


class MixBaseline(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        self.inter_bl = ['TinyBERT', 'MiniLMv2', 'MiniLM', 'DistilBERT']
        # 支持 pre_loss hard 的模型必须放在最后, 保证只算一次
        self.pre_bl_pretrain_soft = ['DistilBERT', 'TinyBERT']  # 重复的预训练软标签构建方式不需要
        self.pre_bl_finetune_soft = ['TinyBERT']  # 默认包含 KD(super())
        self.baselines = set(self.inter_bl + self.pre_bl_pretrain_soft + self.pre_bl_finetune_soft)
        for c in self.baselines:
            setattr(self, c, eval(c)(language_model, args, show_pre=False, show_inter=False))
        self.summary_loss = True

    def get_teacher_hook(self, **kwargs):
        if self.args.mixbaseline_wo_inter:
            return {}
        hooks = [getattr(self, c).get_teacher_hook() for c in self.inter_bl]
        return merge_dict(hooks)

    def get_student_hook(self, **kwargs):
        if self.args.mixbaseline_wo_inter:
            return {}
        hooks = [getattr(self, c).get_student_hook() for c in self.inter_bl]
        return merge_dict(hooks)

    def forward(self, *inputs, **kwargs):
        if 'TinyBERT' in self.baselines:
            return self.TinyBERT(*inputs, **kwargs)
        else:
            return super().forward(*inputs, **kwargs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        for c in self.inter_bl:
            distill_temperature = self.args.distill_temperature
            if hasattr(self.args, f'mixbaseline_{c.lower()}_t'):
                self.args.distill_temperature = getattr(self.args, f'mixbaseline_{c.lower()}_t')
            l = getattr(self, c).inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs)
            super().add_summary(f'inter_loss/{c}', l)
            loss_ += l
            self.args.distill_temperature = distill_temperature
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_ = 0.
        show_pre = self.show_pre
        self.show_pre = False
        distill_ft_hard, distill_pt_hard = self.args.distill_ft_hard, self.args.distill_pt_hard
        self.args.distill_ft_hard = self.args.distill_pt_hard = False  # 硬标签只需要算一次
        pre_loss_description = ['all pre_loss:']
        # KD pre_loss
        if self.args.finetune:
            l += super().pre_loss(s_logits, t_logits, loss, **kwargs)
            super().add_summary(f'pre_loss/KD', l)
            loss_ += l
            pre_loss_description.append(f'\tKD - {self.pre_loss_description}')
        # other pre_loss
        pre_bl = self.pre_bl_finetune_soft if self.args.finetune else self.pre_bl_pretrain_soft
        for i, c in enumerate(pre_bl):
            if i == len(pre_bl) - 1:
                self.args.distill_ft_hard, self.args.distill_pt_hard = distill_ft_hard, distill_pt_hard
            distill_temperature = self.args.distill_temperature
            if hasattr(self.args, f'mixbaseline_{c.lower()}_t'):
                self.args.distill_temperature = getattr(self.args, f'mixbaseline_{c.lower()}_t')
            l = getattr(self, c).pre_loss(s_logits, t_logits, loss, **kwargs)
            super().add_summary(f'pre_loss/{c}', l)
            loss_ += l
            pre_loss_description.append(f'\t{c} - {self.pre_loss_description}')
            self.args.distill_temperature = distill_temperature
        # show
        if show_pre:
            print_rank_0('\n'.join(pre_loss_description))
        return loss_


student_model_D = {
    None: None,
    'kd': GLMStudent,
    'tinybert': TinyBERT,
    'erdistill': ERDistill,
    'minilmv2': MiniLMv2,
    'minilm': MiniLM,
    'distilbert': DistilBERT,
    'mixbaseline': MixBaseline,
}
