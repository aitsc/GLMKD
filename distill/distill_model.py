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
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from fp16 import fp32_to_fp16, fp16_to_fp32


class GLMStudent(torch.nn.Module):
    def __init__(self, language_model: GLMModel, args, show_pre=True, show_inter=True, summary_loss=True, **kwargs):
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

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, return_dict=False, **kwargs):
        loss_ = 0.
        self.pre_loss_description = 'pre_loss: 0'
        T = self.args.distill_temperature
        if self.args.distill_wo_loss_mask:
            loss_mask = None
        loss_D = {}
        if self.args.finetune:
            if self.args.distill_ft_soft:
                self.pre_loss_description += ' + distill_ft_soft(T%s)'%T
                if loss_mask is None:
                    loss_mask = 1.
                    self.pre_loss_description += '/wom'
                else:  # 可用于 seq2seq_forward_step
                    loss_mask = loss_mask.view(*loss_mask.size(), 1)
                    self.pre_loss_description += '/mask'
                student_likelihood = F.log_softmax(s_logits * loss_mask / T, dim=-1).view(-1, s_logits.size(-1))
                targets_prob = F.softmax(t_logits * loss_mask / T, dim=-1).view(-1, t_logits.size(-1))
                if self.args.distill_ft_soft_kl:
                    l = F.kl_div(student_likelihood, targets_prob, reduction="batchmean") * T ** 2
                else:
                    l = (- targets_prob * student_likelihood).mean()
                l = mpu.gather_from_model_parallel_region(l).mean()
                self.add_summary('pre_loss/ft_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_ft_hard:
                self.pre_loss_description += ' + distill_ft_hard'
                self.add_summary('pre_loss/ft_hard', loss)
                loss_ += loss
                loss_D['hard'] = l
        else:
            if self.args.distill_pt_soft:
                self.pre_loss_description += ' + distill_pt_soft(T%s)'%T
                if loss_mask is None:
                    loss_mask = 1.
                    self.pre_loss_description += '/wom'
                else:
                    loss_mask = loss_mask.view(*loss_mask.size(), 1)
                    self.pre_loss_description += '/mask'
                student_likelihood = F.log_softmax(s_logits * loss_mask / T, dim=-1).view(-1, s_logits.size(-1))
                targets_prob = F.softmax(t_logits * loss_mask / T, dim=-1).view(-1, t_logits.size(-1))
                if self.args.distill_pt_soft_ce:
                    l = (- targets_prob * student_likelihood).mean()
                else:
                    l = F.kl_div(student_likelihood, targets_prob, reduction="batchmean") * T ** 2
                l = mpu.gather_from_model_parallel_region(l).mean()
                self.add_summary('pre_loss/pt_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_pt_hard:
                self.pre_loss_description += ' + distill_pt_hard'
                self.add_summary('pre_loss/pt_hard', loss)
                loss_ += loss
                loss_D['hard'] = l
        loss_D['loss'] = loss_
        # show
        if self.show_pre:
            print_rank_0(self.pre_loss_description)
            self.show_pre = False
        if return_dict:
            return loss_D
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
            for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                student_rep.distill = teacher_rep.distill = True
                l = F.mse_loss(student_rep, teacher_rep)
                l = mpu.reduce_from_model_parallel_region(l)
                super().add_summary(f'inter_loss/attention_scores.{i}', l)
                loss_ += l
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
            l = F.mse_loss(student_rep, teacher_rep)
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
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
        if self.args.distilbert_fix_layernorm:
            self.layernorm = LayerNorm(args.hidden_size)
            self.t_layernorm = LayerNorm(args.teacher_hidden_size)
            self.t_layernorm.requires_grad_(False)

    def get_teacher_hook(self, **kwargs):
        return {'transformer': {'output': None} if not self.args.distilbert_fix_layernorm else {
            'layers': {self.args.teacher_num_layers - 1: {'tf_output': None}}}}

    def get_student_hook(self, **kwargs):
        return {'transformer': {'output': None} if not self.args.distilbert_fix_layernorm else {
            'layers': {self.args.num_layers - 1: {'tf_output': None}}}}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, loss_mask=None, labels=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or loss_mask is None:
            return loss_
        if self.args.distilbert_alpha_cos <= 0.:
            return loss_
        if not self.args.distilbert_fix_layernorm:
            s_o = s_inter_vars[s_hook['transformer']['output']]
            t_o = t_inter_vars[t_hook['transformer']['output']]
            s_o.distill = t_o.distill = True
        else:
            s_o = s_inter_vars[s_hook['transformer']['layers'][self.args.num_layers - 1]['tf_output']]
            t_o = t_inter_vars[t_hook['transformer']['layers'][self.args.teacher_num_layers - 1]['tf_output']]
            s_o.distill = t_o.distill = True
            if self.args.fp16:
                s_o = fp16_to_fp32(self.layernorm(fp32_to_fp16(s_o)))
                t_o = fp16_to_fp32(self.t_layernorm(fp32_to_fp16(t_o)))
            else:
                s_o = self.layernorm(s_o)
                t_o = self.t_layernorm(t_o)
        assert s_o.size() == t_o.size(), f'{s_o.size()} == {t_o.size()}'
        if self.args.distilbert_cos_mask_padding:
            loss_mask = labels > 0
        loss_mask = loss_mask.view(*loss_mask.size(), 1)
        s_o = (s_o * loss_mask).view(-1, s_o.size(-1))
        t_o = (t_o * loss_mask).view(-1, t_o.size(-1))
        target = s_o.new(s_o.size(0)).fill_(1)
        l = F.cosine_embedding_loss(s_o, t_o, target) * self.args.distilbert_alpha_cos
        super().add_summary(f'inter_loss/distilbert_alpha_cos', l)
        loss_ += l
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, labels=None, **kwargs):
        loss_ = 0.
        if self.args.finetune:
            return super().pre_loss(s_logits, t_logits, loss)
        self.pre_loss_description = 'pre_loss: 0'
        if self.args.distilbert_alpha_ce > 0:
            T = self.args.distill_temperature
            self.pre_loss_description += ' + %s*distilbert_alpha_ce(T%s)' % (self.args.distilbert_alpha_ce, T)
            if self.args.distill_wo_loss_mask:
                loss_mask = 1.
                self.pre_loss_description += '/wom'
            elif self.args.distilbert_ce_mask_padding:
                loss_mask = labels.view(*labels.size(), 1) > 0
                self.pre_loss_description += '/mask_pad'
            else:
                loss_mask = loss_mask.view(*loss_mask.size(), 1)
                self.pre_loss_description += '/mask_A_pad'
            s_logits = (s_logits * loss_mask / T).view(-1, s_logits.size(-1))
            t_logits = (t_logits * loss_mask / T).view(-1, t_logits.size(-1))
            kl_loss = F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction="batchmean")
            kl_loss = mpu.gather_from_model_parallel_region(kl_loss)
            l = kl_loss.mean() * T ** 2 * self.args.distilbert_alpha_ce
            super().add_summary(f'pre_loss/distilbert_alpha_ce', l)
            loss_ += l
        if self.args.distilbert_alpha_mlm > 0:
            self.pre_loss_description += ' + %s*distilbert_alpha_mlm' % self.args.distilbert_alpha_mlm
            l = loss * self.args.distilbert_alpha_mlm
            super().add_summary(f'pre_loss/distilbert_alpha_mlm', l)
            loss_ += l
        # show
        if self.show_pre:
            print_rank_0(self.pre_loss_description)
            self.show_pre = False
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
            setattr(self, c, eval(c)(language_model, args, show_pre=False, show_inter=False, summary_loss=False))

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
            l = super().pre_loss(s_logits, t_logits, loss, **kwargs)
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


class PKD(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        return {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[1: -1]},
            'output': None,
        }}

    def get_student_hook(self, **kwargs):
        return {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in range(1, self.args.num_layers)},
            'output': None,
        }}

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
            if self.args.pkd_normalized_patience:
                student_rep = F.normalize(student_rep, p=2, dim=-1)
                teacher_rep = F.normalize(teacher_rep, p=2, dim=-1)
            l = F.mse_loss(student_rep, teacher_rep)
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
        super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook)
        return loss_ * self.args.pkd_beta

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_D = super().pre_loss(s_logits, t_logits, loss, return_dict=True, **kwargs)
        loss_ = (1 - self.args.pkd_alpha) * loss_D['hard'] + self.args.pkd_alpha * loss_D['soft']
        return loss_


student_model_D = {
    None: None,
    'kd': GLMStudent,
    'tinybert': TinyBERT,
    'minilmv2': MiniLMv2,
    'minilm': MiniLM,
    'distilbert': DistilBERT,
    'mixbaseline': MixBaseline,
    'pkd': PKD,
}
