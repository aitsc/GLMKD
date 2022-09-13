import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel, GLMModel_empty
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce, hook_add
from utils import print_rank_0
import math
from tsc_base import merge_dict
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from fp16 import fp32_to_fp16, fp16_to_fp32
from distill.tools import all_mean_custom, aux_layer


class GLMStudent(torch.nn.Module):
    def __init__(self, language_model: GLMModel, args, show_pre=True, show_inter=True, summary_loss=True, **kwargs):
        super().__init__()
        self.origin_model = GLMModel_empty(language_model) if args.student_use_empty_glm else language_model
        self.args = args
        self.pre_loss_description = ''
        self.show_pre = show_pre
        self.show_inter = show_inter
        self.summary_writer = None
        self.summary_loss = summary_loss
        self.summary_suffix = ''  # 可用于多教师时增加标注
        self.inter_show_hooks = {}  # 用于滞后展示,例如多教师情况

    def get_teacher_hook(self, **kwargs):
        if self.args.distill_logits_parallel:
            return {'logits_parallel': None}
        return {}

    def get_student_hook(self, **kwargs):
        if self.args.distill_logits_parallel:
            return {'logits_parallel': None}
        return {}

    def forward(self, *inputs, **kwargs):
        return self.origin_model(*inputs, **kwargs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=None, **kwargs):
        loss_ = 0.
        # logits_parallel
        if self.args.distill_logits_parallel and 'logits_parallel' in s_hook and s_inter_vars:
            s_logits = s_inter_vars[s_hook['logits_parallel']]
            t_logits = t_inter_vars[t_hook['logits_parallel']]
            s_logits.distill = t_logits.distill = True
            T = self.args.distill_temperature
            student_likelihood = F.log_softmax(s_logits / T, dim=-1)
            targets_prob = F.softmax(t_logits / T, dim=-1)
            l = F.kl_div(student_likelihood, targets_prob, reduction="none") * T ** 2
            l = l.sum(-1)
            l = all_mean_custom(l, keep_batch, reduce=True)
            self.add_summary('inter_loss/logits_parallel', l)
            loss_ += l
        # show
        self.inter_show_hooks = {
            'student': hook_reduce(s_hook, s_inter_vars, filter=None),
            'teacher': hook_reduce(t_hook, t_inter_vars, filter=None),
        }
        if self.show_inter:
            print_rank_0(self.inter_show_hooks)
            self.show_inter = False
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, return_dict=False, labels=False, keep_batch=False, t_no=None, **kwargs):
        loss_ = 0.
        self.pre_loss_description = 'pre_loss: 0'
        T = self.args.distill_temperature
        loss_D = {}
        if loss_mask is None or self.args.distill_wo_loss_mask:
            mask = 1.
            self.pre_loss_description += '/wom'
        elif labels is not None and self.args.distill_only_mask_pad:
            mask = labels.view(*labels.size(), 1) > 0
            self.pre_loss_description += '/mask_pad'
        else:  # 在 finetune 中一般用于 seq2seq_forward_step
            mask = loss_mask.view(*loss_mask.size(), 1)
            self.pre_loss_description += '/mask_A_pad'
        if self.args.finetune:
            if self.args.distill_ft_soft:
                self.pre_loss_description += ' + %s*distill_ft_soft(T%s)'%(self.args.distill_soft_rate,T)
                if self.args.distill_ft_soft_mse:
                    l = F.mse_loss(s_logits * mask, t_logits * mask, reduction='none')
                    self.pre_loss_description += '(mse)'
                else:
                    student_likelihood = F.log_softmax(s_logits * mask / T, dim=-1)
                    targets_prob = F.softmax(t_logits * mask / T, dim=-1)
                    if self.args.distill_ft_soft_kl:
                        l = F.kl_div(student_likelihood, targets_prob, reduction="none") * T ** 2
                    else:
                        l = (- targets_prob * student_likelihood)
                    l = l.sum(-1)
                l = all_mean_custom(l, keep_batch, reduce=True)  # 可能等于加权(1/模型并行数)
                l = l * self.args.distill_soft_rate
                self.add_summary('pre_loss/ft_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_ft_hard:
                self.pre_loss_description += ' + %s*distill_ft_hard'%self.args.distill_hard_rate
                l = loss * self.args.distill_hard_rate
                self.add_summary('pre_loss/ft_hard', l)
                loss_ += l
                loss_D['hard'] = l
        else:
            if self.args.distill_pt_soft:
                self.pre_loss_description += ' + %s*distill_pt_soft(T%s)'%(self.args.distill_soft_rate,T)
                if self.args.distill_pt_soft_mse:
                    l = F.mse_loss(s_logits * mask, t_logits * mask, reduction='none')
                    self.pre_loss_description += '(mse)'
                else:
                    student_likelihood = F.log_softmax(s_logits * mask / T, dim=-1)
                    targets_prob = F.softmax(t_logits * mask / T, dim=-1)
                    if self.args.distill_pt_soft_ce:
                        l = (- targets_prob * student_likelihood)
                    else:
                        l = F.kl_div(student_likelihood, targets_prob, reduction="none") * T ** 2
                    l = l.sum(-1)
                l = all_mean_custom(l, keep_batch, reduce=True)  # 可能等于加权(1/模型并行数)
                l = l * self.args.distill_soft_rate
                self.add_summary('pre_loss/pt_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_pt_hard:
                self.pre_loss_description += ' + %s*distill_pt_hard'%self.args.distill_hard_rate
                l = loss * self.args.distill_hard_rate
                self.add_summary('pre_loss/pt_hard', l)
                loss_ += l
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
            if hasattr(value, 'item'):
                if len(value.shape) > 1 or len(value.shape) >= 1 and value.shape[0] > 1:
                    value = value.mean()
                value = value.item()
            self.summary_writer.add_scalar(name + self.summary_suffix, value, self.args.iteration)
            return True
        else:
            return False


def unpacking_student_model(model, attrs=('origin_model', 'get_teacher_hook')):
    # 默认拆包 model 直到遇到 GLMStudent, 用于使用 GLMStudent 内部的函数, 或者修改 attrs 用于其他内部模型
    while True:
        if sum([hasattr(model, a) for a in attrs]) == len(attrs):
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
        Linear = mpu.ColumnParallelLinear if args.tinybert_fit_parallel else torch.nn.Linear
        if args.tinybert_fit_compatible_mt and args.mt_hidden_size:
            mt_hidden_size = [int(i) for i in args.mt_hidden_size.split(':')]
            for i, hidden_size in enumerate(mt_hidden_size):
                setattr(self, f'fit_dense_{i}', Linear(args.hidden_size, hidden_size))
        else:
            self.fit_dense = Linear(args.hidden_size, args.teacher_hidden_size)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        if self.args.tinybert_wo_inter:
            hook = {}
        elif self.args.tinybert_only_emb_final:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}, 'output': None}}
        else:
            hook ={'transformer': {
                'layers': {} if self.args.tinybert_inter_final else {
                    **{i: {'layernorm_output': None} for i in layers[:-1]},
                    **({} if self.args.tinybert_wo_att else {i - 1: {'attention_scores': None} for i in layers[1:]}),
                },
                'output': None,
            }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(range(self.args.num_layers))
        if self.args.tinybert_wo_inter:
            hook = {}
        elif self.args.tinybert_only_emb_final:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}, 'output': None}}
        else:
            hook = {'transformer': {
                'layers': {} if self.args.tinybert_inter_final else {i: {
                    'layernorm_output': None,
                    **({} if self.args.tinybert_wo_att else {'attention_scores': None}),
                } for i in layers},
                'output': None,
            }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None and not self.args.tinybert_wo_inter and inter_vars \
            and not (self.args.tinybert_fit_compatible_mt and self.args.mt_hidden_size):
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            for v in hook['transformer']['layers'].values():
                inter_vars[v['layernorm_output']] = self.fit_dense(inter_vars[v['layernorm_output']])
            inter_vars[hook['transformer']['output']] = self.fit_dense(inter_vars[hook['transformer']['output']])
        return hook_return(hook, inter_vars, outputs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=None, **kwargs):
        loss_ = 0.
        if self.args.tinybert_wo_inter or len(s_inter_vars) == 0:
            return loss_
        # 学生中间层 W
        if self.args.tinybert_fit_compatible_mt and self.args.mt_hidden_size \
            and 'transformer' in s_hook and 'layers' in s_hook['transformer'] and t_no is not None:
            s_inter_vars = s_inter_vars.copy()
            fit_dense = getattr(self, f'fit_dense_{t_no}')
            for v in s_hook['transformer']['layers'].values():
                if 'layernorm_output' in v:
                    s_inter_vars[v['layernorm_output']] = aux_layer(self.args, fit_dense, s_inter_vars[v['layernorm_output']])
            if 'output' in s_hook['transformer']:
                s_inter_vars[s_hook['transformer']['output']] = aux_layer(self.args, fit_dense, s_inter_vars[s_hook['transformer']['output']])
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        # attentions
        if not self.args.tinybert_inter_final and not self.args.tinybert_wo_att \
            and not self.args.tinybert_only_emb_final:
            student_reps = get_layer_f('s', 'attention_scores')
            teacher_reps = get_layer_f('t', 'attention_scores')
            for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                student_rep.distill = teacher_rep.distill = True
                l = F.mse_loss(student_rep, teacher_rep, reduction='none')
                l = all_mean_custom(l, keep_batch, reduce=True)
                super().add_summary(f'inter_loss/attention_scores.{i}', l)
                loss_ += l
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
            l = F.mse_loss(student_rep, teacher_rep, reduction='none')
            l = all_mean_custom(l, keep_batch)
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, t_no=t_no, **kwargs)
        return loss_


class MiniLMv2(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        if self.args.minilmv2_wo_inter:
            hook = {}
        else:
            hook = {'transformer': {'layers': {self.args.minilmv2_teacher_layer - 1: {
                    'mixed_query_layer': None, 'mixed_key_layer': None, 'mixed_value_layer': None
            }}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        if self.args.minilmv2_wo_inter:
            hook = {}
        else:
            hook = {'transformer': {'layers': {self.args.num_layers - 1: {
                    'mixed_query_layer': None, 'mixed_key_layer': None, 'mixed_value_layer': None
            }}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or self.args.minilmv2_wo_inter:
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
            kl_loss = F.kl_div(F.log_softmax(s_rep, dim=-1), F.softmax(t_rep, dim=-1), reduction="none")
            kl_loss = kl_loss.sum(-1)
            kl_loss = all_mean_custom(kl_loss, keep_batch, reduce=True)
            loss_ += kl_loss
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_


class MiniLM(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        hook = {'transformer': {'layers': {self.args.teacher_num_layers - 1: {
                'attention_probs': None, 'value_layer': None
        }}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        hook = {'transformer': {'layers': {self.args.num_layers - 1: {
                'attention_probs': None, 'value_layer': None
        }}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, **kwargs):
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
        kl_loss = F.kl_div(F.log_softmax(s_a, dim=-1), F.softmax(t_a, dim=-1), reduction="none")
        kl_loss += F.kl_div(F.log_softmax(s_v2, dim=-1), F.softmax(t_v2, dim=-1), reduction="none")
        kl_loss = kl_loss.sum(-1)
        kl_loss = all_mean_custom(kl_loss, keep_batch, reduce=True)
        loss_ += kl_loss
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_


class DistilBERT(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        if self.args.distilbert_fix_layernorm:
            self.layernorm = LayerNorm(args.hidden_size)
            self.t_layernorm = LayerNorm(args.teacher_hidden_size)
            self.t_layernorm.requires_grad_(False)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        hook = {'transformer': {'output': None} if not self.args.distilbert_fix_layernorm else {
            'layers': {self.args.teacher_num_layers - 1: {'tf_output': None}}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        hook = {'transformer': {'output': None} if not self.args.distilbert_fix_layernorm else {
            'layers': {self.args.num_layers - 1: {'tf_output': None}}}}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, loss_mask=None, labels=None, keep_batch=False, **kwargs):
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
        l = F.cosine_embedding_loss(s_o, t_o, target, reduction='none') * self.args.distilbert_alpha_cos
        l = all_mean_custom(l, keep_batch)
        super().add_summary(f'inter_loss/distilbert_alpha_cos', l)
        loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, loss_mask=loss_mask, labels=labels, keep_batch=keep_batch, **kwargs)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, loss_mask=None, labels=None, keep_batch=False, **kwargs):
        loss_ = 0.
        self.pre_loss_description = 'pre_loss: 0'
        if self.args.distilbert_alpha_ce > 0:
            T = self.args.distill_temperature
            self.pre_loss_description += ' + %s*distilbert_alpha_ce(T%s)' % (self.args.distilbert_alpha_ce, T)
            if self.args.distill_wo_loss_mask or loss_mask is None:
                loss_mask = 1.
                self.pre_loss_description += '/wom'
            elif self.args.distilbert_ce_mask_padding:
                loss_mask = labels.view(*labels.size(), 1) > 0
                self.pre_loss_description += '/mask_pad'
            else:
                loss_mask = loss_mask.view(*loss_mask.size(), 1)
                self.pre_loss_description += '/mask_A_pad'
            s_logits = (s_logits * loss_mask / T)
            t_logits = (t_logits * loss_mask / T)
            kl_loss = F.kl_div(F.log_softmax(s_logits, dim=-1), F.softmax(t_logits, dim=-1), reduction="none")
            kl_loss = kl_loss.sum(-1)
            kl_loss = all_mean_custom(kl_loss, keep_batch, reduce=True)
            l = kl_loss * T ** 2 * self.args.distilbert_alpha_ce
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
        hook_L = [super().get_teacher_hook(**kwargs)]
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers, layers_per_block))
        hook = self.get_inter_hook(layers, st='t')
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(i for i in range(self.args.num_layers))
        hook = self.get_inter_hook(layers, st='s')
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_model=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or self.args.erdistill_inter in {'', None}:
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        student_reps = get_layer_f('s', 'layernorm_output') + [s_inter_vars[s_hook['transformer']['output']]]
        teacher_reps = get_layer_f('t', 'layernorm_output') + [t_inter_vars[t_hook['transformer']['output']]]
        # 局部相似
        if not self.args.erdistill_wo_local:
            t_reps = []
            for t_rep in teacher_reps:
                t_rep.distill = True
                t_rep = torch.matmul(t_rep, t_rep.transpose(-1,-2))
                if self.args.erdistill_inter_mse:
                    t_rep = t_rep / t_rep.size(-1)
                else:
                    t_rep = F.softmax(t_rep / math.sqrt(t_rep.size(-1)), dim=-1)
                t_reps.append(t_rep)
            if self.args.erdistill_inter in {'1plus'}:
                t_reps += t_reps
            for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
                s_rep.distill = True
                s_rep = torch.matmul(s_rep, s_rep.transpose(-1,-2))
                if self.args.erdistill_inter_mse:
                    l = F.mse_loss(s_rep / s_rep.size(-1), t_rep, reduction='none')
                else:
                    l = F.kl_div(F.log_softmax(s_rep / math.sqrt(s_rep.size(-1)), dim=-1), t_rep, reduction="none")
                    l = l.sum(-1)
                l = all_mean_custom(l, keep_batch)
                super().add_summary(f'inter_loss/local.{i}', l)
                l = l * 0.1 if i == 0 and self.args.erdistill_inter in {'1plus'} else l
                loss_ += l
        # 全局相似
        if not self.args.erdistill_wo_global:
            t_emb_w = find_model_inter_var(t_model, 'word_embeddings.weight')
            s_emb_w = find_model_inter_var(self, 'word_embeddings.weight')
            t_reps = []
            for i, t_rep in enumerate(teacher_reps):
                t_rep.distill = True
                t_rep = mpu.copy_to_model_parallel_region(t_rep)
                t_rep = fp32_to_fp16(t_rep) if self.args.fp16 else t_rep
                t_rep = F.linear(t_rep, t_emb_w)
                t_rep = t_rep if self.args.erdistill_inter_mse else F.softmax(t_rep, dim=-1)
                t_reps.append(t_rep)
            if self.args.erdistill_inter in {'1plus'}:
                t_reps += t_reps
            for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
                s_rep.distill = True
                s_rep = mpu.copy_to_model_parallel_region(s_rep)
                s_rep = fp32_to_fp16(s_rep) if self.args.fp16 else s_rep
                s_rep = F.linear(s_rep, s_emb_w)
                if self.args.erdistill_inter_mse:
                    l = F.mse_loss(s_rep, t_rep, reduction='none')
                else:
                    l = F.kl_div(F.log_softmax(s_rep, dim=-1), t_rep, reduction="none")
                    l = l.sum(-1)
                l = all_mean_custom(l, keep_batch, reduce=True)
                super().add_summary(f'inter_loss/global.{i}', l)
                l = l * 0.1 if i == 0 and self.args.erdistill_inter in {'1plus'} else l
                loss_ += fp16_to_fp32(l)
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, t_model=t_model, **kwargs)
        return loss_


class MixBaseline(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        self.inter_bl = args.mixbaseline_inter_bl.split(',')
        # 支持 --distill_ft_hard 的模型必须放在最后, 保证只算一次
        self.pre_bl_pretrain_soft = args.mixbaseline_pre_bl_pt_soft.split(',')  # 重复的预训练软标签构建方式不需要
        self.pre_bl_pretrain_soft = [i for i in self.pre_bl_pretrain_soft if i]
        self.pre_bl_finetune_soft = args.mixbaseline_pre_bl_ft_soft.split(',')  # 有 distill_ft 相关参数就等于包含 KD(super())
        self.pre_bl_finetune_soft = [i for i in self.pre_bl_finetune_soft if i]
        self.baselines = set(self.inter_bl + self.pre_bl_pretrain_soft + self.pre_bl_finetune_soft)
        for c in self.baselines:
            setattr(self, c, eval(c)(language_model, args, show_pre=False, show_inter=False, summary_loss=False))

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        if not self.args.mixbaseline_wo_inter:
            hook_L += [getattr(self, c).get_teacher_hook() for c in self.inter_bl]
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        if not self.args.mixbaseline_wo_inter:
            hook_L += [getattr(self, c).get_student_hook() for c in self.inter_bl]
        return merge_dict(hook_L)

    def forward(self, *inputs, **kwargs):
        if 'TinyBERT' in self.baselines:
            return self.TinyBERT(*inputs, **kwargs)
        else:
            return super().forward(*inputs, **kwargs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        distill_logits_parallel = self.args.distill_logits_parallel
        self.args.distill_logits_parallel = False  # 使得这个参数只执行一次
        for c in self.inter_bl:
            distill_temperature = self.args.distill_temperature
            if hasattr(self.args, f'mixbaseline_{c.lower()}_t'):
                self.args.distill_temperature = getattr(self.args, f'mixbaseline_{c.lower()}_t')
            l = getattr(self, c).inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs)
            super().add_summary(f'inter_loss/{c}', l)
            loss_ += l
            self.args.distill_temperature = distill_temperature
        self.args.distill_logits_parallel = distill_logits_parallel
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_ = 0.
        show_pre = self.show_pre
        self.show_pre = False
        pre_loss_description = ['all pre_loss:']
        pre_bl = self.pre_bl_finetune_soft if self.args.finetune else self.pre_bl_pretrain_soft
        distill_ft_hard, distill_pt_hard = self.args.distill_ft_hard, self.args.distill_pt_hard
        if pre_bl:
            self.args.distill_ft_hard = self.args.distill_pt_hard = False  # 硬标签只需要算一次
        # KD pre_loss
        if self.args.finetune:
            l = super().pre_loss(s_logits, t_logits, loss, **kwargs)
            super().add_summary(f'pre_loss/KD', l)
            loss_ += l
            pre_loss_description.append(f'\tKD - {self.pre_loss_description}')
        # other pre_loss
        for i, c in enumerate(pre_bl):
            if i == len(pre_bl) - 1:
                self.args.distill_ft_hard, self.args.distill_pt_hard = distill_ft_hard, distill_pt_hard
            distill_temperature = self.args.distill_temperature
            if hasattr(self.args, f'mixbaseline_{c.lower()}_t'):
                self.args.distill_temperature = getattr(self.args, f'mixbaseline_{c.lower()}_t')
            l = getattr(self, c).pre_loss(s_logits, t_logits, loss, **kwargs)
            super().add_summary(f'pre_loss/{c}', l)
            loss_ += l
            pre_loss_description.append(f'\t{c} - {getattr(self, c).pre_loss_description}')
            self.args.distill_temperature = distill_temperature
        # show
        self.pre_loss_description = '\n'.join(pre_loss_description)
        if show_pre:
            print_rank_0(self.pre_loss_description)
        return loss_


class PKD(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        x = 0 if self.args.pkd_use_embed else 1
        hook = {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[x: -1]},
            'output': None,
        }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        x = 0 if self.args.pkd_use_embed else 1
        hook = {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in range(x, self.args.num_layers)},
            'output': None,
        }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, **kwargs):
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
            l = F.mse_loss(student_rep, teacher_rep, reduction='none')
            l = all_mean_custom(l, keep_batch)
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_ * self.args.pkd_beta

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_D = super().pre_loss(s_logits, t_logits, loss, return_dict=True, **kwargs)
        loss_ = 0.
        if 'hard' in loss_D:
            loss_ += (1 - self.args.pkd_alpha) * loss_D['hard']
        if 'soft' in loss_D:
            loss_ += self.args.pkd_alpha * loss_D['soft']
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
    'pkd': PKD,
}