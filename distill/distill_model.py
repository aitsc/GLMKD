import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel, GLMModel_empty
import mpu
import torch.nn.functional as F
from mpu import hook_model, hook_return, hook_reduce, hook_add
from utils import print_rank_0, find_model_inter_var
import math
from tsc_base import merge_dict, fast_uniform_seg, cumulative_sum
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from fp16 import fp32_to_fp16, fp16_to_fp32
from distill.tools import aux_layer, get_checkpoint_forward_args
import random
import copy
from distill.cite_mgskd import SampleLoss, TokenPhraseLoss
from mpu import checkpoint, get_cuda_rng_tracker
import deepspeed
from distill.custom_loss import CustomLoss


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
        self.summary_suffix = ''  # 可用于多教师,多次数据等情况增加标注
        self.inter_show_hooks = {}  # 用于滞后展示,例如多教师情况

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def get_teacher_hook(self, t_no=0, **kwargs):
        hook = {}
        if self.args.distill_logits_parallel:
            hook['logits_parallel'] = None
        return hook

    def get_student_hook(self, **kwargs):
        hook = {}
        if self.args.distill_logits_parallel:
            if self.args.distill_logit_mask_pad:
                hook['position_ids'] = None
            hook['logits_parallel'] = None
        return hook

    def get_teacher_hook_op(self, t_no=0, **kwargs):
        return {}

    def get_student_hook_op(self, **kwargs):
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
            if self.args.distill_logit_mask_map and self.origin_model.map_vocab_size:
                origin_id_mask_map = self.origin_model.map_vocab_paras['origin_id_mask_map']
                origin_id_mask_map = mpu.scatter_to_model_parallel_region(origin_id_mask_map)
                s_logits = torch.masked_select(s_logits, origin_id_mask_map)
                t_logits = torch.masked_select(t_logits, origin_id_mask_map)
            if self.args.distill_logit_mask_pad:
                position_ids = s_inter_vars[s_hook['position_ids']]
                position_ids.distill = True
                mask = (position_ids[:, 0] > 0).int()
                mask[:, 0] = 1
            else:
                mask = 1.
            if self.args.distill_logit_mse:
                l = CustomLoss.mse_loss(s_logits, t_logits, parallel='gather', input_mask=mask, keep_batch=keep_batch)
            else:
                T = self.args.distill_temperature
                l = CustomLoss.kl_div(s_logits / T, t_logits / T, parallel='gather', input_mask=mask, keep_batch=keep_batch) * T ** 2
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
        parallel = 'gather' if self.args.custom_logits_paralle else ''
        if self.args.finetune:
            if self.args.distill_ft_soft and self.args.distill_soft_rate:
                self.pre_loss_description += ' + %s*distill_ft_soft(T%s)'%(self.args.distill_soft_rate,T)
                if self.args.distill_ft_soft_mse:
                    l = CustomLoss.mse_loss(s_logits, t_logits, maks=mask, parallel=parallel, keep_batch=keep_batch)
                    self.pre_loss_description += 'mse'
                else:
                    if self.args.distill_ft_soft_kl:
                        l = CustomLoss.kl_div(s_logits / T, t_logits / T, input_mask=mask, parallel=parallel, keep_batch=keep_batch) * T ** 2
                        self.pre_loss_description += 'kl'
                    else:
                        l = CustomLoss.cross_entropy(s_logits / T, t_logits / T, input_mask=mask, parallel=parallel, keep_batch=keep_batch)
                        self.pre_loss_description += 'ce'
                l = l * self.args.distill_soft_rate
                self.add_summary('pre_loss/ft_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_ft_hard and self.args.distill_hard_rate:
                self.pre_loss_description += ' + %s*distill_ft_hard'%self.args.distill_hard_rate
                l = loss * self.args.distill_hard_rate
                self.add_summary('pre_loss/ft_hard', l)
                loss_ += l
                loss_D['hard'] = l
        else:
            if self.args.distill_pt_soft and self.args.distill_soft_rate:
                self.pre_loss_description += ' + %s*distill_pt_soft(T%s)'%(self.args.distill_soft_rate,T)
                if self.args.distill_pt_soft_mse:
                    l = CustomLoss.mse_loss(s_logits, t_logits, mask=mask, parallel=parallel, keep_batch=keep_batch)
                    self.pre_loss_description += 'mse'
                else:
                    if self.args.distill_pt_soft_ce:
                        l = CustomLoss.cross_entropy(s_logits / T, t_logits / T, input_mask=mask, parallel=parallel, keep_batch=keep_batch)
                        self.pre_loss_description += 'ce'
                    else:
                        l = CustomLoss.kl_div(s_logits / T, t_logits / T, input_mask=mask, parallel=parallel, keep_batch=keep_batch) * T ** 2
                        self.pre_loss_description += 'kl'
                l = l * self.args.distill_soft_rate
                self.add_summary('pre_loss/pt_soft', l)
                loss_ += l
                loss_D['soft'] = l
            if self.args.distill_pt_hard and self.args.distill_hard_rate:
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
        # add random hook 功能
        self.teachers_hook = {}  # {t_no:hook,..}; 用于hook更换前维持上次随机的hook
        self.last_epoch = 0
        self.last_iter = 0

    def get_teacher_hook(self, t_no=0, **kwargs):
        if t_no in self.teachers_hook and self.args.tinybert_random_layers:
            if self.args.tinybert_random_e:
                if self.args.custom_current_epoch % self.args.tinybert_random_e != 0 or \
                    self.args.custom_current_epoch == self.last_epoch \
                    :
                    return copy.deepcopy(self.teachers_hook[t_no])
            elif self.args.tinybert_random_i:
                if self.args.iteration % self.args.tinybert_random_i != 0 or \
                    self.args.iteration == self.last_iter \
                    :
                    return copy.deepcopy(self.teachers_hook[t_no])
        self.last_epoch = self.args.custom_current_epoch
        self.last_iter = self.args.iteration
        # 重新生成hook
        hook_L = [super().get_teacher_hook(t_no=0, **kwargs)]
        if self.args.tinybert_random_layers:
            layers = random.sample(range(1, self.args.teacher_num_layers), self.args.num_layers - 1)
            layers.sort()
            if self.args.tinybert_random_show:
                print_rank_0(f'TinyBERT.get_teacher_hook(t_no={t_no})-new_layers: {layers}')
            layers = [0] + layers + [self.args.teacher_num_layers]
        else:
            layers = [0] + cumulative_sum(fast_uniform_seg(self.args.num_layers, [1] * self.args.teacher_num_layers))
            # layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
            # layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        if self.args.tinybert_wo_inter:
            hook = {}
        elif self.args.tinybert_only_emb_final:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}, 'output': None}}
        elif self.args.tinybert_only_emb:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}}}
        else:
            hook ={'transformer': {
                'layers': {} if self.args.tinybert_inter_final else {
                    **{i: {'layernorm_output': None} for i in layers[1 if self.args.tinybert_wo_emb else 0:-1]},
                    **({} if self.args.tinybert_wo_att else {i - 1: {'attention_scores': None} for i in layers[1:]}),
                },
                **({} if self.args.tinybert_wo_final else {'output': None}),
            }}
        if (self.args.tinybert_only_emb_final or self.args.tinybert_inter_final) and self.args.tinybert_custom_final > 1:
            if 'output' in hook['transformer']:
                del hook['transformer']['output']
            layer = self.args.teacher_num_layers - self.args.tinybert_custom_final + 1
            hook['transformer']['layers'].setdefault(layer, {}).setdefault('layernorm_output', None)
        hook_L.append(hook)
        self.teachers_hook[t_no] = merge_dict(hook_L)
        return copy.deepcopy(self.teachers_hook[t_no])

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(range(self.args.num_layers))
        if self.args.tinybert_wo_inter:
            hook = {}
        elif self.args.tinybert_only_emb_final:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}, 'output': None}}
        elif self.args.tinybert_only_emb:
            hook = {'transformer': {'layers': {0: {'layernorm_output': None}}}}
        else:
            hook = {'transformer': {
                'layers': {} if self.args.tinybert_inter_final else {i: {
                    **({} if i==0 and self.args.tinybert_wo_emb else {'layernorm_output': None}),
                    **({} if self.args.tinybert_wo_att else {'attention_scores': None}),
                } for i in layers},
                **({} if self.args.tinybert_wo_final else {'output': None}),
            }}
        if (self.args.tinybert_only_emb_final or self.args.tinybert_inter_final) and self.args.tinybert_custom_final > 1:
            if 'output' in hook['transformer']:
                del hook['transformer']['output']
            layer = self.args.num_layers - self.args.tinybert_custom_final + 1
            hook['transformer']['layers'].setdefault(layer, {}).setdefault('layernorm_output', None)
        hook_L.append(hook)
        return merge_dict(hook_L)

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None and not self.args.tinybert_wo_inter and inter_vars \
            and not (self.args.tinybert_fit_compatible_mt and self.args.mt_hidden_size) \
            and 'transformer' in hook:
            # {'transformer': {'layers':{0:{'layernorm_output':,'attention_scores':},..},'output':,..},..}
            if 'layers' in hook['transformer']:
                for v in hook['transformer']['layers'].values():
                    if 'layernorm_output' not in v:
                        continue
                    inter_vars[v['layernorm_output']] = self.fit_dense(inter_vars[v['layernorm_output']])
            if 'output' in hook['transformer']:
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
                l = CustomLoss.mse_loss(student_rep, teacher_rep, parallel='gather', keep_batch=keep_batch)
                super().add_summary(f'inter_loss/attention_scores.{i}', l)
                loss_ += l
        # emb + hidden_states
        student_reps = get_layer_f('s', 'layernorm_output')
        teacher_reps = get_layer_f('t', 'layernorm_output')
        if 'transformer' in s_hook and 'output' in s_hook['transformer']:
            student_reps += [s_inter_vars[s_hook['transformer']['output']]]
            teacher_reps += [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
            l = CustomLoss.mse_loss(student_rep, teacher_rep, keep_batch=keep_batch)
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
            if self.args.minilmv2_teacher_layer < 0:
                minilmv2_teacher_layer = self.args.teacher_num_layers + self.args.minilmv2_teacher_layer
            else:
                minilmv2_teacher_layer = self.args.minilmv2_teacher_layer - 1
            hook = {'transformer': {'layers': {minilmv2_teacher_layer: {
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

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=0, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or self.args.minilmv2_wo_inter:
            return loss_
        s_qkv, t_qkv = [], []
        if self.args.minilmv2_teacher_layer < 0:
            mt_num_layers = [int(i) for i in self.args.mt_num_layers.split(':')] if self.args.mt_num_layers else [self.args.teacher_num_layers]
            minilmv2_teacher_layer = mt_num_layers[t_no] + self.args.minilmv2_teacher_layer
        else:
            minilmv2_teacher_layer = self.args.minilmv2_teacher_layer - 1
        for i in ['mixed_query_layer', 'mixed_key_layer', 'mixed_value_layer']:
            s_qkv.append(s_inter_vars[s_hook['transformer']['layers'][self.args.num_layers - 1][i]])
            t_qkv.append(t_inter_vars[t_hook['transformer']['layers'][minilmv2_teacher_layer][i]])
        relation_heads_mt = [
            int(i) for i in self.args.minilmv2_relation_heads_mt.split(',')
        ] if self.args.minilmv2_relation_heads_mt else [self.args.minilmv2_relation_heads]
        n_heads = int(relation_heads_mt[t_no] / mpu.get_model_parallel_world_size())
        # q k v
        for s_rep, t_rep in zip(s_qkv, t_qkv):
            s_rep.distill = t_rep.distill = True
            s_rep = s_rep.view(*s_rep.size()[:-1], n_heads, -1).permute(0, 2, 1, 3)
            s_rep = torch.matmul(s_rep, s_rep.transpose(-1,-2)) / math.sqrt(s_rep.size(-1))
            t_rep = t_rep.view(*t_rep.size()[:-1], n_heads, -1).permute(0, 2, 1, 3)
            t_rep = torch.matmul(t_rep, t_rep.transpose(-1,-2)) / math.sqrt(t_rep.size(-1))
            l = CustomLoss.kl_div(s_rep, t_rep, parallel='gather', keep_batch=keep_batch)
            loss_ += l
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
        l = CustomLoss.kl_div(s_a, t_a, parallel='gather', keep_batch=keep_batch)
        l += CustomLoss.kl_div(s_v2, t_v2, parallel='gather', keep_batch=keep_batch)
        loss_ += l
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
        l = CustomLoss.cos_distance(s_o, t_o, keep_batch=keep_batch) * self.args.distilbert_alpha_cos
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
            parallel = 'gather' if self.args.custom_logits_paralle else ''
            s_logits = (s_logits * loss_mask / T)
            t_logits = (t_logits * loss_mask / T)
            l = CustomLoss.kl_div(s_logits, t_logits, parallel=parallel, keep_batch=keep_batch)
            l = l * T ** 2 * self.args.distilbert_alpha_ce
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
                t_reps.append(t_rep)
            if self.args.erdistill_inter in {'1plus'}:
                t_reps += t_reps
            for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
                s_rep.distill = True
                s_rep = torch.matmul(s_rep, s_rep.transpose(-1,-2))
                if self.args.erdistill_inter_mse:
                    l = CustomLoss.mse_loss(s_rep / s_rep.size(-1), t_rep / t_rep.size(-1), keep_batch=keep_batch)
                else:
                    l = CustomLoss.kl_div(s_rep / math.sqrt(s_rep.size(-1)), t_rep / math.sqrt(t_rep.size(-1)), keep_batch=keep_batch)
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
                t_reps.append(t_rep)
            if self.args.erdistill_inter in {'1plus'}:
                t_reps += t_reps
            for i, (s_rep, t_rep) in enumerate(zip(student_reps, t_reps)):
                s_rep.distill = True
                s_rep = mpu.copy_to_model_parallel_region(s_rep)
                s_rep = fp32_to_fp16(s_rep) if self.args.fp16 else s_rep
                s_rep = F.linear(s_rep, s_emb_w)
                if self.args.erdistill_inter_mse:
                    l = CustomLoss.mse_loss(s_rep, t_rep, parallel='gather', keep_batch=keep_batch)
                else:
                    l = CustomLoss.kl_div(s_rep, t_rep, parallel='gather', keep_batch=keep_batch)
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
            if self.args.mixbaseline_inter_checkpoint:
                def custom(loss_, *inputs, **kwargs):
                    l = getattr(self, c).inter_loss(*inputs, **kwargs)
                    loss_ = loss_ + l
                    return loss_, l
                loss_, l = checkpoint(*get_checkpoint_forward_args(
                    custom, loss_, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs))
            else:
                l = getattr(self, c).inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs)
                loss_ += l
            super().add_summary(f'inter_loss/{c}', l)
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
            **({} if self.args.pkd_wo_final else {'output': None}),
        }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(range(self.args.num_layers + 1))
        x = 0 if self.args.pkd_use_embed else 1
        hook = {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[x: -1]},
            **({} if self.args.pkd_wo_final else {'output': None}),
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

        student_reps = get_layer_f('s', 'layernorm_output')
        teacher_reps = get_layer_f('t', 'layernorm_output')
        if 'transformer' in s_hook and 'output' in s_hook['transformer']:
            student_reps += [s_inter_vars[s_hook['transformer']['output']]]
            teacher_reps += [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
            if self.args.pkd_normalized_patience:
                student_rep = F.normalize(student_rep, p=2, dim=-1)
                teacher_rep = F.normalize(teacher_rep, p=2, dim=-1)
            l = CustomLoss.mse_loss(student_rep, teacher_rep, keep_batch=keep_batch)
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
        loss_ = loss_ * self.args.pkd_beta
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        loss_D = super().pre_loss(s_logits, t_logits, loss, return_dict=True, **kwargs)
        loss_ = 0.
        if 'hard' in loss_D:
            loss_ += (1 - self.args.pkd_alpha) * loss_D['hard']
        if 'soft' in loss_D:
            loss_ += self.args.pkd_alpha * loss_D['soft']
        return loss_


class RAIL_KD(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        self.teachers_hook = {}  # {t_no:hook,..}; 用于hook更换前维持上次随机的hook
        self.last_epoch = 0
        self.last_iter = 0
        Linear = mpu.ColumnParallelLinear
        m = 1
        if args.rail_kd_concatenated:
            m = self.args.num_layers - 1
            if args.rail_kd_has_embed:
                m += 1
            if args.rail_kd_has_final:
                m += 1
        # teacher fit_dense
        mt_hidden_size = [int(i) for i in args.mt_hidden_size.split(':')] if args.mt_hidden_size else [args.teacher_hidden_size]
        for i, hidden_size in enumerate(mt_hidden_size):
            setattr(self, f't_fit_dense_{i}', Linear(hidden_size * m, args.rail_kd_u))
        # student fit_dense
        self.s_fit_dense = Linear(args.hidden_size * m, args.rail_kd_u)

    def get_teacher_hook(self, t_no=0, **kwargs):
        if t_no in self.teachers_hook:
            if self.args.rail_kd_epochs:
                if self.args.custom_current_epoch % self.args.rail_kd_epochs != 0 or \
                    self.args.custom_current_epoch == self.last_epoch \
                    :
                    return copy.deepcopy(self.teachers_hook[t_no])
            elif self.args.rail_kd_iters:
                if self.args.iteration % self.args.rail_kd_iters != 0 or \
                    self.args.iteration == self.last_iter \
                    :
                    return copy.deepcopy(self.teachers_hook[t_no])
        self.last_epoch = self.args.custom_current_epoch
        self.last_iter = self.args.iteration
        # 重新生成hook
        hook_L = [super().get_teacher_hook(t_no=0, **kwargs)]
        if self.args.rail_kd_no_random:
            layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
            layers = list(range(0, self.args.teacher_num_layers + 1, layers_per_block))[1: -1]
        else:
            layers = random.sample(range(1, self.args.teacher_num_layers), self.args.num_layers - 1)
            layers.sort()
        if self.args.rail_kd_has_embed:
            layers.insert(0, 0)
        hook = {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers},
            **({'output': None} if self.args.rail_kd_has_final else {}),
        }}
        hook_L.append(hook)
        self.teachers_hook[t_no] = merge_dict(hook_L)
        # 提示
        if self.args.rail_kd_show_hook_change:
            layers_ = layers + (['final'] if self.args.rail_kd_has_final else [])
            print_rank_0(f'RAIL_KD.get_teacher_hook(t_no={t_no})-new_layers: {layers_}')
        return copy.deepcopy(self.teachers_hook[t_no])

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(range(self.args.num_layers + 1))
        x = 0 if self.args.rail_kd_has_embed else 1
        hook = {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[x: -1]},
            **({'output': None} if self.args.rail_kd_has_final else {}),
        }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def forward(self, *inputs, hook=None, **kwargs):
        inter_vars = []
        outputs = hook_model(hook, inter_vars, self.origin_model, *inputs, **kwargs)
        if hook is not None and inter_vars and 'transformer' in hook:
            if 'layers' in hook['transformer']:
                for v in hook['transformer']['layers'].values():
                    if 'layernorm_output' not in v:
                        continue
                    inter_vars[v['layernorm_output']] = inter_vars[v['layernorm_output']].mean(1)
            if 'output' in hook['transformer']:
                inter_vars[hook['transformer']['output']] = inter_vars[hook['transformer']['output']].mean(1)
        return hook_return(hook, inter_vars, outputs)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=0, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0:
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            r = [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]
            return r

        student_reps = get_layer_f('s', 'layernorm_output')
        teacher_reps = get_layer_f('t', 'layernorm_output')
        if 'transformer' in s_hook and 'output' in s_hook['transformer']:
            s_inter_vars[s_hook['transformer']['output']].distill = True
            student_reps += [s_inter_vars[s_hook['transformer']['output']]]
            teacher_reps += [t_inter_vars[t_hook['transformer']['output']]]
        for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
            student_rep.distill = teacher_rep.distill = True
        teacher_reps = [i.mean(1) for i in teacher_reps]
        if self.args.rail_kd_concatenated:
            student_reps = torch.cat(student_reps, 1)
            teacher_reps = torch.cat(teacher_reps, 1)
        else:
            student_reps = torch.stack(student_reps, 1)
            teacher_reps = torch.stack(teacher_reps, 1)
        student_reps = aux_layer(self.args, self.s_fit_dense, student_reps)
        teacher_reps = aux_layer(self.args, getattr(self, f't_fit_dense_{t_no}'), teacher_reps)
        student_reps = F.normalize(student_reps, p=2, dim=-1)
        teacher_reps = F.normalize(teacher_reps, p=2, dim=-1)
        l = F.mse_loss(student_reps, teacher_reps, reduction='none')
        if keep_batch:
            l = l.sum(list(range(1, len(l.shape))))
        else:
            l = l.sum()
        super().add_summary(f'inter_loss/hidden_states', l)
        loss_ += l
        loss_ = loss_ * self.args.rail_kd_inter_rate
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_


class MGSKD(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        n_relation_heads = self.args.mgskd_multi_heads
        k1 = self.args.mgskd_triplet_k1
        k2 = self.args.mgskd_triplet_k2
        # 外部调用, 多教师兼容性较差, 例如 keep_batch/no_grad 问题
        self.sample_loss = SampleLoss(n_relation_heads)
        self.tokenphrase_loss = TokenPhraseLoss(n_relation_heads, k1, k2)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
        layers = tuple(range(0, self.args.teacher_num_layers + 1, layers_per_block))
        hook = {} if self.args.mgskd_wo_inter else {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[: -1]},
            'output': None,
        }}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        layers = tuple(range(self.args.num_layers + 1))
        hook = {} if self.args.mgskd_wo_inter else {'transformer': {
            'layers': {i: {'layernorm_output': None} for i in layers[: -1]},
            'output': None,
        }, 'position_ids': None}
        hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, tokenizer=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or self.args.mgskd_wo_inter:
            return loss_
        def get_layer_f(st, name):
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            return [inter_vars[i[1][name]] for i in sorted(hook['transformer']['layers'].items()) if name in i[1]]

        student_reps = get_layer_f('s', 'layernorm_output')
        teacher_reps = get_layer_f('t', 'layernorm_output')
        if 'transformer' in s_hook and 'output' in s_hook['transformer']:
            student_reps += [s_inter_vars[s_hook['transformer']['output']]]
            teacher_reps += [t_inter_vars[t_hook['transformer']['output']]]
        # mask
        mask_pad_token = (s_inter_vars[s_hook['position_ids']][:, 0] > 0).int()
        mask_pad_token[:, 0] = 1
        if self.args.mgskd_span_max_rate:
            seq_len = student_reps[0].size(1)
            span_max_num = int(seq_len * self.args.mgskd_span_max_rate)
            a = abs(torch.normal(0, 1, (span_max_num,)))
            a = (a / a.sum() * (seq_len - span_max_num)).int() + 1
            a[a.sum()-seq_len:] += 1
            mask_pad_span = torch.split(mask_pad_token, a.tolist(), dim=1)
            mask_pad_span = [rep.sum(1, keepdim=True) for rep in mask_pad_span]
            mask_pad_span = (torch.cat(mask_pad_span, 1) > 0).int()
        # loss
        for i, (s_rep, t_rep) in enumerate(zip(student_reps, teacher_reps)):
            s_rep.distill = t_rep.distill = True
            token_loss = phrase_loss = sample_loss = 0.
            # token span
            if i < self.args.mgskd_sample_level_m:
                if self.args.mgskd_span_max_rate:
                    s_span = torch.split(s_rep, a.tolist(), dim=1)
                    s_span = [rep.mean(1, keepdim=True) for rep in s_span]
                    s_span = torch.cat(s_span, 1)
                    t_span = torch.split(t_rep, a.tolist(), dim=1)
                    t_span = [rep.mean(1, keepdim=True) for rep in t_span]
                    t_span = torch.cat(t_span, 1)
                else:
                    ...
                token_loss = self.tokenphrase_loss(s_rep, t_rep, mask_pad_token)
                phrase_loss = self.tokenphrase_loss(s_span, t_span, mask_pad_span)
            else:
                sample_loss = self.sample_loss(s_rep, t_rep, mask_pad_token)
            l = token_loss * self.args.mgskd_weight_token + \
                phrase_loss * self.args.mgskd_weight_span + \
                sample_loss * self.args.mgskd_weight_sample
            super().add_summary(f'inter_loss/hidden_states.{i}', l)
            loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_


class DIITO(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
        self.show_hook_op = True
        self.last_iter = 0
        self.s_alignment_layer = 0  # 用于保存这一批次随机生成的层,full情况下
        # interchange
        self.s_interchanged_variables = None  # {layer:tensor,..}
        self.t_interchanged_variables = {}  # {t_no:{layer:tensor,..},..}
        self.interchange_mask = None  # [16,512]; 原始mask
        self.dual_interchange_mask = None  # [16,512]; 打乱后的mask

    def get_hook(self, st='s', t_no=0, **kwargs):
        # 获取需要抽取的中间层
        if self.args.diito_alignment == 'full':
            if self.args.iteration == self.last_iter and self.s_alignment_layer:
                layer = self.s_alignment_layer
            else:
                layer = self.s_alignment_layer = random.choice(range(1, self.args.num_layers + 1))
                super().add_summary(f'other/s_alignment_layer', layer)
            if st=='s':
                alignment_hook = {'transformer': {
                    **({} if layer==self.args.num_layers else {'layers': {layer: {'layernorm_output': None}}}),
                    **({'output': None} if layer==self.args.num_layers else {}),
                }}
            else:
                layers_per_block = int(self.args.teacher_num_layers / self.args.num_layers)
                layers = list(range((layer - 1) * layers_per_block + 1, layer * layers_per_block + 1))
                alignment_hook = {'transformer': {
                    'layers': {l: {'layernorm_output': None} for l in layers if l != self.args.teacher_num_layers},
                    **({'output': None} if layers[-1]==self.args.teacher_num_layers else {}),
                }}
        elif self.args.diito_alignment == 'middle':
            if st=='s':
                alignment_hook = {'transformer': {
                    'layers': {int(self.args.num_layers / 2) + 1: {'layernorm_output': None}},
                }}
            else:
                alignment_hook = {'transformer': {
                    'layers': {int(self.args.teacher_num_layers / 2) + 1: {'layernorm_output': None}},
                }}
        elif self.args.diito_alignment == 'late':
            if st=='s':
                alignment_hook = {'transformer': {
                    'layers': {self.args.num_layers-1: {'layernorm_output': None}},
                    'output': None
                }}
            else:
                alignment_hook = {'transformer': {
                    'layers': {self.args.teacher_num_layers-1: {'layernorm_output': None}},
                    'output': None
                }}
        else:
            raise NameError(f'error diito_alignment: {self.args.diito_alignment}')
        self.last_iter = self.args.iteration
        # 其他层
        hook_L = []
        if self.args.forward_repeat_current_n == 0:
            hook_L.append(alignment_hook)
            if self.args.diito_alpha_cos:
                hook_L.append({'transformer': {'output': None}})
        else:
            if self.args.diito_alpha_causal_cos:
                hook_L.append({'transformer': {'output': None}})
        return merge_dict(hook_L)

    def get_teacher_hook(self, t_no=0, **kwargs):
        hook_L = [self.get_hook(st='t', t_no=t_no)]
        if self.args.forward_repeat_current_n <= 0:
            hook_L.append(super().get_teacher_hook(t_no=0, **kwargs))
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [self.get_hook(st='s')]
        if self.args.forward_repeat_current_n <= 0:
            hook_L.append(super().get_student_hook(**kwargs))
        hook_L.append({'position_ids': None})
        return merge_dict(hook_L)

    def get_hook_op(self, interchanged_variables, st='s', t_no=0):
        def hook_op(interchanged_variable, interchange_mask=self.interchange_mask, 
            dual_interchange_mask=self.dual_interchange_mask, show=self.show_hook_op, st=st, t_no=t_no, layer=0):
            if self.args.checkpoint_activations:  # compatible deepspeed backward
                interchanged_variable = interchanged_variable.detach()
            if self.args.fp16:
                interchanged_variable = fp32_to_fp16(interchanged_variable)
            def hook_op_(layernorm_output=None, output=None):
                def stat_f(v):
                    return {'shape': v.shape, 'mean': '%e'%v.mean().item(),}

                var = layernorm_output if output is None else output
                if show:
                    print_rank_0(f'DIITO.{st}_{t_no}.layer({layer})_origin: {str(stat_f(var))}')
                var[interchange_mask] = interchanged_variable[dual_interchange_mask]
                if show:
                    print_rank_0(f'DIITO.{st}_{t_no}.layer({layer}): {str(stat_f(var))}')
                return var
            return hook_op_
        hook_ops = {'transformer': {'layers': {
            k: {
                'layernorm_output': hook_op(v, layer=k)
            } for k, v in interchanged_variables.items() if k != 'output'
        }, 'output': hook_op(
            interchanged_variables['output'], layer='output'
        ) if 'output' in interchanged_variables else None}}
        return hook_ops

    def get_teacher_hook_op(self, t_no=0, **kwargs):
        if self.args.forward_repeat_current_n <= 0:
            return {}
        return self.get_hook_op(self.t_interchanged_variables[t_no], st='t', t_no=t_no)

    def get_student_hook_op(self, **kwargs):
        if self.args.forward_repeat_current_n <= 0:
            return {}
        return self.get_hook_op(self.s_interchanged_variables, st='s')

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=0, **kwargs):
        loss_ = 0.
        if self.args.forward_repeat_current_n > 0:
            self.show_hook_op = False
        if len(s_inter_vars) == 0:
            return loss_
        # record interchange
        mask = (s_inter_vars[s_hook['position_ids']][:, 0] > 0).int()
        mask[:, 0] = 1
        def get_interchanged_variables_f(st='s'):
            ret = {}  # {layer:tensor,..}; layer == int or 'output'
            inter_vars, hook = (s_inter_vars, s_hook) if st == 's' else (t_inter_vars, t_hook)
            if 'transformer' in hook and 'layers' in hook['transformer']:
                for i in sorted(hook['transformer']['layers'].items()):
                    a = inter_vars[i[1]['layernorm_output']]
                    ret[i[0]] = torch.cat([a[1:],a[0:1]])
            if self.args.diito_alignment == 'full' and \
                self.s_alignment_layer == self.args.num_layers or \
                self.args.diito_alignment == 'late':
                a = inter_vars[hook['transformer']['output']]
                ret['output'] = torch.cat([a[1:],a[0:1]])
            return ret

        if self.args.forward_repeat_current_n <= 0:
            if t_no==0:
                lengths = mask.sum(-1)
                self.interchange_mask, self.dual_interchange_mask = self.prepare_interchange_mask(
                    lengths,
                    dual_lengths=torch.cat([lengths[1:],lengths[0:1]]),
                    pred_mask=mask,
                    dual_pred_mask=torch.cat([mask[1:],mask[0:1]]),
                )
                self.s_interchanged_variables = get_interchanged_variables_f('s')
            self.t_interchanged_variables[t_no] = get_interchanged_variables_f('t')
        # loss
        if self.args.forward_repeat_current_n > 0:
            diito_alpha_cos = self.args.diito_alpha_causal_cos
        else:
            diito_alpha_cos = self.args.diito_alpha_cos
        if diito_alpha_cos <= 0.:
            return loss_
        s_o = s_inter_vars[s_hook['transformer']['output']]
        t_o = t_inter_vars[t_hook['transformer']['output']]
        s_o.distill = t_o.distill = True
        assert s_o.size() == t_o.size(), f'{s_o.size()} == {t_o.size()}'
        mask = mask.unsqueeze(-1)
        s_o = (s_o * mask).view(-1, s_o.size(-1))
        t_o = (t_o * mask).view(-1, t_o.size(-1))
        l = CustomLoss.cos_distance(s_o, t_o, keep_batch=keep_batch) * diito_alpha_cos
        super().add_summary(f'inter_loss/diito_alpha_cos_{self.args.forward_repeat_current_n}', l)
        loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, **kwargs)
        return loss_

    def pre_loss(self, s_logits, t_logits, loss, **kwargs):
        distill_soft_rate = self.args.distill_soft_rate
        distill_hard_rate = self.args.distill_hard_rate
        summary_suffix = self.summary_suffix
        if self.args.forward_repeat_current_n > 0:
            self.args.distill_soft_rate = self.args.diito_alpha_causal_ce
            self.args.distill_hard_rate = 0.
            self.summary_suffix += f'_causal{self.args.forward_repeat_current_n}'
        else:
            self.args.distill_soft_rate = self.args.diito_alpha_ce
            self.args.distill_hard_rate = self.args.diito_alpha_mlm
        loss_ = super().pre_loss(s_logits, t_logits, loss, **kwargs)
        self.args.distill_soft_rate = distill_soft_rate
        self.args.distill_hard_rate = distill_hard_rate
        self.summary_suffix = summary_suffix
        return loss_

    def prepare_interchange_mask(self, lengths, dual_lengths, pred_mask, dual_pred_mask):
        # params
        interchange_prop = self.args.diito_interchange_prop
        interchange_max_token = self.args.diito_interchange_max_token  # if -1 then we don't restrict on this.
        interchange_masked_token_only = True if self.args.diito_interchange_way == 'masked' else False
        interchange_consecutive_only = True if self.args.diito_interchange_way == 'consecutive' else False
        # source: https://github.com/frankaging/Causal-Distill/blob/main/distillation/causal_distiller.py#L430
        interchange_mask = torch.zeros_like(pred_mask, dtype=torch.bool)
        dual_interchange_mask = torch.zeros_like(dual_pred_mask, dtype=torch.bool)
        batch_size, max_seq_len = pred_mask.shape[0], pred_mask.shape[1]
        _, dual_max_seq_len = dual_pred_mask.shape[0], dual_pred_mask.shape[1]
        interchange_position = []
        for i in range(0, batch_size):
            min_len = min(lengths[i].tolist(), dual_lengths[i].tolist())
            if interchange_consecutive_only:
                if interchange_max_token != -1:
                    interchange_count = min(interchange_max_token, int(min_len*interchange_prop))
                else:
                    interchange_count = int(min_len*interchange_prop)
                start_index = random.randint(0, lengths[i].tolist()-interchange_count)
                end_index = start_index + interchange_count
                dual_start_index = random.randint(0, dual_lengths[i].tolist()-interchange_count)
                dual_end_index = dual_start_index + interchange_count
                interchange_mask[i][start_index:end_index] = 1
                dual_interchange_mask[i][dual_start_index:dual_end_index] = 1
            else:
                # we follow these steps to sample the position:
                # 1. sample positions in the main example
                # 2. get the actual sampled positions
                # 3. sample accordingly from the dual example
                if interchange_masked_token_only:
                    # a corner case we need to consider is that the masked token
                    # numbers may differ across two examples.
                    interchange_count = pred_mask[i].sum()
                    if interchange_count > dual_lengths[i]:
                        # not likely, but we need to handle this.
                        interchange_count = dual_lengths[i]
                    interchange_position = pred_mask[i].nonzero().view(-1).tolist()
                    interchange_position = random.sample(interchange_position, interchange_count)
                    interchange_mask[i][interchange_position] = 1
                    dual_interchange_position = random.sample(range(dual_max_seq_len), interchange_count)
                    dual_interchange_mask[i][dual_interchange_position] = 1
                else:
                    if interchange_max_token != -1:
                        interchange_count = min(interchange_max_token, int(min_len*interchange_prop))
                    else:
                        interchange_count = int(min_len*interchange_prop)
                    interchange_position = random.sample(range(max_seq_len), interchange_count)
                    interchange_mask[i][interchange_position] = 1
                    dual_interchange_position = random.sample(range(dual_max_seq_len), interchange_count)
                    dual_interchange_mask[i][dual_interchange_position] = 1
        # sanity checks
        assert interchange_mask.long().sum(dim=-1).tolist() == \
                dual_interchange_mask.long().sum(dim=-1).tolist()
        return interchange_mask, dual_interchange_mask


class LogitsDistil(GLMStudent):
    def __init__(self, language_model: GLMModel, args, **kwargs):
        super().__init__(language_model, args, **kwargs)

    def get_teacher_hook(self, **kwargs):
        hook_L = [super().get_teacher_hook(**kwargs)]
        if not self.args.logitsdistil_wo_inter:
            hook = {'logits_parallel': None}
            hook_L.append(hook)
        return merge_dict(hook_L)

    def get_student_hook(self, **kwargs):
        hook_L = [super().get_student_hook(**kwargs)]
        if not self.args.logitsdistil_wo_inter:
            hook = {'logits_parallel': None}
            if self.args.logitsdistil_mask_pad:
                hook['position_ids'] = None
            hook_L.append(hook)
        return merge_dict(hook_L)

    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=False, t_no=None, **kwargs):
        loss_ = 0.
        if len(s_inter_vars) == 0 or self.args.logitsdistil_wo_inter:
            return loss_
        s_logits = s_inter_vars[s_hook['logits_parallel']]
        t_logits = t_inter_vars[t_hook['logits_parallel']]
        s_logits.distill = t_logits.distill = True
        if self.args.logitsdistil_mask_pad:
            position_ids = s_inter_vars[s_hook['position_ids']]
            position_ids.distill = True
            mask = (position_ids[:, 0] > 0).int()
            mask[:, 0] = 1
            mask = mask.unsqueeze(dim=-1)
            s_logits = s_logits * mask
            t_logits = t_logits * mask
        if self.args.distill_logit_mask_map and self.origin_model.map_vocab_size:
            origin_id_mask_map = self.origin_model.map_vocab_paras['origin_id_mask_map']
            origin_id_mask_map = mpu.scatter_to_model_parallel_region(origin_id_mask_map)
            s_logits = s_logits[..., origin_id_mask_map]
            t_logits = t_logits[..., origin_id_mask_map]
        # logits 处理
        top_n = self.args.logitsdistil_top_n
        if top_n:
            t_vals_parallel, t_indices_parallel = t_logits.topk(k=top_n, dim=-1, largest=True, sorted=True)
            if self.args.logitsdistil_teacher_min:
                t_vals_max_min = t_vals_parallel.min(dim=-1, keepdim=True)[0]
                t_vals_min = t_logits.min(dim=-1, keepdim=True)[0]
                if mpu.get_model_parallel_world_size() > 1:
                    t_vals_max_min = mpu.gather_from_model_parallel_region(t_vals_max_min)
                    t_vals_max_min = t_vals_max_min.min(dim=-1, keepdim=True)[0]
                    t_vals_min = mpu.gather_from_model_parallel_region(t_vals_min)
                    t_vals_min = t_vals_min.min(dim=-1, keepdim=True)[0]
                s_logits_, parallel = s_logits, 'gather'
                t_logits_ = torch.where(t_logits >= t_vals_max_min, t_logits, t_vals_min)
            else:
                s_vals_parallel = torch.gather(s_logits, -1, t_indices_parallel)
                keep_batch_dim = -1 if keep_batch else []
                if mpu.get_model_parallel_world_size() > 1:
                    t_vals = mpu.gather_from_model_parallel_region(t_vals_parallel)
                    s_vals = mpu.gather_from_model_parallel_region(s_vals_parallel)
                    # logits_
                    t_vals_top, t_vals_top_indices = t_vals.topk(k=top_n, dim=-1, largest=True, sorted=True)
                    s_vals_top = torch.gather(s_vals, -1, t_vals_top_indices)
                    s_logits_, t_logits_, parallel = s_vals_top, t_vals_top, ''
                    # threshold loss 1
                    threshold = s_vals_top.min(dim=-1, keepdim=True)[0]
                    loss1 = torch.where(s_logits > threshold, s_logits - threshold, s_logits.new_zeros(s_logits.size()))
                    loss1.scatter_(-1, t_indices_parallel, 0)
                    count_nonzero = loss1.count_nonzero(keep_batch_dim)
                    loss1 = loss1.sum(keep_batch_dim) / count_nonzero
                    loss1 = mpu.reduce_from_model_parallel_region(loss1) / mpu.get_model_parallel_world_size()
                    self.add_summary('inter_loss/s_logits_threshold', loss1)
                    count_nonzero = mpu.reduce_from_model_parallel_region(count_nonzero)
                    self.add_summary('inter_loss/s_logits_threshold_count_nonzero', count_nonzero)
                    loss_ += loss1
                    # threshold loss 2
                    loss2 = torch.where(s_vals > threshold, s_vals - threshold, s_vals.new_zeros(s_vals.size()))
                    loss2.scatter_(-1, t_vals_top_indices, 0)
                    count_nonzero = loss2.count_nonzero(keep_batch_dim)
                    loss2 = loss2.sum(keep_batch_dim) / count_nonzero
                    self.add_summary('inter_loss/s_vals_threshold', loss2)
                    self.add_summary('inter_loss/s_vals_threshold_count_nonzero', count_nonzero)
                    loss_ += loss2
                else:
                    s_logits_, t_logits_, parallel = s_vals_parallel, t_vals_parallel, ''
                    # threshold loss 1
                    threshold = s_vals_parallel.min(dim=-1, keepdim=True)[0]
                    loss1 = torch.where(s_logits > threshold, s_logits - threshold, s_logits.new_zeros(s_logits.size()))
                    loss1.scatter_(-1, t_indices_parallel, 0)
                    count_nonzero = loss1.count_nonzero(keep_batch_dim)
                    loss1 = loss1.sum(keep_batch_dim) / count_nonzero
                    self.add_summary('inter_loss/s_logits_threshold', loss1)
                    self.add_summary('inter_loss/s_logits_threshold_count_nonzero', count_nonzero)
                    loss_ += loss1
        else:
            s_logits_, t_logits_, parallel = s_logits, t_logits, 'gather'
        # loss
        if self.args.logitsdistil_mse:
            l = CustomLoss.mse_loss(s_logits_, t_logits_, parallel=parallel, keep_batch=keep_batch)
        else:
            T = self.args.distill_temperature
            l = CustomLoss.kl_div(s_logits_ / T, t_logits_ / T, parallel=parallel, keep_batch=keep_batch) * T ** 2
        self.add_summary('inter_loss/logits_parallel', l)
        loss_ += l
        loss_ += super().inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, keep_batch=keep_batch, t_no=None, **kwargs)
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
    'rail_kd': RAIL_KD,
    'mgskd': MGSKD,
    'diito': DIITO,
    'logitsdistil': LogitsDistil,
}
