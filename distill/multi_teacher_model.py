import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
from utils import print_rank_0
import mpu
import math
from distill.tools import aux_layer, get_checkpoint_forward_args
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from mpu import checkpoint, get_cuda_rng_tracker
import deepspeed
from distill.custom_loss import CustomLoss


class AvgTeacher(torch.nn.Module):
    def __init__(self, args, show_inter=True, show_pre=True, **kwargs):
        super().__init__()
        self.args = args
        self.show_inter = show_inter
        self.show_pre = show_pre
        self.max_forward_repeat_current_n = 0
        self.show_inter_origin = show_inter
        self.show_pre_origin = show_pre

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def record_and_show(self, student_model, op='init', t_no=-1, loss=0):
        # 初始化显示和记录的参数
        if op == 'init':
            self.inter_show_hooks = {}
            self.pre_loss_description = []
            student_model.show_inter = False
            student_model.show_pre = False
            self.summary_suffix_s = student_model.summary_suffix
        # 计算单个教师模型结果之前的准备
        elif op == 't_start':
            student_model.summary_suffix = self.summary_suffix_s + f'_t{t_no}'
            if self.args.forward_repeat_current_n > 0:
                student_model.summary_suffix += f'_repeat{self.args.forward_repeat_current_n}'
        # 计算完教师模型结果之后的处理
        elif op == 't_end':
            # 记录loss联合方式
            self.inter_show_hooks.update(student_model.inter_show_hooks)
            if 'teacher' in self.inter_show_hooks:
                self.inter_show_hooks[f't_{t_no}'] = self.inter_show_hooks['teacher']
            if 'student' in self.inter_show_hooks:
                self.inter_show_hooks[f's_{t_no}'] = self.inter_show_hooks['student']
            prefix = f's-t_{t_no}'
            if self.args.forward_repeat_current_n > 0:
                prefix += f'_repeat{self.args.forward_repeat_current_n}'
            self.pre_loss_description.append(f'{prefix}: ' + student_model.pre_loss_description)
            # 记录 tensorboard
            student_model.add_summary(f'teacher_loss/teacher_{t_no}', loss)
            student_model.summary_suffix = self.summary_suffix_s  # 复原
        # 最终处理展示
        elif op == 'final_show':
            if self.args.forward_repeat_current_n > self.max_forward_repeat_current_n:
                self.show_inter = self.show_inter_origin
                self.show_pre = self.show_pre_origin
                self.max_forward_repeat_current_n = self.args.forward_repeat_current_n
            if self.show_pre:
                print_rank_0('\n'.join(self.pre_loss_description))
                self.show_pre = False
            if self.show_inter:
                if 'teacher' in self.inter_show_hooks:
                    del self.inter_show_hooks['teacher']
                if 'student' in self.inter_show_hooks:
                    del self.inter_show_hooks['student']
                print_rank_0(self.inter_show_hooks)
                self.show_inter = False
        else:
            raise NameError(f'未知的 op={op} !')

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            # pre_loss
            pre_loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_out['loss'], loss_mask=loss_mask, labels=labels, t_no=i)
            # inter_loss
            inter_loss = student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels, t_no=i)
            # loss
            loss = pre_loss + inter_loss
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        return sum(loss_L) / len(loss_L)

    def hooks_process(self, t_hook_L, **kwargs):
        # get_teachers_hook 之后的结果再处理
        return t_hook_L


class MT_BERT(AvgTeacher):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        mt_hidden_size = [args.teacher_hidden_size]
        if args.mt_hidden_size and not args.mt_bert_wo_convert_layer:  # W
            mt_hidden_size = [int(i) for i in args.mt_hidden_size.split(':')]
            for i, hidden_size in enumerate(mt_hidden_size):
                if args.mt_bert_fit_teacher:
                    i_hs, o_hs = hidden_size, args.hidden_size
                else:
                    i_hs, o_hs = args.hidden_size, hidden_size
                setattr(self, f'fit_dense_{i}', mpu.ColumnParallelLinear(i_hs, o_hs))
        if self.args.mt_bert_fix_layernorm:
            self.layernorm = LayerNorm(args.hidden_size)
            for i, hidden_size in enumerate(mt_hidden_size):
                setattr(self, f't_layernorm_{i}', LayerNorm(hidden_size))
                getattr(self, f't_layernorm_{i}').requires_grad_(False)

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            # mt_bert_fix_layernorm
            if self.args.mt_bert_fix_layernorm:
                if 'transformer' in s_hook:
                    s_inter_vars = s_inter_vars.copy()
                    if 'layers' in s_hook['transformer']:
                        for v in s_hook['transformer']['layers'].values():
                            if 'layernorm_output' in v:
                                s_inter_vars[v['layernorm_output']] = aux_layer(self.args, self.layernorm, s_inter_vars[v['layernorm_output']])
                    if 'output' in s_hook['transformer']:
                        s_inter_vars[s_hook['transformer']['output']] = aux_layer(self.args, self.layernorm, s_inter_vars[s_hook['transformer']['output']])
                if 'transformer' in t_hook:
                    t_inter_vars = t_inter_vars.copy()
                    if 'layers' in t_hook['transformer']:
                        for v in t_hook['transformer']['layers'].values():
                            if 'layernorm_output' in v:
                                t_inter_vars[v['layernorm_output']] = aux_layer(self.args, getattr(self, f't_layernorm_{i}'), t_inter_vars[v['layernorm_output']])
                    if 'output' in t_hook['transformer']:
                        t_inter_vars[t_hook['transformer']['output']] = aux_layer(self.args, getattr(self, f't_layernorm_{i}'), t_inter_vars[t_hook['transformer']['output']])
            # 教师/学生中间层 W
            if self.args.mt_bert_fit_teacher:
                hook, inter_vars = t_hook, t_inter_vars.copy()  # 不修改 t_inter_vars_L
            else:
                hook, inter_vars = s_hook, s_inter_vars.copy()  # 不修改 s_inter_vars
            if self.args.mt_hidden_size and 'transformer' in hook and 'layers' in hook['transformer'] \
                and not self.args.mt_bert_wo_convert_layer:
                fit_dense = getattr(self, f'fit_dense_{i}')
                for v in hook['transformer']['layers'].values():
                    if 'layernorm_output' in v:
                        inter_vars[v['layernorm_output']] = aux_layer(self.args, fit_dense, inter_vars[v['layernorm_output']])
                if 'output' in hook['transformer']:
                    inter_vars[hook['transformer']['output']] = aux_layer(self.args, fit_dense, inter_vars[hook['transformer']['output']])
            # pre_loss
            pre_loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_out['loss_batch'], loss_mask=loss_mask, labels=labels, keep_batch=True, t_no=i)
            pre_loss *= 1 / (1 + t_out['loss_batch'])  # 加权, 依赖参数 --mt_has_loss
            # inter_loss
            if self.args.mt_bert_fit_teacher:
                s_iv, t_iv = s_inter_vars, inter_vars
            else:
                s_iv, t_iv = inter_vars, t_inter_vars
            inter_loss = student_model.inter_loss(s_iv, t_iv, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels, t_no=i)
            loss = pre_loss.mean() + inter_loss
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        if self.args.mt_bert_wo_hard:
            return sum(loss_L)
        else:
            student_model.add_summary('multi_teacher_model/hard_loss', s_out['loss'])
            return sum(loss_L) + s_out['loss']  # 这里加入了硬标签, pre_loss 不应再有硬标签参数


class Uncertainty(AvgTeacher):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        # mask
        if self.args.uncertainty_wo_loss_mask or loss_mask is None:
            mask = 1.
        elif labels is not None and self.args.uncertainty_only_mask_pad:
            mask = labels.view(*labels.size(), 1) > 0
        else:
            mask = loss_mask.view(*loss_mask.size(), 1)
        # entropy
        parallel = 'gather' if self.args.custom_logits_paralle else ''
        def norm_entropy_f(t):  # norm 信息熵
            t = t * mask
            entropy = CustomLoss.info_entropy(t, parallel=parallel, keep_batch=True)
            norm_entropy = entropy / math.log(t.size(-1))
            return norm_entropy
        s_entropy = norm_entropy_f(s_out['logits'])
        # rate: batch_size * teacher_num
        if len(t_hook_L) > 1 and not self.args.uncertainty_wo_rate:
            if self.args.uncertainty_hard and s_entropy.size(0) >= len(t_hook_L):
                sort_i = (-s_entropy).sort(0).indices.sort(0).indices.unsqueeze(-1) + 1.
                sort_i = sort_i / sort_i.size(0) * len(t_hook_L)
                rate = sort_i > torch.arange(0, len(t_hook_L), device=sort_i.device)
                rate *= sort_i <= torch.arange(1, len(t_hook_L) + 1, device=sort_i.device)
            else:
                rate = torch.arange(0, len(t_hook_L), device=s_entropy.device) / (len(t_hook_L) - 1.)
                rate = (1 - 2 * s_entropy).unsqueeze(-1) * rate
                rate = rate + s_entropy.unsqueeze(-1)
                rate = rate / rate.sum(-1).unsqueeze(-1)
        else:
            rate = s_entropy.new_ones([*s_entropy.shape, len(t_hook_L)])
        # 教师大小顺序
        if self.args.uncertainty_teacher_seq:
            t_seq = [int(i) for i in self.args.uncertainty_teacher_seq.split(':')]
            assert len(t_seq) == len(t_hook_L), f'参数数量错误: {len(t_seq)} != {len(t_hook_L)}'
            assert tuple(sorted(t_seq)) == tuple(range(len(t_hook_L))), f'序号要从0到{len(t_hook_L)-1}! {t_seq}'
        else:
            t_seq = list(range(len(t_hook_L)))
        # loss
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            self.record_and_show(student_model, op='t_start', t_no=i)
            # pre_loss
            pre_loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_out['loss_batch'], loss_mask=loss_mask, labels=labels, keep_batch=True, t_no=i)
            pre_loss = (rate[...,t_seq[i]] * pre_loss)
            # inter_loss
            keep_batch = True if self.args.uncertainty_inter_entropy else False
            inter_loss = student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels, keep_batch=keep_batch, t_no=i)
            if self.args.uncertainty_inter_entropy:
                inter_loss = (inter_loss * s_entropy).mean()
                if inter_loss > 0:
                    pre_loss = abs(1 - s_entropy) * pre_loss
            # add loss
            loss = pre_loss.mean() + inter_loss
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        return sum(loss_L) / len(loss_L)


class RL_KD(AvgTeacher):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        # Semantic Representation + Teacher CE Loss
        semantic_len = 0
        tn = len(args.mt_hidden_size.split(':')) if args.mt_hidden_size else 1
        if args.rl_kd_semantic_model is not None:
            semantic_len = int(args.mt_hidden_size.split(':')[args.rl_kd_semantic_model])
            tn -= 1  # multi Teacher CE Loss
        if args.custom_sample_shape:
            sample_shape = [int(i) for i in args.custom_sample_shape.split(',')]
        if len(sample_shape) == 2:  # 分类方式差异
            if self.args.task.lower() in {'record'}:
                semantic_len *= self.get_class_num()
            else:
                semantic_len *= sample_shape[0]
        self.agent_semantic_mt_loss = torch.nn.Linear(semantic_len + tn, tn)
        # Teacher soft labels
        class_dim = self.get_class_num() if self.get_class_num() else args.vocab_size
        if args.custom_logits_paralle:  # 注意这里教师序号等于是连续的
            self.agent_mt_soft = mpu.RowParallelLinear(class_dim * tn, tn, input_is_parallel=True)
        else:
            self.agent_mt_soft = torch.nn.Linear(class_dim * tn, tn)
        # Environment: bs * tn
        self.semantic_mt_loss_rep = None
        self.mt_soft_rep = None
        self.teacher_select = None  # a
    
    def get_class_num(self):
        from tasks.superglue.dataset import PROCESSORS  # 参考
        task_class_num = {  # {任务:当作几分类,..}
            'copa': 2,
            'wsc': 10,
            'record': 10,
            'rte': 2,
            'boolq': 2,
            'wic': 2,
            'cb': 3,
            'multirc': 2,
        }
        if self.args.task.lower() in task_class_num and not self.args.custom_logits_paralle:
            return task_class_num[self.args.task.lower()]
        else:
            return None

    def hooks_process(self, t_hook_L, **kwargs):
        if self.args.rl_kd_semantic_model is not None:
            t_hook_L[self.args.rl_kd_semantic_model] = {'transformer': {'output': None}}
        return t_hook_L
        
    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out,
                loss_mask=None, labels=None, **kwargs):
        loss_L = []
        self.record_and_show(student_model)
        # mask
        if self.args.rl_kd_wo_loss_mask or loss_mask is None:
            mask = 1.
        elif labels is not None and self.args.rl_kd_only_mask_pad:
            mask = labels.view(*labels.size(), 1) > 0
        else:
            mask = loss_mask.view(*loss_mask.size(), 1)
        semantic_mt_loss_rep = []
        mt_soft_rep = []
        # loss
        for i, (t_hook, t_inter_vars, t_out, t_model) in enumerate(zip(t_hook_L, t_inter_vars_L, t_out_L, teacher_models)):
            if i == self.args.rl_kd_semantic_model:
                # Semantic Representation: [CLS]
                if not self.args.rl_kd_only_avg:
                    rep = t_inter_vars[t_hook['transformer']['output']][...,0,:].squeeze(-2)
                    rep = rep.contiguous().view(s_out['loss_batch'].size(0), -1)
                    semantic_mt_loss_rep = [rep.detach().clone()] + semantic_mt_loss_rep
                continue
            self.record_and_show(student_model, op='t_start', t_no=i)
            # other Environment rep
            if not self.args.rl_kd_only_avg:
                semantic_mt_loss_rep.append(t_out['loss_batch'].detach().clone().unsqueeze(-1))
                if len(t_out['logits'].shape) == 3:
                    logits = (t_out['logits'] * mask).mean(-2)
                else:
                    logits = t_out['logits'] * mask
                mt_soft_rep.append(logits.detach().clone())
            # pre_loss
            if self.args.rl_kd_only_avg:
                s_loss, keep_batch = s_out['loss'], False
            else:
                s_loss, keep_batch = s_out['loss_batch'], True
            pre_loss = student_model.pre_loss(s_out['logits'], t_out['logits'], s_loss, loss_mask=loss_mask, labels=labels, keep_batch=keep_batch, t_no=i)
            # inter_loss
            inter_loss = student_model.inter_loss(s_inter_vars, t_inter_vars, s_hook, t_hook, t_model=t_model, loss_mask=loss_mask, labels=labels, keep_batch=keep_batch, t_no=i)
            # loss
            loss = pre_loss + inter_loss
            loss_L.append(loss)
            self.record_and_show(student_model, op='t_end', t_no=i, loss=loss)
        self.record_and_show(student_model, op='final_show')
        if self.args.rl_kd_only_avg:
            final_loss = sum(loss_L) / len(loss_L)
        else:
            # Teacher Selector
            semantic_mt_loss_rep = torch.cat(semantic_mt_loss_rep, -1)
            mt_soft_rep = torch.cat(mt_soft_rep, -1)
            s = aux_layer(self.args, self.agent_semantic_mt_loss, semantic_mt_loss_rep)
            s += aux_layer(self.args, self.agent_mt_soft, mt_soft_rep)
            s = s.sigmoid()
            teacher_select = torch.rand(*s.shape, device=s.device) < s
            final_loss = torch.stack(loss_L, -1) * teacher_select
            final_loss = final_loss.mean()
            # Update agent
            if self.semantic_mt_loss_rep is not None \
                and self.mt_soft_rep is not None \
                and self.teacher_select is not None \
                :  # 隔代更新不用数据重复使用
                s = aux_layer(self.args, self.agent_semantic_mt_loss, self.semantic_mt_loss_rep)
                s += aux_layer(self.args, self.agent_mt_soft, self.mt_soft_rep)
                s = s.sigmoid()
                pi = s * self.teacher_select + (1 - s) * (1 - self.teacher_select * 1)
                reward = [
                    - s_out['loss'],
                    - s_out['loss'] - t_out['loss'],
                ]
                rl_loss = - pi.sum() * reward[self.args.rl_kd_reward - 1].detach().clone()
                student_model.add_summary('multi_teacher_model/rl_loss', rl_loss)
                final_loss = final_loss + rl_loss
            self.semantic_mt_loss_rep = semantic_mt_loss_rep
            self.mt_soft_rep = mt_soft_rep
            self.teacher_select = teacher_select
        if self.args.rl_kd_wo_hard:
            return final_loss
        student_model.add_summary('multi_teacher_model/hard_loss', (1 - self.args.rl_kd_alpha) * s_out['loss'])
        return final_loss * self.args.rl_kd_alpha + (1 - self.args.rl_kd_alpha) * s_out['loss']


class MixMT(AvgTeacher):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.mixmt_model = args.mixmt_model.split(',')
        self.baselines = set(self.mixmt_model)
        for c in self.baselines:
            setattr(self, c, eval(c)(args, **kwargs))
        self.show_c = True

    def hooks_process(self, t_hook_L, **kwargs):
        for c in self.baselines:
            getattr(self, c).hooks_process(t_hook_L, **kwargs)
        return t_hook_L
        
    def compute(self, teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out, **kwargs):
        loss_ = 0.
        for c in self.baselines:
            if self.show_c:
                print(c)
            mt = getattr(self, c)
            l = mt.compute(teacher_models, t_hook_L, t_inter_vars_L, t_out_L, student_model, s_hook, s_inter_vars, s_out, **kwargs)
            student_model.add_summary(f'multi_teacher_model/{c}', l)
            loss_ += l
        self.show_c = False
        return loss_


multi_teacher_model_D = {
    None: AvgTeacher,
    '': AvgTeacher,
    'tmkd': AvgTeacher,
    'mt_bert': MT_BERT,
    'uncertainty': Uncertainty,
    'rl_kd': RL_KD,
    'mixmt': MixMT,
}
