import sys, os
sys.path.append(os.getcwd())

from arguments import get_args as get_args_
import argparse
from train_utils import get_model
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration
from train_utils import load_pretrained
from distill.distill_model import student_model_D, unpacking_student_model
from distill.multi_teacher_model import multi_teacher_model_D
import torch
import time


def get_args():
    py_parser = argparse.ArgumentParser(add_help=False)
    # generic
    py_parser.add_argument('--student_model', type=str, default=None, help='学生模型类型,没有则是原生模型')
    py_parser.add_argument('--student_truncate_tn', type=int, default=None, help='非None或不小于0的话代表选择第几个教师前面层截断作为初始化(长于学生的维度靠前截断参数,短于学生的维度默认学生参数不变)')
    py_parser.add_argument('--distill_ft_soft', action='store_true', help='是否在微调蒸馏阶段使用软标签')
    py_parser.add_argument('--distill_ft_hard', action='store_true', help='是否在微调蒸馏阶段使用硬标签')
    py_parser.add_argument('--distill_pt_soft', action='store_true', help='是否在预训练蒸馏阶段使用软标签')
    py_parser.add_argument('--distill_pt_hard', action='store_true', help='是否在预训练蒸馏阶段使用硬标签')
    py_parser.add_argument('--distill_soft_rate', type=float, default=1., help='蒸馏阶段使用软标签的比例')
    py_parser.add_argument('--distill_hard_rate', type=float, default=1., help='蒸馏阶段使用硬标签的比例,可配合多教师')
    py_parser.add_argument('--distill_temperature', type=float, default=1., help='ce/kl散度蒸馏温度')
    py_parser.add_argument('--distill_wo_loss_mask', action='store_true', help='蒸馏软标签不mask')
    py_parser.add_argument('--distill_only_mask_pad', action='store_true', help='蒸馏软标签只mask padding')
    py_parser.add_argument('--distill_ft_soft_kl', action='store_true', help="使用kl散度计算ft_soft")
    py_parser.add_argument('--distill_pt_soft_ce', action='store_true', help="使用交叉熵计算pt_soft")
    py_parser.add_argument('--distill_ft_soft_mse', action='store_true', help="使用mse计算ft_soft")
    py_parser.add_argument('--distill_pt_soft_mse', action='store_true', help="使用mse计算pt_soft")
    py_parser.add_argument('--distill_logits_parallel', action='store_true', help='是否将logits_parallel当作inter_loss使用,无mask,只有在NLU的ft阶段有价值,其他重复时可能产生soft权重*2的效果,注意一般不受wo_inter类参数的约束')
    # teacher
    py_parser.add_argument('--teacher_num_attention_heads', type=int, default=16)
    py_parser.add_argument('--teacher_hidden_size', type=int, default=1024)
    py_parser.add_argument('--teacher_num_layers', type=int, default=24)
    py_parser.add_argument('--teacher_max_position_embeddings', type=int, default=512)
    py_parser.add_argument('--teacher_load_pretrained', type=str, default=None)
    py_parser.add_argument('--teacher_fp16', action='store_true')

    # tinybert
    py_parser.add_argument('--tinybert_inter_final', action='store_true', help="只使用最后隐层做损失")
    py_parser.add_argument('--tinybert_only_emb_final', action='store_true', help="只使用嵌入层和最后隐层做损失")
    py_parser.add_argument('--tinybert_custom_final', type=int, default=1, help="1代表final指倒数第一层,2代表指倒数第二层")
    py_parser.add_argument('--tinybert_only_emb', action='store_true', help="只使用嵌入层做损失")
    py_parser.add_argument('--tinybert_wo_att', action='store_true', help="不使用注意力矩阵的损失")
    py_parser.add_argument('--tinybert_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")
    py_parser.add_argument('--tinybert_wo_final', action='store_true', help="不使用最后层,不适用于tinybert_only_emb_final,tinybert_custom_final,tinybert_inter_final")
    py_parser.add_argument('--tinybert_wo_emb', action='store_true', help="不使用嵌入层,不适用于tinybert_only_emb_final,tinybert_only_emb")
    py_parser.add_argument('--tinybert_fit_parallel', action='store_true', help='转换层是否使用模型并行')
    py_parser.add_argument('--tinybert_fit_compatible_mt', action='store_true', help='是否使用多个转换层兼容多教师')
    py_parser.add_argument('--tinybert_random_layers', action='store_true', help="是否随机选择中间层(emb和final不动)")
    py_parser.add_argument('--tinybert_random_e', type=int, default=1, help="每几轮训练后随机选择层,大于0有效,优先")
    py_parser.add_argument('--tinybert_random_i', type=int, default=3000, help="每几次迭代后随机选择层,大于0有效,tinybert_random_i为0这个参数才有效")
    py_parser.add_argument('--tinybert_random_show', action='store_true', help="显示每次随机后的教师中间取层(不含emb/final)")
    # minilmv2
    py_parser.add_argument('--minilmv2_relation_heads', type=int, default=48, help="base=48,large=64")
    py_parser.add_argument('--minilmv2_teacher_layer', type=int, default=12, help="start at one")
    py_parser.add_argument('--minilmv2_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")
    # distilbert
    py_parser.add_argument('--distilbert_alpha_ce', type=float, default=1., help="类似 distill_pt_soft")
    py_parser.add_argument('--distilbert_alpha_mlm', type=float, default=1., help="类似 distill_pt_hard")
    py_parser.add_argument('--distilbert_alpha_cos', type=float, default=1.)
    py_parser.add_argument('--distilbert_fix_layernorm', action='store_true')
    py_parser.add_argument('--distilbert_cos_mask_padding', action='store_true', help='隐层只mask padding')
    py_parser.add_argument('--distilbert_ce_mask_padding', action='store_true', help='软标签只mask padding')
    # mixbaseline
    py_parser.add_argument('--mixbaseline_wo_inter', action='store_true', help="不使用中间层,用于二次微调")
    py_parser.add_argument('--mixbaseline_tinybert_t', type=float, default=1., help="专用的temperature")
    py_parser.add_argument('--mixbaseline_inter_bl', type=str, default='', help='TinyBERT,MiniLMv2,MiniLM,DistilBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_pt_soft', type=str, default='', help='DistilBERT,TinyBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_ft_soft', type=str, default='', help='TinyBERT')
    # pkd
    py_parser.add_argument('--pkd_normalized_patience', action='store_true')
    py_parser.add_argument('--pkd_alpha', type=float, default=0.5, help="soft权重")
    py_parser.add_argument('--pkd_beta', type=float, default=100., help="中间层权重")
    py_parser.add_argument('--pkd_use_embed', action='store_true', help="中间层是否包括嵌入层")
    py_parser.add_argument('--pkd_wo_final', action='store_true', help="中间层是否去除最后一层")
    # rail_kd
    py_parser.add_argument('--rail_kd_inter_rate', type=float, default=0.3334, help="中间层权重")
    py_parser.add_argument('--rail_kd_layer_wise_alpha', type=float, default=1., help="Layer-wise RAIL-KD方法的权重alpha i")
    py_parser.add_argument('--rail_kd_u', type=int, default=128, help="层变换后的维度")
    py_parser.add_argument('--rail_kd_epochs', type=int, default=1, help="每几轮训练后随机选择层,大于0有效,优先")
    py_parser.add_argument('--rail_kd_iters', type=int, default=3000, help="每几次迭代后随机选择层,大于0有效,rail_kd_epochs为0这个参数才有效")
    py_parser.add_argument('--rail_kd_concatenated', action='store_true', help="是否使用Concatenated RAIL-KD方法")
    py_parser.add_argument('--rail_kd_has_embed', action='store_true', help="中间层是否包括嵌入层")
    py_parser.add_argument('--rail_kd_has_final', action='store_true', help="中间层是否包含最后一层")
    py_parser.add_argument('--rail_kd_show_hook_change', action='store_true', help="显示每次随机后的教师取层")
    py_parser.add_argument('--rail_kd_no_random', action='store_true', help="取消该方法的随机取层,变成隔层取")
    # mgskd
    py_parser.add_argument('--mgskd_weight_sample', type=float, default=4., help="权重")
    py_parser.add_argument('--mgskd_weight_token', type=float, default=1., help="权重")
    py_parser.add_argument('--mgskd_weight_span', type=float, default=1., help="权重")
    py_parser.add_argument('--mgskd_sample_level_m', type=int, default=2, help="从这个层开始使用sample损失,一般为学生层数/2")
    py_parser.add_argument('--mgskd_triplet_k1', type=int, default=20, help="对注意力分数排名前几的向量使用 Triplet-wise Geometric Angle")
    py_parser.add_argument('--mgskd_triplet_k2', type=int, default=20, help="对k1个向量中注意力分数排名前几的点拿出来组成新的矩阵")
    py_parser.add_argument('--mgskd_multi_heads', type=int, default=64, help="隐层切分成多头的数量")
    py_parser.add_argument('--mgskd_span_max_rate', type=float, default=0.4, help="大于0则随机分割词组使用,相对于整体序列长度的比例")
    py_parser.add_argument('--mgskd_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")

    # multi-teacher 多个教师的模型参数用冒号分隔, 优先级高于 teacher_ 参数
    py_parser.add_argument('--mt_num_attention_heads', type=str, default='')
    py_parser.add_argument('--mt_hidden_size', type=str, default='')
    py_parser.add_argument('--mt_num_layers', type=str, default='')
    py_parser.add_argument('--mt_max_position_embeddings', type=str, default='')
    py_parser.add_argument('--mt_load_pretrained', type=str, default='')
    # multi-teacher model (指将多个教师联合在一起的模型)
    py_parser.add_argument('--multi_teacher_model', type=str, default=None, help='多教师模型名称')
    py_parser.add_argument('--mt_model_load', type=str, default=None, help='可选额外加载的多教师模型路径,可以自动从其他学生模型路径中提取')
    py_parser.add_argument('--mt_has_loss', action='store_true', help='是否每个教师都需要计算最终loss,配合某些多教师模型')
    py_parser.add_argument('--mt_has_grad', action='store_true', help='是否每个教师都需要梯度,是的话教师模型会作为学生模型的一部分进行更新')
    py_parser.add_argument('--student_use_empty_glm', action='store_true', help='学生模型中的glm模型置空,可配合mt_has_grad训练活的多教师')
    py_parser.add_argument('--mt_load_from_s', type=str, default=None, help='从整合多教师模型的学生模型路径中加载多教师的参数,将替代teacher_/mt_load_pretrained,mt_*参数中多教师顺序与当初保存的要一致')
    # mt_bert
    py_parser.add_argument('--mt_bert_fit_teacher', action='store_true', help='内层变换是否针对教师,否则是学生')
    py_parser.add_argument('--mt_bert_wo_hard', action='store_true', help='取消默认自带的硬标签')
    py_parser.add_argument('--mt_bert_wo_convert_layer', action='store_true', help='取消自带的神经网络层转换,可用于学生自带或相同隐层不需要')
    py_parser.add_argument('--mt_bert_fix_layernorm', action='store_true')
    # uncertainty
    py_parser.add_argument('--uncertainty_wo_loss_mask', action='store_true', help='NLG的logits熵不mask')
    py_parser.add_argument('--uncertainty_only_mask_pad', action='store_true', help='NLG的logits熵只mask padding')
    py_parser.add_argument('--uncertainty_inter_entropy', action='store_true', help='是否用信息熵方式处理inter_loss权重')
    py_parser.add_argument('--uncertainty_teacher_seq', type=str, default=None, help='教师模型从小到大的序号顺序(从0开始),默认mt_*参数是从小到大,冒号分隔')
    py_parser.add_argument('--uncertainty_hard', action='store_true', help='pre_loss Hard Selection,要求单卡batch size大于等于教师数量')
    py_parser.add_argument('--uncertainty_wo_rate', action='store_true', help='是否不使用软标签的权重')
    # rl_kd
    py_parser.add_argument('--rl_kd_wo_loss_mask', action='store_true', help='用于agent-NLG的logits不mask')
    py_parser.add_argument('--rl_kd_only_mask_pad', action='store_true', help='用于agent-NLG的logits只mask padding')
    py_parser.add_argument('--rl_kd_reward', type=int, default=1, help='reward type')
    py_parser.add_argument('--rl_kd_semantic_model', type=int, default=None, help='第几个教师模型会拿来做Environment的Semantic Representation,这个教师模型将不参与其他计算,默认不使用Semantic')
    py_parser.add_argument('--rl_kd_only_avg', action='store_true', help='只使用平均教师loss不使用强化学习')
    py_parser.add_argument('--rl_kd_wo_hard', action='store_true', help='取消默认自带的硬标签')
    py_parser.add_argument('--rl_kd_alpha', type=float, default=0.5, help='非自带硬标签部分的权重,(1-权重)为自带硬标签的权重,保留默认自带的硬标签才生效')
    # mixmt
    py_parser.add_argument('--mixmt_model', type=str, default='', help='AvgTeacher,MT_BERT,Uncertainty,RL_KD')

    known, args_list = py_parser.parse_known_args()
    args = get_args_(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # check args
    get_teachers_hook(args)
    if args.student_truncate_tn is not None and args.student_truncate_tn < 0:
        args.student_truncate_tn = None
    return args


def glm_wrap(model, args, teacher_models=None):
    # 学生模型构建, 包含多教师模型
    student_model = student_model_D[args.student_model]
    if student_model is None:
        return model
    student_model = student_model(model, args)
    multi_teacher_model = multi_teacher_model_D[args.multi_teacher_model](args)
    setattr(student_model, 'multi_teacher_model', multi_teacher_model)
    if teacher_models and args.mt_has_grad:  # 活的教师模型
        for i, teacher_model in enumerate(teacher_models):
            setattr(student_model, f'teacher_model_{i}', teacher_model)
    return student_model
    

def get_teacher_model(args, **kwargs):
    # 构建多个教师模型并加载
    if not (args.teacher_load_pretrained or args.mt_num_attention_heads):
        return None
    transfer_vars = [
        'num_attention_heads',
        'hidden_size',
        'num_layers',
        'max_position_embeddings',
        'load_pretrained',
    ]
    original_vars = [getattr(args, i) for i in transfer_vars]
    fp16 = args.fp16
    # 统一参数加载
    if args.mt_load_from_s:
        load_dir, tag, release, success = get_checkpoint_iteration(args.mt_load_from_s)
        checkpoint_name = get_checkpoint_name(load_dir, tag, release)
        sd = torch.load(checkpoint_name, map_location='cpu')['module']
    else:
        sd = {}
    # 替换
    args.fp16 = args.teacher_fp16
    teacher_models = []
    if args.mt_num_attention_heads:
        paras = zip(*[getattr(args, 'mt_' + i).split(':') for i in transfer_vars])
    else:
        paras = [[getattr(args, 'teacher_' + i) for i in transfer_vars]]
    for i, vars in enumerate(paras):
        print_rank_0(f'加载 {i} 号教师模型... ' + str(dict(zip(transfer_vars, vars))))
        for name, v, original_v in zip(transfer_vars, vars, original_vars):
            if name == 'max_position_embeddings':
                if original_v > int(v):  # 主要解决NLG序列增长问题
                    print_rank_0(f'teacher_{i}-max_position_embeddings was modified to {original_v}')
                    v = original_v
            original_v = '' if original_v is None else original_v
            setattr(args, name, type(original_v)(v))
        teacher_model = get_model(args, **kwargs)  # without deepspeed.initialize
        # 加载参数
        if f'student.teacher_model_{i}' in sd:
            sd_ = {'module': sd[f'student.teacher_model_{i}']}
            print_rank_0(f'mt_load_from_s: student.teacher_model_{i}')
        else:
            sd_ = None
        load_pretrained(teacher_model, args.load_pretrained, args, sd=sd_)
        if not args.mt_has_grad:
            teacher_model.eval()
        teacher_models.append(teacher_model)
    # 复原
    for v, name in zip(original_vars, transfer_vars):
        setattr(args, name, v)
    args.fp16 = fp16
    return teacher_models


def get_teachers_hook(args, student_model=None):
    # 学生模型针对多个教师模型生成hook
    transfer_vars = [
        'num_attention_heads',
        'hidden_size',
        'num_layers',
        'max_position_embeddings',
        'load_pretrained',
    ]
    check = [len(getattr(args, 'mt_' + i).split(':')) - 1 for i in transfer_vars]
    assert check[0] * len(transfer_vars) == sum(check), 'args中的多教师参数不是一一对应!'
    if student_model is None:  # only check
        return None
    if check[0] == 0:
        return [student_model.get_teacher_hook()]
    original_vars = [getattr(args, 'teacher_' + i) for i in transfer_vars]
    # 替换
    hooks = []
    for i, vars in enumerate(zip(*[getattr(args, 'mt_' + i).split(':') for i in transfer_vars])):
        for name, v, original_v in zip(transfer_vars, vars, original_vars):
            original_v = '' if original_v is None else original_v
            setattr(args, 'teacher_' + name, type(original_v)(v))
        hooks.append(student_model.get_teacher_hook(t_no=i))
    # 复原
    for v, name in zip(original_vars, transfer_vars):
        setattr(args, 'teacher_' + name, v)
    # 再处理
    if hasattr(student_model, 'multi_teacher_model'):
        hooks = student_model.multi_teacher_model.hooks_process(hooks)
    return hooks


def mt_repeat_operation(input_L, operate_f, output_f):
    # 对多个教师模型的重复操作
    out_L = []
    for i in input_L:
        if isinstance(i, (list, tuple)):
            out_L.append(output_f(operate_f(*i)))
        else:
            out_L.append(output_f(operate_f(**i)))
    return out_L


def mt_model_load(model, checkpoint_path):
    # 尝试额外加载多教师模型/学生模型中 multi_teacher_model 的模型参数
    student_model = unpacking_student_model(model)
    if not checkpoint_path or student_model is None:
        return False
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    sd = torch.load(checkpoint_name, map_location='cpu')
    mt_model = student_model.multi_teacher_model
    if 'student.multi_teacher_model' in sd['module']:
        module = sd['module']['student.multi_teacher_model']
    else:
        module = sd['module']
    missing_keys, unexpected_keys = mt_model.load_state_dict(module, strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"multi_teacher_model: Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
        time.sleep(3)
    return True


def truncate_teacher_as_student(model, teacher_models, args):
    # 提取教师模型的部分glm参数作为学生模型的初始化
    if args.student_truncate_tn is None or len(teacher_models) <= args.student_truncate_tn:
        return False
    s_model = unpacking_student_model(model).origin_model
    t_model = teacher_models[args.student_truncate_tn]
    s_sd = s_model.state_dict()
    s_sd_new = {}  # {'状态名称':张量,..}
    print_rank_0(f'从教师模型 {args.student_truncate_tn} 中截断出学生模型参数 ...')
    for k, v in t_model.state_dict().items():
        if k not in s_sd:
            continue
        if s_sd[k].size() != v.size():
            print_rank_0(f'trim {k}: {v.size()} -> {s_sd[k].size()}')
            min_size = [slice(0, min(i)) for i in zip(s_sd[k].size(), v.size())]
            s_sd_new[k] = s_sd[k].clone()
            s_sd_new[k][min_size] = v[min_size].clone()
        else:
            s_sd_new[k] = v.clone()
    missing_keys, unexpected_keys = s_model.load_state_dict(s_sd_new, strict=False)
    if missing_keys or unexpected_keys:
        print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
        time.sleep(3)
    return True


class NoneWith:
    def __enter__(*x): ...
    def __exit__(*x): ...
