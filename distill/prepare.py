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
    py_parser.add_argument('--student_model', type=str, default=None)
    py_parser.add_argument('--student_truncate', type=int, default=None, help='如果有的话选择第n个教师前几层截断作为初始化')
    py_parser.add_argument('--distill_ft_soft', action='store_true')
    py_parser.add_argument('--distill_ft_hard', action='store_true')
    py_parser.add_argument('--distill_pt_soft', action='store_true')
    py_parser.add_argument('--distill_pt_hard', action='store_true')
    py_parser.add_argument('--distill_temperature', type=float, default=1.)
    py_parser.add_argument('--distill_wo_loss_mask', action='store_true')
    py_parser.add_argument('--distill_only_mask_pad', action='store_true')
    py_parser.add_argument('--distill_ft_soft_kl', action='store_true', help="使用kl散度计算ft_soft")
    py_parser.add_argument('--distill_pt_soft_ce', action='store_true', help="使用交叉熵计算pt_soft")
    # teacher
    py_parser.add_argument('--teacher_num_attention_heads', type=int, default=16)
    py_parser.add_argument('--teacher_hidden_size', type=int, default=1024)
    py_parser.add_argument('--teacher_num_layers', type=int, default=24)
    py_parser.add_argument('--teacher_max_position_embeddings', type=int, default=512)
    py_parser.add_argument('--teacher_load_pretrained', type=str, default=None)
    py_parser.add_argument('--teacher_fp16', action='store_true')

    # tinybert
    py_parser.add_argument('--tinybert_inter_final', action='store_true', help="inter: final layer")
    py_parser.add_argument('--tinybert_wo_inter', action='store_true', help="不使用中间层,用于二次微调")
    py_parser.add_argument('--tinybert_fit_parallel', action='store_true')
    # minilmv2
    py_parser.add_argument('--minilmv2_relation_heads', type=int, default=48, help="base=48,large=64")
    py_parser.add_argument('--minilmv2_teacher_layer', type=int, default=12, help="start at one")
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
    py_parser.add_argument('--mixbaseline_inter_bl', type=str, default='TinyBERT,MiniLMv2,MiniLM,DistilBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_pt_soft', type=str, default='DistilBERT,TinyBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_ft_soft', type=str, default='TinyBERT')
    # pkd
    py_parser.add_argument('--pkd_normalized_patience', action='store_true')
    py_parser.add_argument('--pkd_alpha', type=float, default=0.5, help="soft权重")
    py_parser.add_argument('--pkd_beta', type=float, default=100., help="中间层权重")

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

    known, args_list = py_parser.parse_known_args()
    args = get_args_(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # check args
    get_teachers_hook(args)
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
                if original_v > int(v):
                    print_rank_0(f'teacher_{i}-max_position_embeddings was modified to {original_v}')
                    v = original_v
            original_v = '' if original_v is None else original_v
            setattr(args, name, type(original_v)(v))
        teacher_model = get_model(args, **kwargs)  # without deepspeed.initialize
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
    for vars in zip(*[getattr(args, 'mt_' + i).split(':') for i in transfer_vars]):
        for name, v, original_v in zip(transfer_vars, vars, original_vars):
            original_v = '' if original_v is None else original_v
            setattr(args, 'teacher_' + name, type(original_v)(v))
        hooks.append(student_model.get_teacher_hook())
    # 复原
    for v, name in zip(original_vars, transfer_vars):
        setattr(args, 'teacher_' + name, v)
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
    student_model = unpacking_student_model(model)


class NoneWith:
    def __enter__(*x): ...
    def __exit__(*x): ...
