import sys, os
sys.path.append(os.getcwd())

from arguments import get_args as get_args_
import argparse
from train_utils import get_model


def get_args():
    py_parser = argparse.ArgumentParser(add_help=False)
    # generic
    py_parser.add_argument('--student_model', type=str, default=None)
    py_parser.add_argument('--distill_ft_soft', action='store_true')
    py_parser.add_argument('--distill_ft_hard', action='store_true')
    py_parser.add_argument('--distill_pt_soft', action='store_true')
    py_parser.add_argument('--distill_pt_hard', action='store_true')
    py_parser.add_argument('--distill_temperature', type=float, default=1.)
    # teacher
    py_parser.add_argument('--teacher_num_attention_heads', type=int, default=16)
    py_parser.add_argument('--teacher_hidden_size', type=int, default=1024)
    py_parser.add_argument('--teacher_num_layers', type=int, default=24)
    py_parser.add_argument('--teacher_max_position_embeddings', type=int, default=512)
    py_parser.add_argument('--teacher_load_pretrained', type=str, default=None)
    py_parser.add_argument('--teacher_fp16', action='store_true')
    # tinybert
    py_parser.add_argument('--tinybert_ft_pre', action='store_true', help="微调2次蒸馏用预测层")
    py_parser.add_argument('--tinybert_ft_hard', action='store_true', help="微调2次蒸馏用硬标签")
    py_parser.add_argument('--tinybert_pt_hard', action='store_true')
    py_parser.add_argument('--tinybert_inter_final', action='store_true', help="inter: final layer")
    # minilmv2
    py_parser.add_argument('--minilmv2_relation_heads', type=int, default=48, help="base=48,large=64")
    py_parser.add_argument('--minilmv2_teacher_layer', type=int, default=12, help="start at one")
    # distilbert
    py_parser.add_argument('--distilbert_alpha_ce', type=float, default=0.33)
    py_parser.add_argument('--distilbert_alpha_mlm', type=float, default=0.33)
    py_parser.add_argument('--distilbert_alpha_cos', type=float, default=0.33)
    py_parser.add_argument('--distilbert_temperature', type=float, default=10.)

    known, args_list = py_parser.parse_known_args()
    args = get_args_(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    if args.student_model == 'tinybert':
        args.distill_ft_soft = args.tinybert_ft_pre
        args.distill_ft_hard = args.tinybert_ft_hard
        args.distill_pt_hard = args.tinybert_pt_hard
    if args.teacher_num_layers < args.minilmv2_teacher_layer:
        args.minilmv2_teacher_layer = args.teacher_num_layers
    return args
    

def get_teacher_model(args, **kwargs):
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    max_position_embeddings = args.max_position_embeddings
    load_pretrained = args.load_pretrained
    fp16 = args.fp16

    args.num_layers = args.teacher_num_layers
    args.hidden_size = args.teacher_hidden_size
    args.num_attention_heads = args.teacher_num_attention_heads
    args.max_position_embeddings = args.teacher_max_position_embeddings
    args.load_pretrained = args.teacher_load_pretrained
    args.fp16 = args.teacher_fp16
    teacher_model = get_model(args, **kwargs)  # without deepspeed.initialize

    args.num_layers = num_layers
    args.hidden_size = hidden_size
    args.num_attention_heads = num_attention_heads
    args.max_position_embeddings = max_position_embeddings
    args.load_pretrained = load_pretrained
    args.fp16 = fp16
    return teacher_model