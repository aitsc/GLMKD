import sys, os
sys.path.append(os.getcwd())

from arguments import get_args as get_args_
import argparse
from train_utils import get_model
from utils import print_rank_0


def get_args():
    py_parser = argparse.ArgumentParser(add_help=False)
    # generic
    py_parser.add_argument('--student_model', type=str, default=None)
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
    py_parser.add_argument('--mixbaseline_inter_bl', type=str, default='', help='TinyBERT,MiniLMv2,MiniLM,DistilBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_pt_soft', type=str, default='', help='DistilBERT,TinyBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_ft_soft', type=str, default='', help='TinyBERT')
    # pkd
    py_parser.add_argument('--pkd_normalized_patience', action='store_true')
    py_parser.add_argument('--pkd_alpha', type=float, default=0.5, help="soft权重")
    py_parser.add_argument('--pkd_beta', type=float, default=100., help="中间层权重")

    known, args_list = py_parser.parse_known_args()
    args = get_args_(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    return args
    

def get_teacher_model(args, **kwargs):
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    num_attention_heads = args.num_attention_heads
    max_position_embeddings = args.max_position_embeddings
    load_pretrained = args.load_pretrained
    fp16 = args.fp16
    if max_position_embeddings > args.teacher_max_position_embeddings:
        args.teacher_max_position_embeddings = max_position_embeddings
        print_rank_0('teacher_max_position_embeddings was modified to %d'%max_position_embeddings)

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
