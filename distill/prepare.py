import sys, os
sys.path.append(os.getcwd())

from arguments import get_args as get_args_
import argparse
from train_utils import get_model
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration, find_model_inter_var
from train_utils import load_pretrained
from distill.distill_model import student_model_D, unpacking_student_model
from distill.multi_teacher_model import multi_teacher_model_D
import torch
import time
import mpu
import copy
import torch.nn.functional as F


def get_args():
    py_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # generic
    py_parser.add_argument('--student_model', type=str, default=None, help='学生模型类型,没有则是原生模型')
    py_parser.add_argument('--student_truncate_tn', type=int, default=None, help='非None或不小于0的话代表选择第n个教师前面层截断作为初始化(长于学生的维度靠前截断参数,短于学生的维度默认学生参数不变)')
    py_parser.add_argument('--student_build_map_vocab', action='store_true', help='是否重新构建映射词表.通常只有预训练(且不加载已有学生参数)时需要构建初始映射')
    py_parser.add_argument('--student_map_vocab_tn', type=int, default=None, help='非None或不小于0的话代表选择第n个教师的word_embeddings计算省去token的最大相似token作为替代,否则使用unk作为替代')
    py_parser.add_argument('--student_map_vocab_method', type=str, default='decoder', help='cosine/euclidean/decoder,指定student_map_vocab_tn的情况下使用什么相似度计算')
    py_parser.add_argument('--enable_parallel_entropy', action='store_true', help='是否使用并行的loss熵计算,可能会减少显存占用')
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
    py_parser.add_argument('--distill_logits_parallel', action='store_true', help='是否将logits_parallel当作inter_loss使用,只有在NLU的ft阶段有价值,其他重复时可能产生soft权重*2的效果,注意一般不受wo_inter类参数的约束')
    py_parser.add_argument('--distill_logit_mask_pad', action='store_true', help='--distill_logits_parallel 参数下是否mask padding')
    py_parser.add_argument('--distill_logit_mask_map', action='store_true', help='使用logits_parallel计算并且学生有map_vocab时,是否忽略映射token的蒸馏相似度计算')
    py_parser.add_argument('--distill_logit_mse', action='store_true', help='是否用MSE计算--distill_logits_parallel')
    # 分析
    py_parser.add_argument('--distill_test_output', action='store_true', help='是否测试输出,会运行测试代码,针对有测试的方法,如theseus')
    py_parser.add_argument('--distil_analysis_inter', action='store_true', help="是否从所有角度分析中间层教师与学生的相关相似度,保存于tensorboard,会增加时空消耗.受到AttHead和HSDim限制的不会统计.小心增加新的蒸馏方法,否则使用这个这可能导致一些只依赖hook传入中间层进行蒸馏loss计算的方法计算了过多的中间层")
    py_parser.add_argument('--distil_analysis_frcn', type=str, default='0', help="针对distil_analysis_inter,分析哪一次forward_repeat_current_n的中间层,多个用半角逗号隔开")
    # 引入随机数据
    py_parser.add_argument('--distill_random_data', type=str, default='', help='dual:数据batch size变成原来一倍,随机数据加载后面;replace:直接替换数据成随机数据;空则不使用随机数据,评估时也不生效')
    py_parser.add_argument('--distill_random_data_n', type=str, default='0', help='针对args.forward_repeat_num的第几次重复引入随机数据,例如1或者0,1')
    py_parser.add_argument('--distill_random_data_method', type=str, default='shuffle', help='shuffle:将要随机的token随机改变顺序;sample:将已有token随机替换为其他任何一个token')
    # py_parser.add_argument('--distill_random_data_rate', type=float, default=1., help='随机的比例')
    # py_parser.add_argument('--distill_random_data_only_a', action='store_true', help='是否只随机Part A,不让Part B和Padding部分也随机')
    # teacher
    py_parser.add_argument('--teacher_num_attention_heads', type=int, default=16)
    py_parser.add_argument('--teacher_hidden_size', type=int, default=1024)
    py_parser.add_argument('--teacher_num_layers', type=int, default=24)
    py_parser.add_argument('--teacher_max_position_embeddings', type=int, default=512)
    py_parser.add_argument('--teacher_load_pretrained', type=str, default=None)
    py_parser.add_argument('--teacher_fp16', action='store_true')
    py_parser.add_argument('--teacher_inverted_bottleneck_mode', action='store_true')
    py_parser.add_argument('--teacher_ib_hidden_size', type=int, default=1024)
    py_parser.add_argument('--teacher_ib_ffn_num', type=int, default=1)
    py_parser.add_argument('--teacher_ib_word_emb', type=int, default=0)
    py_parser.add_argument('--teacher_compress_word_emb', type=int, default=0)
    py_parser.add_argument('--teacher_map_vocab_size', type=float, default=0)
    py_parser.add_argument('--teacher_cross_layer_parameter_sharing', action='store_true')

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
    py_parser.add_argument('--tinybert_fit_compatible_mt', action='store_true', help='是否使用多个转换层兼容多教师,多个HS维度不同的教师必备')
    py_parser.add_argument('--tinybert_random_layers', action='store_true', help="是否随机选择中间层(emb和final不动)")
    py_parser.add_argument('--tinybert_random_e', type=int, default=1, help="每几轮训练后随机选择层,大于0有效,优先")
    py_parser.add_argument('--tinybert_random_i', type=int, default=3000, help="每几次迭代后随机选择层,大于0有效,tinybert_random_i为0这个参数才有效")
    py_parser.add_argument('--tinybert_random_show', action='store_true', help="显示每次随机后的教师中间取层(不含emb/final)")
    # minilmv2
    py_parser.add_argument('--minilmv2_relation_heads', type=int, default=48, help="base=48,large=64")
    py_parser.add_argument('--minilmv2_teacher_layer', type=int, default=12, help="start at one,-1就代表倒数第一层")
    py_parser.add_argument('--minilmv2_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")
    py_parser.add_argument('--minilmv2_relation_heads_mt', type=str, default=None, help="针对多教师的配置,有值将忽略minilmv2_relation_heads,例如设置为48,64")
    # distilbert
    py_parser.add_argument('--distilbert_alpha_ce', type=float, default=1., help="类似 distill_pt_soft")
    py_parser.add_argument('--distilbert_alpha_mlm', type=float, default=1., help="类似 distill_pt_hard")
    py_parser.add_argument('--distilbert_alpha_cos', type=float, default=1., help='最后输出层的权重')
    py_parser.add_argument('--distilbert_fix_layernorm', action='store_true')
    py_parser.add_argument('--distilbert_cos_mask_padding', action='store_true', help='隐层只mask padding')
    py_parser.add_argument('--distilbert_ce_mask_padding', action='store_true', help='软标签只mask padding')
    # mixbaseline
    py_parser.add_argument('--mixbaseline_wo_inter', action='store_true', help="不使用中间层,用于二次微调")
    py_parser.add_argument('--mixbaseline_tinybert_t', type=float, default=1., help="专用的temperature")
    py_parser.add_argument('--mixbaseline_inter_bl', type=str, default='', help='TinyBERT,MiniLMv2,MiniLM,DistilBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_pt_soft', type=str, default='', help='DistilBERT,TinyBERT')
    py_parser.add_argument('--mixbaseline_pre_bl_ft_soft', type=str, default='', help='TinyBERT')
    py_parser.add_argument('--mixbaseline_inter_checkpoint', action='store_true', help='可能改变运算,受全局参数/随机因素/二次执行问题的影响导致这个参数意义不大')
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
    # diito
    py_parser.add_argument('--diito_alignment', type=str, default='full', help='full,middle,late')
    py_parser.add_argument('--diito_interchange_prop', type=float, default=0.3, help='')
    py_parser.add_argument('--diito_interchange_way', type=str, default='consecutive', help='consecutive,masked,random')
    py_parser.add_argument('--diito_interchange_max_token', type=int, default=-1, help='-1表示不限制交换长度')
    py_parser.add_argument('--diito_alpha_mlm', type=float, default=1., help="类似 distill_pt_hard")
    py_parser.add_argument('--diito_alpha_ce', type=float, default=1., help="类似 distill_pt_soft")
    py_parser.add_argument('--diito_alpha_causal_ce', type=float, default=1.)
    py_parser.add_argument('--diito_alpha_cos', type=float, default=1.)
    py_parser.add_argument('--diito_alpha_causal_cos', type=float, default=1.)
    # logitsdistil
    py_parser.add_argument('--logitsdistil_mask_pad', action='store_true', help='logits_parallel是否mask padding')
    py_parser.add_argument('--logitsdistil_mse', action='store_true', help='logits_parallel是否用MSE计算')
    py_parser.add_argument('--logitsdistil_top_n', type=float, default=None, help="大于0生效,0-1之间则当比例获取topn")
    py_parser.add_argument('--logitsdistil_teacher_min', action='store_true', help="将教师logits中top_n之后的值都置为最小值,而不是对学生logits进行约束,避免长尾难以压制的问题")
    py_parser.add_argument('--logitsdistil_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")
    py_parser.add_argument('--logitsdistil_teacher_input_ids_map', action='store_true', help="在学生map_vocab状态下,是否也对教师输入映射token,使得学生和教师输入一样")
    py_parser.add_argument('--logitsdistil_mask_a', action='store_true', help="是否mask part A部分")
    # sid
    py_parser.add_argument('--sid_accumulate_t', type=float, default=0., help='cosine loss threshold')
    py_parser.add_argument('--sid_lim_e', type=str, default='avg', help='limited number of epochs per layer')
    # alp_kd
    py_parser.add_argument('--alp_kd_lambda', type=float, default=0.2, help='ALP损失权重')
    # ckd
    py_parser.add_argument('--ckd_window_size', type=int, default=21, help='三角关系的窗口大小,只计算这周围的几个token')
    py_parser.add_argument('--ckd_wrdist_w', type=float, default=1, help='同层token成对关系权重')
    py_parser.add_argument('--ckd_ltrdist_w', type=float, default=1, help='不同层同位置token成对关系权重')
    py_parser.add_argument('--ckd_wrangle_w', type=float, default=10, help='同层token三角关系权重')
    py_parser.add_argument('--ckd_ltrangle_w', type=float, default=10, help='不同层同位置token三角关系权重')
    # theseus
    py_parser.add_argument('--theseus_replacing_rate', type=float, default=0.3, help='初始替换率')
    py_parser.add_argument('--theseus_not_replaced_steps', type=float, default=0.66, help='跑到总迭代次数的多少比例时全部替换为学生层,即替换率为1')
    # universal_kd
    py_parser.add_argument('--universal_kd_wo_inter', action='store_true', help="不使用中间层,可用于二次微调")
    py_parser.add_argument('--universal_kd_cg', action='store_true', help="使用CG而不是IL/CA的损失，等于只用最后一层")
    py_parser.add_argument('--universal_kd_avg', action='store_true', help="使用平均池化而不是[CLS]计算中间层")
    py_parser.add_argument('--universal_kd_gamma', type=float, default=0.5, help='内层权重')
    py_parser.add_argument('--universal_kd_size', type=int, default=0, help='中间层变换后的维度，0表示使用学生维度')
    # lrc_bert
    py_parser.add_argument('--lrc_bert_alpha', type=float, default=1., help='内层权重')
    py_parser.add_argument('--lrc_bert_gard_perturb', action='store_true', help='Training based on Gradient Perturbation')
    py_parser.add_argument('--lrc_bert_gather_dp', action='store_true', help='计算对比损失且使用数据并行的时候将中间层输出合并,会增加通讯量和空间,但可以在数据并行的时候维持负例数量等于总batch size-1')
    # annealing_kd
    py_parser.add_argument('--annealing_kd_max_t', type=float, default=7., help='最大温度')
    # mobilebert
    py_parser.add_argument('--mobilebert_kd_w', type=float, default=0.5, help='中间层损失权重')
    py_parser.add_argument('--mobilebert_pkt_small_lr', type=float, default=0.1, help='pkt中间stage的较低学习率')

    # multi-teacher 多个教师的模型参数用冒号分隔, 优先级高于 teacher_ 参数
    py_parser.add_argument('--mt_num_attention_heads', type=str, default='')
    py_parser.add_argument('--mt_hidden_size', type=str, default='')
    py_parser.add_argument('--mt_num_layers', type=str, default='')
    py_parser.add_argument('--mt_max_position_embeddings', type=str, default='')
    py_parser.add_argument('--mt_load_pretrained', type=str, default='')
    py_parser.add_argument('--mt_disable_operation', type=str, default='0', help='是否不计算教师模型的输出结果(用替代值),可用于Theseus等只需要教师中间层的方法.0表示计算,1表示不计算,冒号分隔则针对每个教师分别处理.如果使用distil_analysis_inter则不能使用这个')
    py_parser.add_argument('--mt_ib_hidden_size', type=str, default='')
    py_parser.add_argument('--mt_ib_ffn_num', type=str, default='')
    py_parser.add_argument('--mt_ib_word_emb', type=str, default='')
    py_parser.add_argument('--mt_compress_word_emb', type=str, default='')
    py_parser.add_argument('--mt_map_vocab_size', type=str, default='')
    # multi-teacher model (指将多个教师联合在一起的模型)
    py_parser.add_argument('--multi_teacher_model', type=str, default=None, help='多教师模型名称')
    py_parser.add_argument('--mt_model_load', type=str, default=None, help='可选额外加载的多教师模型路径,可以自动从其他学生模型路径中提取')
    py_parser.add_argument('--mt_has_loss', action='store_true', help='是否每个教师都需要计算最终loss,配合某些多教师模型')
    py_parser.add_argument('--mt_has_grad', action='store_true', help='是否每个教师都需要梯度,是的话教师模型会作为学生模型的一部分进行更新')
    py_parser.add_argument('--student_use_empty_glm', action='store_true', help='学生模型中的glm模型置空,可配合mt_has_grad训练活的多教师')
    py_parser.add_argument('--mt_load_from_s', type=str, default=None, help='从整合多教师模型的学生模型路径中加载多教师的参数,将替代teacher_/mt_load_pretrained,mt_*参数中多教师顺序与当初保存的要一致')
    # default
    py_parser.add_argument('--avgmt_inter_checkpoint', action='store_true')
    py_parser.add_argument('--avgmt_pre_checkpoint', action='store_true')
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
    if not (args.teacher_load_pretrained or args.mt_num_attention_heads) or not args.student_model:
        return None
    transfer_vars = [
        'num_attention_heads',
        'hidden_size',
        'num_layers',
        'max_position_embeddings',
        'load_pretrained',
        'ib_hidden_size',
        'ib_ffn_num',
        'ib_word_emb',
        'compress_word_emb',
        'map_vocab_size',
    ]
    original_vars = [getattr(args, i) for i in transfer_vars]
    # 统一参数加载
    if args.mt_load_from_s:
        load_dir, tag, release, success = get_checkpoint_iteration(args.mt_load_from_s)
        checkpoint_name = get_checkpoint_name(load_dir, tag, release)
        sd = torch.load(checkpoint_name, map_location='cpu')['module']
    else:
        sd = {}
    # 非mt参数替换
    fp16, args.fp16 = args.fp16, args.teacher_fp16
    unmap_vocab_output, args.unmap_vocab_output = args.unmap_vocab_output, False  # 教师不兼容该选项
    inverted_bottleneck_mode, args.inverted_bottleneck_mode = args.inverted_bottleneck_mode, args.teacher_inverted_bottleneck_mode
    cross_layer_parameter_sharing, args.cross_layer_parameter_sharing = args.cross_layer_parameter_sharing, args.teacher_cross_layer_parameter_sharing
    # mt参数替换
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
    args.unmap_vocab_output = unmap_vocab_output
    args.inverted_bottleneck_mode = inverted_bottleneck_mode
    args.cross_layer_parameter_sharing = cross_layer_parameter_sharing
    return teacher_models


def get_teachers_hook(args, student_model=None, is_op=False, **kwargs):
    # 学生模型针对多个教师模型生成hook
    transfer_vars = [
        'num_attention_heads',
        'hidden_size',
        'num_layers',
        'max_position_embeddings',
        'load_pretrained',
        'ib_hidden_size',
        'ib_ffn_num',
        'ib_word_emb',
        'compress_word_emb',
        'map_vocab_size',
    ]
    check = [len(getattr(args, 'mt_' + i).split(':')) - 1 for i in transfer_vars]
    assert check[0] * len(transfer_vars) == sum(check), 'args中的多教师参数不是一一对应!'
    if student_model is None:  # only check
        return None
    get_teacher_hook = student_model.get_teacher_hook_op if is_op else student_model.get_teacher_hook
    if check[0] == 0:
        return [get_teacher_hook(**kwargs)]
    original_vars = [getattr(args, 'teacher_' + i) for i in transfer_vars]
    # 替换
    hooks = []
    for i, vars in enumerate(zip(*[getattr(args, 'mt_' + i).split(':') for i in transfer_vars])):
        for name, v, original_v in zip(transfer_vars, vars, original_vars):
            original_v = '' if original_v is None else original_v
            setattr(args, 'teacher_' + name, type(original_v)(v))
        hooks.append(get_teacher_hook(t_no=i, **kwargs))
    # 复原
    for v, name in zip(original_vars, transfer_vars):
        setattr(args, 'teacher_' + name, v)
    # 再处理
    if hasattr(student_model, 'multi_teacher_model'):
        hooks = student_model.multi_teacher_model.hooks_process(hooks)
    return hooks


def mt_repeat_operation(input_L, operate_f, output_f, is_disable_L=None, disable_ret_L=None):
    """对多个教师模型的重复操作

    Args:
        input_L (list): 每次处理的输入,包括教师模型和参数等等
        operate_f (func): 针对输出进行的操作,每组输入操作一样
        output_f (func): 输出处理之后的再提取,规范到固定的输出格式
        is_disable_L (list, optional): 是否不对 input 进行处理
        disable_ret_L (list, optional): 不对 input 进行处理后的替代值

    Returns:
        [{'logits':,..},..]: 每个教师的返回结果
    """
    # 初始化 disable
    input_L = list(input_L)
    if is_disable_L is None:
        is_disable_L = [False] * len(input_L)
    if len(is_disable_L) == 1:
        is_disable_L *= len(input_L)
    assert len(is_disable_L) == len(input_L)
    if disable_ret_L is None:
        disable_ret_L = [{} for _ in len(input_L)]
    if len(disable_ret_L) == 1:
        disable_ret_L += [copy.deepcopy(disable_ret_L[0]) for _ in range(len(input_L) - 1)]
    assert len(disable_ret_L) == len(input_L)
    # 开始操作
    out_L = []
    for i, is_disable, disable_ret in zip(input_L, is_disable_L, disable_ret_L):
        if is_disable:
            out_L.append(disable_ret)
        else:
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
    print_rank_0(f'> 从教师模型 {args.student_truncate_tn} 中截断出学生模型参数 ...')
    t_state_dict = t_model.state_dict()
    # map_vocab
    t_model_glm = unpacking_student_model(t_model, attrs=('map_input_to_ids', 'get_word_embeddings_weight'))
    if t_model_glm.map_vocab_size:
        print_rank_0(f'get word_embeddings.weight before mapping from teacher')
        weight = find_model_inter_var(t_model, 'get_word_embeddings_weight')(
            parallel_output=True, before_map_vocab=True)
        t_state_dict['word_embeddings.weight'] = weight
    if s_model.map_vocab_size:
        print_rank_0(f'from word_embeddings.weight before mapping to after mapping')
        unmasked_origin_id = s_model.map_vocab_paras['target_pos_to_origin_id'][:s_model.map_vocab_size]
        if mpu.get_model_parallel_world_size() == 1:
            weight = t_state_dict['word_embeddings.weight']
            t_state_dict['word_embeddings.weight'] = F.embedding(unmasked_origin_id, weight)
        else:
            weight = mpu.gather_from_model_parallel_region(t_state_dict['word_embeddings.weight'].T)
            weight = weight[..., unmasked_origin_id]
            t_state_dict['word_embeddings.weight'] = mpu.scatter_to_model_parallel_region(weight).T
    # load
    s_sd_new = {}  # {'状态名称':张量,..}
    for k, v in t_state_dict.items():
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

def build_map_vocab_for_student(model, teacher_models, args, tokenizer):
    # 初始化学生模型的词表映射参数, args 需要先经过 prepare_tokenizer 处理
    path = f'.pytorch_pretrained_bert/{args.tokenizer_type}_{args.tokenizer_model_type}.txt'
    if not args.map_vocab_size or not os.path.exists(path) or not args.student_build_map_vocab:
        return False
    # 获取排序的映射关系
    origin_no_L = []  # 顺序就是教师词表的id,里面的值表示学生word_embeddings的序号
    sort_id_L = []  # word_embeddings的序号对应的id
    with open(path, 'r', encoding='utf8') as f:
        head_no = {h: i for i, h in enumerate(f.readline().strip().split('\t'))}
        for line in f:
            line = line.strip().split('\t')
            origin_no_L.append(int(line[head_no['origin_no']]))
            sort_id_L.append(int(line[head_no['sort_id']]))
            assert line[head_no['token']] == tokenizer.IdToToken(sort_id_L[-1]), 'tokenizer 与 map_vocab 不一致!'
    for i in range(len(sort_id_L)):
        assert sort_id_L[origin_no_L[i]] == i, 'origin_no 和 sort_id 不对应! {}'.format(i)
    assert len(set(origin_no_L)) == len(origin_no_L), 'origin_no_L 出现重复错误!'
    for i in range(len(sort_id_L), args.vocab_size):  # 补充因整数延长的id
        sort_id_L.append(i)
        origin_no_L.append(i)
    # map_origin
    map_origin_pos_to_target_id = torch.tensor(sort_id_L, device=torch.cuda.current_device())
    map_origin_id_to_target_pos = torch.tensor(origin_no_L, device=torch.cuda.current_device())
    # other 替换
    other_id = map_origin_pos_to_target_id[args.map_vocab_size:].clone()
    if args.student_map_vocab_tn is None or len(teacher_models) <= args.student_map_vocab_tn:
        map_origin_id_to_target_pos[other_id] = origin_no_L[tokenizer.get_command('unk').Id]
        other_map_id = [tokenizer.get_command('unk').Id] * (args.vocab_size - args.map_vocab_size)
        other_map_id = torch.tensor(other_map_id, device=torch.cuda.current_device())
    else:
        t_model = teacher_models[args.student_map_vocab_tn]
        t_word_embeddings = find_model_inter_var(t_model, 'word_embeddings')
        save_id = map_origin_pos_to_target_id[:args.map_vocab_size].clone()
        start_index, end_index = mpu.VocabUtility.vocab_range_from_global_vocab_size(
            args.vocab_size - args.map_vocab_size,
            mpu.get_model_parallel_rank(), mpu.get_model_parallel_world_size())
        other_embeddings = t_word_embeddings(other_id[start_index: end_index])
        save_embeddings = t_word_embeddings(save_id)
        if args.student_map_vocab_method == 'decoder':
            other_save = torch.matmul(other_embeddings, save_embeddings.T)
        else:
            a2 = (other_embeddings ** 2).sum(-1, keepdim=True)
            b2 = (save_embeddings ** 2).sum(-1, keepdim=True)
            ab = torch.matmul(other_embeddings, save_embeddings.T)
            if args.student_map_vocab_method == 'euclidean':
                # - (Euclidean distance) ** 2
                other_save = - (a2 + b2.T - 2 * ab)
            elif args.student_map_vocab_method == 'cosine':
                # Cosine similarity
                other_save = ab / (a2 * b2.T) ** 0.5
            else:
                raise Exception("Invalid --student_map_vocab_method={}".format(args.student_map_vocab_method))
        other_save = other_save.argmax(dim=-1)
        other_save = mpu.gather_from_model_parallel_region(other_save)
        other_map_id = torch.gather(save_id, dim=0, index=other_save)
        map_origin_id_to_target_pos[other_id] = map_origin_id_to_target_pos[other_map_id]
    origin_id_mask_map = torch.ones(args.vocab_size, dtype=torch.bool, device=torch.cuda.current_device())
    origin_id_mask_map[other_id] = False
    map_origin_pos_to_target_id[args.map_vocab_size:] = other_map_id
    # show
    show_top = 20
    print_rank_0('> 展示map_vocab中后半截[{},{})前{}个被替换的id和token:'.format(
        args.map_vocab_size, args.vocab_size, show_top))
    print_rank_0('替换前的id token\t替换后的id token')
    for old_id, new_id in zip(other_id, other_map_id[:show_top]):
        old_id, new_id = int(old_id), int(new_id)
        print_rank_0('{} {}\t{} {}'.format(
            old_id, tokenizer.IdToToken(old_id), new_id, tokenizer.IdToToken(new_id)))
    print_rank_0('> {}种token被替换为{}种'.format(len(other_map_id), len(set(int(i) for i in other_map_id))))
    # load
    s_model = unpacking_student_model(model).origin_model
    s_model.map_vocab_paras['origin_id_to_target_pos'] = map_origin_id_to_target_pos
    s_model.map_vocab_paras['origin_id_mask_map'] = origin_id_mask_map
    s_model.map_vocab_paras['target_pos_to_origin_id'] = map_origin_pos_to_target_id
    return True

class NoneWith:
    def __enter__(*x): ...
    def __exit__(*x): ...
