# Single teacher distillation
- pretrain ... : NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --master_port=13761 --include=localhost:0,7 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=384 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/tiny6 --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank --teacher_num_layers=12 --teacher_hidden_size=768 --teacher_num_attention_heads=12 --teacher_max_position_embeddings=512 --teacher_fp16

## KD
- pretrain(unnecessary): ... --student_model=kd --distill_pt_soft --distill_pt_hard --distill_temperature=10
- finetune: ... --student_model=kd --distill_ft_soft --distill_ft_hard --distill_temperature=10
    - --distill_ft_soft_kl

## PD
1. pretrain: without teacher
2. finetune: ... --student_model=kd --distill_ft_soft --distill_temperature=1

## TinyBERT
1. pretrain-inter: ... --student_model=tinybert
2. finetune-inter: ... --student_model=tinybert
3. finetune-pre: ... --student_model=tinybert --distill_ft_soft --tinybert_wo_inter
    - --distill_ft_soft_kl

## MiniLMv2
- pretrain: ... --student_model=minilmv2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12

## MiniLM
- pretrain: ... --student_model=minilm

## DistilBERT
- pretrain: ... --student_model=distilbert --distill_temperature=2 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding
    - --distilbert_ce_mask_padding

## ERDistill
1. pretrain: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter=all --distill_pt_hard
    - + --distill_pt_soft --erdistill_inter_mse --erdistill_wo_global --distill_temperature=10.
2. finetune: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter=all --distill_temperature=10. --distill_ft_soft --distill_ft_hard
    - + --erdistill_inter_mse --erdistill_wo_global
3. finetune: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter= --distill_temperature=10. --distill_ft_soft --distill_ft_hard
    - + --erdistill_inter_mse

## MixBaseline (pt/ft1: TinyBERT + MiniLMv2 + MiniLM + DistilBERT, ft2: KD + TinyBERT)
1. pretrain: ... --student_model=mixbaseline --distill_temperature=2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding --mixbaseline_inter_bl=TinyBERT,MiniLMv2,MiniLM,DistilBERT --mixbaseline_pre_bl_pt_soft=DistilBERT
2. finetune: ... --student_model=mixbaseline --distill_temperature=2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding --mixbaseline_inter_bl=TinyBERT,MiniLMv2,MiniLM,DistilBERT --mixbaseline_pre_bl_ft_soft=DistilBERT
3. finetune: ... --student_model=mixbaseline --mixbaseline_wo_inter --tinybert_wo_inter --distill_ft_soft --distill_ft_hard --distill_temperature=10 --mixbaseline_tinybert_t=1 --mixbaseline_pre_bl_ft_soft=TinyBERT
    - --distill_ft_soft_kl

## PKD
- finetune: ... --student_model=pkd --distill_ft_soft --distill_ft_soft_kl --distill_ft_hard --distill_temperature=10 --pkd_normalized_patience --pkd_alpha=0.5 --pkd_beta=100 --student_truncate_tn=0 --pkd_wo_final

## RAIL_KD
1. pretrain: DistilBERT
2. finetune: ... --student_model=rail_kd --distill_ft_soft --distill_soft_rate=0.3333 --distill_ft_hard --distill_hard_rate=0.3333 --distill_temperature=10 --rail_kd_inter_rate=0.3333 --rail_kd_layer_wise_alpha=1 --rail_kd_u=128 --rail_kd_concatenated --rail_kd_epochs=1 --rail_kd_show_hook_change

## MGSKD
1. pretrain: TinyBERT
2. finetune: ... --student_model=mgskd --mgskd_weight_sample=4 --mgskd_weight_token=1 --mgskd_weight_span=1 --mgskd_sample_level_m=3 --mgskd_triplet_k1=20 --mgskd_triplet_k2=20
3. finetune: ... --student_model=mgskd --distill_ft_soft --distill_ft_soft_kl --distill_temperature=1 --mgskd_wo_inter

## DIITO
- pretrain: ... --student_model=diito --forward_repeat_num=1 --diito_alignment=full --diito_interchange_prop=0.3 --diito_interchange_way=consecutive --diito_interchange_max_token=-1 --diito_alpha_mlm=0.25 --diito_alpha_ce=0.25 --diito_alpha_causal_ce=0.25 --diito_alpha_cos=0.25 --diito_alpha_causal_cos=0  --distill_pt_soft --distill_pt_hard --distill_temperature=2
    - --distill_only_mask_pad
    - --checkpoint-activations cannot be used when backpropagating interchanged_variable

## LogitsDistil
- old method
    1. pretrain: ... --student_model=kd --distill_pt_soft --distill_temperature=15 --distill_wo_loss_mask
        - --distill_only_mask_pad --distill_logits_parallel --distill_temperature=15
    2. finetune: ... --student_model=kd --distill_logits_parallel --distill_temperature=15
        - --distill_logit_mask_pad
    3. finetune: ... --student_model=kd --distill_ft_soft
- new method
    1. pretrain: ... --student_model=logitsdistil --distill_temperature=15
        - --logitsdistil_mask_pad --logitsdistil_mse --logitsdistil_top_n=20 --logitsdistil_teacher_min
    2. finetune: ... --student_model=logitsdistil --distill_temperature=15
        - --logitsdistil_mask_pad --logitsdistil_mse --logitsdistil_top_n=20 --logitsdistil_teacher_min
    3. finetune: ... --student_model=logitsdistil --distill_ft_soft --logitsdistil_wo_inter

## VocabDistil
1. pretrain: ... --student_model=logitsdistil --distill_temperature=15 --map_vocab_size=0.5 --distill_logit_mask_map --student_build_map_vocab --student_map_vocab_tn=0 --student_map_vocab_method=decoder
    - --logitsdistil_teacher_input_ids_map
2. finetune: ... --student_model=logitsdistil --distill_temperature=1 --map_vocab_size=0.5 --distill_logit_mask_map
    - --logitsdistil_teacher_input_ids_map
3. finetune: ... --student_model=logitsdistil --distill_ft_soft --logitsdistil_wo_inter --map_vocab_size=0.5

## SID
1. pretrain: without teacher
2. finetune: ... --student_model=sid --sid_accumulate_t=0 --sid_lim_e=avg --distill_ft_soft --distill_temperature=1

## ALP_KD
- finetune: ... --student_model=alp_kd --alp_kd_lambda=0.2 --distill_soft_rate=0.7 --distill_hard_rate=0.1 --distill_temperature=20 --student_truncate_tn=0 --distill_ft_soft --distill_ft_hard

## CKD
1. pretrain: without teacher
2. finetune: ... --student_model=ckd --ckd_window_size=21 --ckd_wrdist_w=1 --ckd_ltrdist_w=1 --ckd_wrangle_w=10 --ckd_ltrangle_w=10 --distill_ft_soft --distill_ft_hard --distill_temperature=3 --distill_soft_rate=0.9 --distill_hard_rate=0.1

## Theseus
- finetune: ... --student_model=theseus --distill_ft_hard --student_truncate_tn=0 --theseus_replacing_rate=0.3 --theseus_not_replaced_steps=0.66 --mt_disable_operation=1

## Universal_KD
1. finetune-1: ... --student_model=universal_kd --distill_ft_soft --distill_ft_soft_kl --distill_soft_rate=0.5 --universal_kd_gamma=0.5 --student_truncate_tn=0 --universal_kd_size=0
    - --universal_kd_cg --universal_kd_avg
2. finetune-2: without teacher

## LRC_BERT
1. pretrain: without teacher
2. finetune-1: ... --student_model=lrc_bert --lrc_bert_gard_perturb --ignore_first_backward_gard --forward_repeat_num=1
    - It is not recommended to use data parallelism and accumulate steps, which will lead to negative sample reduction.
3. finetune-2: ... --student_model=lrc_bert --lrc_bert_alpha=1 --distill_ft_soft --distill_ft_soft_kl --distill_soft_rate=1 --distill_ft_hard --distill_hard_rate=3 --distill_temperature=1.1 --lrc_bert_gard_perturb --ignore_first_backward_gard --forward_repeat_num=1


# Multi-teacher distillation
- General parameters: ... --mt_num_attention_heads=a1:a2 --mt_hidden_size=h1:h2 --mt_num_layers=l1:l2 --mt_max_position_embeddings=m1:m2 --mt_load_pretrained=p1:p2 --teacher_fp16

## TMKD (similar)
1. pretrain: ... --student_model=kd --distill_pt_soft --distill_pt_soft_mse --distill_only_mask_pad --multi_teacher_model=tmkd --student_truncate_tn=0
2. finetune: ... --student_model=kd --distill_ft_soft --distill_ft_soft_mse --distill_only_mask_pad --distill_ft_hard --distill_hard_rate=1/teacher_num --multi_teacher_model=tmkd

## MT-BERT (similar)
- finetune: ... --student_model=pkd --distill_ft_soft --distill_temperature=1 --pkd_alpha=1 --pkd_beta=1 --student_truncate_tn=0 --multi_teacher_model=mt_bert --mt_has_loss --mt_bert_fit_teacher

## Uncertainty
- finetune: ... --student_model=kd --distill_ft_soft --distill_temperature=1 --distill_ft_soft_kl --distill_only_mask_pad --student_truncate_tn=0 --multi_teacher_model=uncertainty --uncertainty_only_mask_pad --uncertainty_hard

## RL-KD (similar)
1. finetune-avg: ... --student_model=kd --distill_ft_soft --distill_temperature=10 --student_truncate_tn=0 --multi_teacher_model=rl_kd --rl_kd_only_mask_pad --rl_kd_only_avg --rl_kd_alpha=0.5
2. finetune-rl (One more base teacher): ... --student_model=kd --distill_ft_soft --distill_temperature=10 --multi_teacher_model=rl_kd --rl_kd_only_mask_pad --rl_kd_reward=1 --rl_kd_semantic_model=0 --mt_has_loss --rl_kd_alpha=0.5

## MixMT
