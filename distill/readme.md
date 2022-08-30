# 蒸馏命令
- pretrain ... : NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --master_port=13761 --include=localhost:0,7 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=384 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/tiny6 --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank --teacher_num_layers=12 --teacher_hidden_size=768 --teacher_num_attention_heads=12 --teacher_max_position_embeddings=512 --teacher_fp16

## KD
- pretrain(unnecessary): ... --student_model=kd --distill_pt_soft --distill_pt_hard --distill_temperature=10
- finetune: ... --student_model=kd --distill_ft_soft --distill_ft_hard --distill_temperature=10
    - --distill_ft_soft_kl

## TinyBERT
1. pretrain-inter: ... --student_model=tinybert
2. finetune-inter: ... --student_model=tinybert
3. finetune-pre: ... --student_model=tinybert --distill_ft_soft --tinybert_wo_inter
    - --distill_ft_soft_kl

## MiniLMv2
pretrain: ... --student_model=minilmv2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12

## MiniLM
pretrain: ... --student_model=minilm

## DistilBERT
- pretrain: ... --student_model=distilbert --distill_temperature=2 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding --distilbert_ce_mask_padding

## ERDistill
1. pretrain: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter=all --distill_pt_hard
    - + --distill_pt_soft --erdistill_inter_mse --erdistill_wo_global --distill_temperature=10.
2. finetune: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter=all --distill_temperature=10. --distill_ft_soft --distill_ft_hard
    - + --erdistill_inter_mse --erdistill_wo_global
3. finetune: ... --student_model=erdistill --erdistill_inter_mse --erdistill_inter= --distill_temperature=10. --distill_ft_soft --distill_ft_hard
    - + --erdistill_inter_mse

## MixBaseline (KD + TinyBERT + MiniLMv2 + MiniLM + DistilBERT)
1. pretrain: ... --student_model=mixbaseline --distill_temperature=10. --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12
2. finetune: ... --student_model=mixbaseline --distill_temperature=10. --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12
3. finetune: ... --student_model=mixbaseline --mixbaseline_wo_inter --tinybert_wo_inter --distill_ft_soft --distill_ft_hard --distill_temperature=10. --mixbaseline_tinybert_t=1.
    - --distill_ft_soft_kl

## PKD
finetune: ... --student_model=pkd --distill_ft_soft --distill_ft_soft_kl --distill_ft_hard --distill_temperature=10 --pkd_normalized_patience --pkd_alpha=0.5 --pkd_beta=100
