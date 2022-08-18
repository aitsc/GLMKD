# 蒸馏命令
## KD
finetune: ... --student_model=kd --distill_ft_soft --distill_ft_hard --distill_temperature=10.

## TinyBERT
1. pretrain-inter: ... --student_model=tinybert
2. finetune-inter: ... --student_model=tinybert
3. finetune-pre: ... --student_model=tinybert --distill_ft_soft --tinybert_wo_inter

## MiniLMv2
pretrain: ... --student_model=minilmv2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12

## MiniLM
pretrain: ... --student_model=minilm

## DistilBERT
pretrain: ... --student_model=distilbert --distill_temperature=10. --distilbert_alpha_ce=0.33 --distilbert_alpha_mlm=0.33 --distilbert_alpha_cos=0.33

## MixBaseline
1. pretrain: ... --student_model=mixbaseline --distill_temperature=10.
2. finetune: ... --student_model=mixbaseline --distill_temperature=10.
3. finetune: ... --student_model=mixbaseline --mixbaseline_wo_inter --tinybert_wo_inter --distill_ft_soft --distill_ft_hard --distill_temperature=10. --mixbaseline_tinybert_t=1.
