# GLMD
GLMD: General Language Model Distillation without Intermediate Layer Features and Hard Labels

- GLMD saves the tedious work on intermediate layers and golden labels, which allows distillation between different model structures without labeled dataset or the selection of intermediate layers.
- GLMD introduces a novel vocabulary compression method that further helps reducing the final model size.
- GLMD is implemented based on the GKD framework.

For more details about the techniques of GLMD, refer to our paper:

[Are Intermediate Layers and Labels Really Necessary? A General Language Model Distillation Method](https://arxiv.org/abs/2306.06625)

## Related Model Files
Download link: https://pan.baidu.com/s/1Q2lUY96Ix5emMAb-fkJgbQ?pwd=wwm8
- The fine-tuning teacher models for glm-large, glm-base, glm-2b, glm-10b, and ibglm-large.
- The optimal models related to GLMD, along with models from certain other methods.
- The pretraining teacher models for ibglm-large and alglm-base.

# GKD
GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model

- It provides a flexible architecture to efficiently implement various language model distillation methods, while allowing the use of a combination of these methods.
- We have introduced techniques such as model parallelism ([Megatron-LM](https://github.com/NVIDIA/Megatron-LM)) and ZeRO ([DeepSpeed](https://github.com/microsoft/DeepSpeed)) in the toolkit to make it efficient for distilling very large models.

For more details about the techniques of GKD, refer to our paper:

[GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model](https://arxiv.org/abs/2306.06629)

## Get Started
### Docker Image
We prepare a docker image based on Python 3.8.13, PyTorch 1.9.1, and CUDA 11.1. You can pull the pre-built images from Docker Hub and run with docker v19.03+
  ```shell
  docker run --gpus all --rm -it --ipc=host aitsc/glm:v1.5
  ```
### Manual Installation
```shell
git clone https://github.com/THUDM/GKD
cd GKD
conda create -n GLM python=3.8
conda activate GLM
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 22.04-dev
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd .. && rm -rf apex
```

### Model Parallelism
If your encounter the `CUDA out of memory` error, which means you GPU memory is limited, you can try the model parallelism to divide the parameters into multiple GPUs. Take the two-way model parallelism as an example. First run `change_mp.py` to divide the checkpoint:
```shell
python change_mp.py path_to_the_checkpoint 2
```
Then change `--model-parallel-size` in the command to `2`.

## Usage of existing methods
We provide commands for distilling GLM on all methods with deepspeed.

Suppose we want to distill a 12-layer teacher model to a 6-layer student model and test it on the ReCoRD dataset. We can first define 4 command prefixes that are not related to the specific method.
```shell
1. Prefix-pretrain: NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --master_port=13761 --include=localhost:0,1 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=768 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/tiny6 --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader
```
```shell
2. Prefix-finetune: NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2 deepspeed --master_port=20696 --include=localhost:0 --hostfile= distill/finetune.py --finetune --cloze-eval --experiment-name=blank-tiny6-ReCoRD-test --task=ReCoRD --data-dir=../GLM/data/english_data/superglue/ReCoRD --save=../GLM/data/checkpoints/distill/tiny6/test/ft --seq-length=512 --checkpoint-activations --eval-batch-size=16 --save-epoch=100000 --block-lm --num-layers=6 --hidden-size=768 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --load-pretrained=../GLM/data/checkpoints/distill/tiny6/test --fp16 --lr-decay-style=linear --warmup=0.1 --weight-decay=1.0e-1 --pattern-id=0 --save-interval=10000 --log-interval=50 --eval-interval=1000 --eval-iters=100 --batch-size=8 --epochs=5 --lr=1e-5 --overwrite --deepspeed-activation-checkpointing --deepspeed --deepspeed_config=config/config_block_tiny6.json --custom_first_eval
```
```shell
3. Prefix-single-teacher: --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank --teacher_num_layers=12 --teacher_hidden_size=768 --teacher_num_attention_heads=12 --teacher_max_position_embeddings=512 --teacher_fp16
```
```shell
4. Prefix-multi-teacher: --mt_num_attention_heads=a1:a2 --mt_hidden_size=h1:h2 --mt_num_layers=l1:l2 --mt_max_position_embeddings=m1:m2 --mt_load_pretrained=p1:p2 --teacher_fp16
```
Then we can build commands of different methods.
(see [distill/readme.md](distill/readme.md) and [distill/prepare.py](distill/prepare.py) for more detailed descriptions and parameters)
### GLMD-vc
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=logitsdistil --distill_temperature=15 --logitsdistil_mask_pad
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=logitsdistil --distill_temperature=15 --logitsdistil_mask_pad
3. [Prefix-finetune] [Prefix-single-teacher] --student_model=logitsdistil --distill_ft_soft --logitsdistil_wo_inter
### KD
1. [Prefix-finetune] [Prefix-single-teacher] --student_model=kd --distill_ft_soft --distill_ft_hard --distill_temperature=10
### PD
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=kd --distill_ft_soft --distill_temperature=1
### TinyBERT
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=tinybert
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=tinybert
3. [Prefix-finetune] [Prefix-single-teacher] --student_model=tinybert --distill_ft_soft --tinybert_wo_inter
### MiniLMv2
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=minilmv2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12
2. [Prefix-finetune]
### MiniLM
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=minilm
2. [Prefix-finetune]
### DistilBERT
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=distilbert --distill_temperature=2 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding
2. [Prefix-finetune]
### PKD
1. [Prefix-finetune] [Prefix-single-teacher] --student_model=pkd --distill_ft_soft --distill_ft_soft_kl --distill_ft_hard --distill_temperature=10 --pkd_normalized_patience --pkd_alpha=0.5 --pkd_beta=100 --student_truncate_tn=0 --pkd_wo_final --pkd_only_cls
### RAIL_KD
1. from DistilBERT
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=rail_kd --distill_ft_soft --distill_soft_rate=0.3333 --distill_ft_hard --distill_hard_rate=0.3333 --distill_temperature=10 --rail_kd_inter_rate=0.3333 --rail_kd_layer_wise_alpha=1 --rail_kd_u=128 --rail_kd_concatenated --rail_kd_epochs=1 --rail_kd_show_hook_change
### MGSKD
1. from TinyBERT
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=mgskd --mgskd_weight_sample=4 --mgskd_weight_token=1 --mgskd_weight_span=1 --mgskd_sample_level_m=3 --mgskd_triplet_k1=20 --mgskd_triplet_k2=20
3. [Prefix-finetune] [Prefix-single-teacher] --student_model=mgskd --distill_ft_soft --distill_ft_soft_kl --distill_temperature=1 --mgskd_wo_inter
### DIITO
1. [Prefix-pretrain](w/o --checkpoint-activations) [Prefix-single-teacher] --student_model=diito --forward_repeat_num=1 --diito_alignment=full --diito_interchange_prop=0.3 --diito_interchange_way=consecutive --diito_interchange_max_token=-1 --diito_alpha_mlm=0.25 --diito_alpha_ce=0.25 --diito_alpha_causal_ce=0.25 --diito_alpha_cos=0.25 --diito_alpha_causal_cos=0  --distill_pt_soft --distill_pt_hard --distill_temperature=2
2. [Prefix-finetune]
### SID
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=sid --sid_accumulate_t=0 --sid_lim_e=avg --distill_ft_soft --distill_temperature=1
### ALP_KD
1. [Prefix-finetune] [Prefix-single-teacher] --student_model=alp_kd --alp_kd_lambda=0.2 --distill_soft_rate=0.7 --distill_hard_rate=0.1 --distill_temperature=20 --student_truncate_tn=0 --distill_ft_soft --distill_ft_hard
### CKD
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=ckd --ckd_window_size=21 --ckd_wrdist_w=1 --ckd_ltrdist_w=1 --ckd_wrangle_w=10 --ckd_ltrangle_w=10 --distill_ft_soft --distill_ft_hard --distill_temperature=3 --distill_soft_rate=0.9 --distill_hard_rate=0.1
### Theseus
1. [Prefix-finetune] [Prefix-single-teacher] --student_model=theseus --distill_ft_hard --student_truncate_tn=0 --theseus_replacing_rate=0.3 --theseus_not_replaced_steps=0.66 --mt_disable_operation=1
### Universal_KD
1. [Prefix-finetune] [Prefix-single-teacher] --student_model=universal_kd --distill_ft_soft --distill_ft_soft_kl --distill_soft_rate=0.5 --universal_kd_gamma=0.5 --student_truncate_tn=0 --universal_kd_size=0
2. [Prefix-finetune]
### LRC_BERT
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=lrc_bert --lrc_bert_gard_perturb --ignore_first_backward_gard --forward_repeat_num=1 --lrc_bert_gather_dp --fix_variable_num_choices
3. [Prefix-finetune] [Prefix-single-teacher] --student_model=lrc_bert --lrc_bert_alpha=1 --distill_ft_soft --distill_ft_soft_kl --distill_soft_rate=1 --distill_ft_hard --distill_hard_rate=3 --distill_temperature=1.1 --lrc_bert_gard_perturb --ignore_first_backward_gard --forward_repeat_num=1 --lrc_bert_gather_dp --fix_variable_num_choices
### Annealing_KD
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=annealing_kd --annealing_kd_max_t=7 --distill_ft_soft --distill_ft_soft_mse
3. [Prefix-finetune]
### MobileBERT
1. [Prefix-pretrain] --inverted_bottleneck_mode --ib_hidden_size=1024 --ib_ffn_num=1 --hidden-size=512 --num-attention-heads=4 --ib_word_emb=128
2. [Prefix-pretrain] [Prefix-single-teacher] --student_model=mobilebert --mobilebert_kd_w=0.5 --mobilebert_pkt_small_lr=0.1 --distill_pt_hard --inverted_bottleneck_mode --ib_hidden_size=128 --ib_ffn_num=4 --hidden-size=512 --num-attention-heads=4 --ib_word_emb=128 --teacher_inverted_bottleneck_mode --teacher_ib_hidden_size=1024 --teacher_ib_ffn_num=1 --teacher_hidden_size=512 --teacher_num_attention_heads=4 --teacher_ib_word_emb=128
3. [Prefix-finetune] --inverted_bottleneck_mode --ib_hidden_size=128 --ib_ffn_num=4 --hidden-size=512 --num-attention-heads=4 --ib_word_emb=128
### Continuation_KD
1. [Prefix-pretrain]
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=continuation_kd --continuation_kd_max_t=10 --continuation_kd_margin=1 --continuation_kd_psi_sep=0.666 --continuation_kd_psi_denominator=1.333 --distill_ft_soft --distill_ft_soft_mse --distill_ft_hard
### TMKD
1. [Prefix-pretrain] [Prefix-multi-teacher] --student_model=kd --distill_pt_soft --distill_pt_soft_mse --multi_teacher_model=tmkd --student_truncate_tn=0
2. [Prefix-finetune] [Prefix-multi-teacher] --student_model=kd --distill_ft_soft --distill_ft_soft_mse --distill_ft_hard --distill_hard_rate=1/teacher_num --multi_teacher_model=tmkd
### MT-BERT
1. [Prefix-finetune] [Prefix-multi-teacher] --student_model=pkd --distill_ft_soft --distill_temperature=1 --pkd_alpha=1 --pkd_beta=1 --student_truncate_tn=0 --multi_teacher_model=mt_bert --mt_has_loss --mt_bert_fit_teacher
### Uncertainty
1. [Prefix-finetune] [Prefix-multi-teacher] --student_model=kd --distill_ft_soft --distill_temperature=1 --distill_ft_soft_kl --distill_soft_rate=0.5 --distill_hard_rate=0.5 --student_truncate_tn=0 --multi_teacher_model=uncertainty --uncertainty_hard
### RL-KD
1. [Prefix-finetune] [Prefix-multi-teacher] --student_model=kd --distill_ft_soft --distill_temperature=10 --student_truncate_tn=0 --multi_teacher_model=rl_kd --rl_kd_only_avg --rl_kd_alpha=0.5 --rl_kd_semantic_model_dim=768
2. [Prefix-finetune] [Prefix-multi-teacher](One more base teacher) --student_model=kd --distill_ft_soft --distill_temperature=10 --multi_teacher_model=rl_kd --rl_kd_reward=1 --rl_kd_semantic_model=0 --mt_has_loss --rl_kd_alpha=0.5 --fix_variable_num_choices
### ALBERT
1. [Prefix-pretrain] --compress_word_emb=128 --cross_layer_parameter_sharing
2. [Prefix-finetune] --compress_word_emb=128 --cross_layer_parameter_sharing
### Other
- TAKD simply replaces [Prefix-single-teacher] with the student from the previous training using any of the single-teacher methods.
- DGKD just needs to replace [Prefix-multi-teacher] with all the teachers and students previously trained using any of the multi-teacher methods.
- Support for more robust distillation using randomly disturbed data. For example, add the parameters: --distill_random_data=replace --distill_random_data_n=1 --forward_repeat_num=0 --distill_random_data_method=sample at the end of the command.

### Combined use of methods
For example: (pt/ft1: TinyBERT + MiniLMv2 + MiniLM + DistilBERT, ft2: KD + TinyBERT)
1. [Prefix-pretrain] [Prefix-single-teacher] --student_model=mixbaseline --distill_temperature=2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding --mixbaseline_inter_bl=TinyBERT,MiniLMv2,MiniLM,DistilBERT --mixbaseline_pre_bl_pt_soft=DistilBERT
2. [Prefix-finetune] [Prefix-single-teacher] --student_model=mixbaseline --distill_temperature=2 --minilmv2_relation_heads=48 --minilmv2_teacher_layer=12 --distilbert_alpha_ce=5 --distilbert_alpha_mlm=2 --distilbert_alpha_cos=1 --distilbert_cos_mask_padding --mixbaseline_inter_bl=TinyBERT,MiniLMv2,MiniLM,DistilBERT --mixbaseline_pre_bl_ft_soft=DistilBERT
3. [Prefix-finetune] [Prefix-single-teacher] --student_model=mixbaseline --mixbaseline_wo_inter --tinybert_wo_inter --distill_ft_soft --distill_ft_hard --distill_temperature=10 --mixbaseline_tinybert_t=1 --mixbaseline_pre_bl_ft_soft=TinyBERT

## Examples
### GLMD+al 110M-66M
pre-training stage
```shell
deepspeed --master_port=12761 --include=localhost:4,5,6,7 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=768 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/paper --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank --teacher_num_layers=12 --teacher_hidden_size=768 --teacher_num_attention_heads=12 --teacher_max_position_embeddings=512 --teacher_fp16 --student_model=logitsdistil --distill_temperature=15 --map_vocab_size=0.5 --distill_logit_mask_map --student_build_map_vocab --student_map_vocab_tn=0 --student_map_vocab_method=decoder --unmap_vocab_output --logitsdistil_mask_pad --compress_word_emb=384
```
task-specific stage
```shell
python -u distill/auto_tune.py --py_file=distill/finetune.py \
    --gpus=2,3 \
    --model=block_tiny6 \
    --model_path=checkpoints/distill/paper/12.768-6.768_64-15w_glmd-dta_vc.5de-albert \
    --task_t_load=base \
    --tasks=record,copa,wsc,rte,boolq,wic,cb,multirc,wsc_generative \
    --student_model=logitsdistil --distill_temperature=15 --map_vocab_size=0.5 --distill_logit_mask_map --unmap_vocab_output --logitsdistil_mask_pad --compress_word_emb=384 --del_checkpoint_activations \
    --again_1__distill_ft_soft \
    --again_1__distill_temperature=1 \
    --again_1__logitsdistil_wo_inter \
    --seed=1759 \
    --ds_train_micro_batch_size_per_gpu="16;16;32;16;16;16;32;16;32" \
    --ds_gradient_accumulation_steps=2 \
    --rate_ds_train_micro_batch_size_per_gpu=0.25 \
    --rate_ds_gradient_accumulation_steps=1 \
    --ds_optimizer__params__lr="2E-05;5E-06;2E-05;2E-05;2E-05;2E-05;2E-05;5E-06;2E-05"
```
### GLMD-vc 10B-2B
pre-training stage
```shell
deepspeed --master_port=31761 --include=localhost:0,1,2,3,4,5,6,7 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --task-mask --num-layers=36 --hidden-size=2048 --num-attention-heads=32 --max-position-embeddings=1024 --tokenizer-type=GPT2BPETokenizer --checkpoint-activations --model-parallel-size=4 --save-interval=5000 --save=../GLM/data/checkpoints/distill/paper --experiment-name=10b --bert-prob=0.5 --gap-sentence-prob=0.3 --avg-block-length=3 --gpt-min-ratio=0.25 --block-mask-prob=0.1 --short-seq-prob=0.02 --train-data=bert-large --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.1 --warmup=.04 --train-iters=150000 --no-lazy-loader --resume-dataloader --log-interval=50 --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-xxlarge_MP4 --teacher_num_layers=48 --teacher_hidden_size=4096 --teacher_num_attention_heads=64 --teacher_max_position_embeddings=1024 --teacher_fp16 --batch-size=4 --gradient-accumulation-steps=8 --args_to_ds_config --student_model=logitsdistil --distill_temperature=15 --logitsdistil_mask_pad
```
task-specific stage
```shell
python -u distill/auto_tune.py --py_file=distill/finetune.py \
    --gpus=0,1,2,3,4,5,6,7 \
    --model=model_blocklm_10B \
    --num-layers=36 \
    --hidden-size=2048 \
    --num-attention-heads=32 \
    --model_path=checkpoints/distill/paper/48.4096-36.2048_64-15w_glmd-vc-dta_MP4 \
    --model-parallel-size=2 \
    --task_t_load=10b \
    --tasks=record \
    --student_model=logitsdistil --distill_temperature=15 --logitsdistil_mask_pad \
    --again_1__distill_ft_soft \
    --again_1__distill_temperature=1 \
    --again_1__logitsdistil_wo_inter \
    --seed=1234 \
    --ds_train_micro_batch_size_per_gpu=1 \
    --ds_gradient_accumulation_steps=4 \
    --ds_optimizer__params__lr=1E-05 --eval-batch-size=4
```
### TinyBERT 340M-66M
pre-training stage
```shell
deepspeed --master_port=18161 --include=localhost:4,5,6,7 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=768 --num-attention-heads=16 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/paper --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-large-blank --teacher_num_layers=24 --teacher_hidden_size=1024 --teacher_num_attention_heads=16 --teacher_max_position_embeddings=512 --teacher_fp16 --student_model=tinybert
```
task-specific stage
```shell
python -u distill/auto_tune.py --py_file=distill/finetune.py \
    --gpus=0,1,2,3,4,5,6,7 \
    --model=block_tiny6 \
    --model_path=checkpoints/distill/paper/24.1024-6.768_64-15w_tinybert \
    --task_t_load=large \
    --tasks=record,copa,wsc,rte,boolq,wic,cb,multirc,wsc_generative \
    --num-attention-heads=16 \
    --student_model=tinybert \
    --again_1__distill_ft_soft \
    --again_1__tinybert_wo_inter \
    --seed=6899 \
    --ds_train_micro_batch_size_per_gpu="32;32;32;32;16;16;16;16;16" \
    --ds_gradient_accumulation_steps=1 \
    --rate_ds_train_micro_batch_size_per_gpu=0.125 \
    --rate_ds_gradient_accumulation_steps=1 \
    --ds_optimizer__params__lr="1E-05;5E-06;2E-05;1E-05;1E-05;1E-05;2E-05;2E-05;5E-06"
```
For a more detailed, task-specific distillation stage, please refer to [distill/ft_logs](distill/ft_logs).

## Create a new distillation method
Implementing the new distillation method only requires adding a class to the [distill/distill_model.py](distill/distill_model.py) file. For example, a simple implementation for middle layer distillation:
```python
class MethodName(GLMStudent):
    def __init__(self, language_model, args, **kwargs):
        super().__init__(language_model, args, **kwargs)
    def get_teacher_hook(self, **kwargs):
        return {The intermediate layer you want to use}
    def get_student_hook(self, **kwargs):
        return {The intermediate layer you want to use}
    def inter_loss(self, s_inter_vars, t_inter_vars, s_hook, t_hook, **kwargs):
        return Calculate the loss of intermediate layers
```

# Citation
Part of the code is based on [GLM](https://github.com/THUDM/GLM).

Please cite our papers if you find this code useful for your research:
```
@article{tan2023glmd,
  author    = {Shicheng Tan and
               Weng Lam Tam and
               Yuanchun Wang and
               Wenwen Gong and
               Shu Zhao and
               Peng Zhang and
               Jie Tang},
  title     = {Are Intermediate Layers and Labels Really Necessary? A General Language Model Distillation Method},
  booktitle = {ACL},
  year      = {2023},
}

@article{tan2023gkd,
  author    = {Shicheng Tan and
               Weng Lam Tam and
               Yuanchun Wang and
               Wenwen Gong and
               Shu Zhao and
               Peng Zhang and
               Jie Tang},
  title     = {GKD: A General Knowledge Distillation Framework for Large-scale Pre-trained Language Model},
  booktitle = {ACL},
  year      = {2023},
}
```
