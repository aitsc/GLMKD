# pretrain
## distill12-6+wikibook19G
deepspeed --master_port=11161 --include=localhost:2,3,4,5 distill/pretrain.py --deepspeed_config=config/config_block_tiny6.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=6 --hidden-size=768 --num-attention-heads=12 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --fp16 --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/tiny6 --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=120000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=150000 --no-lazy-loader --resume-dataloader --student_model=tinybert --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank --teacher_num_layers=12 --teacher_hidden_size=768 --teacher_num_attention_heads=12 --teacher_max_position_embeddings=512 --teacher_fp16
## distill24-4+wikibook19G-fp32
deepspeed --master_port=11161 --include=localhost:3,4,5,6 distill/pretrain.py --deepspeed_config=config/config_block_tiny4_fp32.json --deepspeed-activation-checkpointing --deepspeed --block-lm --num-layers=4 --hidden-size=768 --num-attention-heads=16 --max-position-embeddings=512 --tokenizer-model-type=bert-base-uncased --tokenizer-type=BertWordPieceTokenizer --checkpoint-activations --model-parallel-size=1 --save-interval=5000 --save=../GLM/data/checkpoints/distill/tiny4 --experiment-name=test --bert-prob=1.0 --train-data=bert-base --split=949,50,1 --distributed-backend=nccl --lr-decay-style=cosine --lr-decay-iters=160000 --lr-decay-ratio=0.05 --warmup=.05 --train-iters=200000 --no-lazy-loader --resume-dataloader --student_model=tinybert --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-large-blank --teacher_num_layers=24 --teacher_hidden_size=1024 --teacher_num_attention_heads=16 --teacher_max_position_embeddings=512
## distill12-4+wikibook19G (TinyBERT method)
```shell
deepspeed --master_port=12113 --include=localhost:0,2,3,4 \
    distill/pretrain.py \
    --deepspeed_config=config/config_block_tiny6.json \
    --deepspeed-activation-checkpointing \
    --deepspeed \
    --block-lm \
    --num-layers=4 \
    --hidden-size=768 \
    --num-attention-heads=12 \
    --max-position-embeddings=512 \
    --tokenizer-model-type=bert-base-uncased \
    --tokenizer-type=BertWordPieceTokenizer \
    --checkpoint-activations \
    --model-parallel-size=1 \
    --save-interval=5000 \
    --save=../GLM/data/checkpoints/distill/tiny4 \
    --experiment-name=test \
    --bert-prob=1.0 \
    --train-data=bert-base \
    --split=949,50,1 \
    --distributed-backend=nccl \
    --lr-decay-style=cosine \
    --lr-decay-iters=120000 \
    --lr-decay-ratio=0.05 \
    --warmup=.05 \
    --train-iters=150000 \
    --no-lazy-loader \
    --resume-dataloader \
    --student_model=tinybert \
    --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-base-blank \
    --teacher_num_layers=12 \
    --teacher_hidden_size=768 \
    --teacher_num_attention_heads=12 \
    --teacher_max_position_embeddings=512 \
    --teacher_fp16 \
    --fp16
```

## distill24-4+wikibook19G (TinyBERT method)
```shell
deepspeed --master_port=12178 --include=localhost:1,2,3,4 \
    distill/pretrain.py \
    --deepspeed_config=config/config_block_tiny4_fp32.json \
    --deepspeed-activation-checkpointing \
    --deepspeed \
    --block-lm \
    --num-layers=4 \
    --hidden-size=768 \
    --num-attention-heads=16 \
    --max-position-embeddings=512 \
    --tokenizer-model-type=bert-base-uncased \
    --tokenizer-type=BertWordPieceTokenizer \
    --checkpoint-activations \
    --model-parallel-size=1 \
    --save-interval=5000 \
    --save=../GLM/data/checkpoints/distill/tiny4 \
    --experiment-name=test \
    --bert-prob=1.0 \
    --train-data=bert-base \
    --split=949,50,1 \
    --distributed-backend=nccl \
    --lr-decay-style=cosine \
    --lr-decay-iters=160000 \
    --lr-decay-ratio=0.05 \
    --warmup=.05 \
    --train-iters=200000 \
    --no-lazy-loader \
    --resume-dataloader \
    --student_model=tinybert \
    --teacher_load_pretrained=../GLM/data/checkpoints/pretrain/blocklm-large-blank \
    --teacher_num_layers=24 \
    --teacher_hidden_size=1024 \
    --teacher_num_attention_heads=16 \
    --teacher_max_position_embeddings=512
```
