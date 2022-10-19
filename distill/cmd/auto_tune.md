## 直接微调
python -u distill/auto_tune.py --py_file=distill/finetune.py \
    --gpus=2 \
    --model=block_tiny6 \
    --model_path=/mnt/yrfs/glm_data/tsc/distill6/distill12-6.384_wb19G_mix17 \
    --tasks=copa,wsc_generative,cb,rte,boolq,wic,wsc,multirc,record \
    --hidden-size=384
