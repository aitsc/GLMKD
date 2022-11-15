from datetime import datetime
import re
import os
from collections import OrderedDict
import sys
from pprint import pprint
import json
import random
import argparse
import time
import copy
from tsc_base import put, get
import random
import traceback

ap = 'data'  # 总目录

class Models:
    @staticmethod
    def model_blocklm_base(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-base"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/blocklm-base-blank"  # 官方模型
        env['MODEL_ARGS'] = ([
            ('--block-lm', None),
            ('--num-layers', '12'), 
            ('--hidden-size', '768'), 
            ('--num-attention-heads', '12'), 
            ('--max-position-embeddings', '512'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
            # ('--fp32-allreduce', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_tiny6{suffix}.json'  # 暂时与 model_blocklm_base 相同, 保证 suffix
        return env

    @staticmethod
    def block_tiny6(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-tiny6"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/block_tiny6"
        env['MODEL_ARGS'] = ([
            ('--block-lm', None), 
            ('--num-layers', '6'), 
            ('--hidden-size', '768'), 
            ('--num-attention-heads', '12'), 
            ('--max-position-embeddings', '512'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_tiny6{suffix}.json'
        return env

    @staticmethod
    def model_blocklm_large(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-large"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/blocklm-large-blank"  # 官方模型
        env['MODEL_ARGS'] = ([
            ('--block-lm', None),
            ('--num-layers', '24'), 
            ('--hidden-size', '1024'), 
            ('--num-attention-heads', '16'), 
            ('--max-position-embeddings', '512'), 
            ('--tokenizer-model-type', 'bert-large-uncased'), 
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_large{suffix}.json'
        return env

    @staticmethod
    def block_tiny4(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-tiny4"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/block_tiny4"
        env['MODEL_ARGS'] = ([
            ('--block-lm', None), 
            ('--num-layers', '4'), 
            ('--hidden-size', '768'), 
            ('--num-attention-heads', '12'), 
            ('--max-position-embeddings', '512'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_tiny6{suffix}.json'
        return env

    @staticmethod
    def block_tiny24_4(env: dict, **kw):
        env['MODEL_TYPE'] = "blank-tiny4"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/block_tiny4"
        env['MODEL_ARGS'] = ([
            ('--block-lm', None), 
            ('--num-layers', '4'), 
            ('--hidden-size', '768'), 
            ('--num-attention-heads', '16'), 
            ('--max-position-embeddings', '512'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_tiny4_fp32{suffix}.json'
        return env

    @staticmethod
    def model_blocklm_10B(env: dict, **kw):
        env['MODEL_TYPE'] = "blocklm-10B"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/blocklm-xxlarge"  # 官方模型
        env['MODEL_ARGS'] = ([
            ('--block-lm', None),
            ('--task-mask', None), 
            ('--num-layers', '48'), 
            ('--hidden-size', '4096'), 
            ('--num-attention-heads', '64'), 
            ('--max-position-embeddings', '1024'), 
            ('--tokenizer-type', 'GPT2BPETokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_10B_1gpu{suffix}.json'
        return env

    @staticmethod
    def model_blocklm_10B_chinese(env: dict, **kw):
        env['MODEL_TYPE'] = "blocklm-10B"
        if env.get('MODEL_PATH') is None:
            env['MODEL_PATH'] = f"{ap}/checkpoints/pretrain/blocklm-xxlarge-zh"  # 官方模型
        env['MODEL_ARGS'] = ([
            ('--block-lm', None),
            ('--task-mask', None), 
            ('--num-layers', '48'), 
            ('--hidden-size', '4096'), 
            ('--num-attention-heads', '64'), 
            ('--max-position-embeddings', '1024'), 
            ('--tokenizer-type', 'ChineseSPTokenizer'), 
            ('--load-pretrained', env['MODEL_PATH']),
            ('--fp16', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config_tasks/config_blocklm_10B_1gpu{suffix}.json'
        return env


class Models_pre:
    @staticmethod
    def block_tiny6(env: dict, **kw):
        env['gpt_options'] = ([
            ('--block-lm', None), 
            ('--bert-prob', '1.0'), 
            ('--experiment-name', 'blocklm-blank'), 
            ('--num-layers', '6'), 
            ('--hidden-size', '768'),
            ('--num-attention-heads', '12'),
            ('--seq-length', '512'),
            ('--max-position-embeddings', '512'), 
            ('--save', '{ap}/checkpoints/pretrain/block_tiny6'),  # 模型保存位置
            # f'--load', '{ap}/checkpoints/pretrain/block_tiny6/blocklm-blank07-31-07-36'),  # 保存文件夹名会和这个一样
            ('--resume-dataloader', None),
            ('--train-data', 'wiki'),
            ('--no-lazy-loader', None),
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--split', '949,50,1'),
            ('--distributed-backend', 'nccl'),
            ('--lr-decay-style', 'cosine'),
            ('--lr-decay-iters', '120000'),
            ('--train-iters', '150000'),
            ('--lr-decay-ratio', '0.05'),
            ('--warmup', '.05'),
            ('--fp16', None),  # 用 ds 还需要设置 deepspeed_config 中的 fp16
            # ('--fp32-allreduce', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config/config_block_tiny6{suffix}.json'
        return env

    @staticmethod
    def block_base(env: dict, **kw):
        env['gpt_options'] = ([
            ('--block-lm', None), 
            ('--bert-prob', '1.0'), 
            ('--experiment-name', 'blocklm-blank'), 
            ('--num-layers', '12'), 
            ('--hidden-size', '768'),
            ('--num-attention-heads', '12'),
            ('--seq-length', '512'),
            ('--max-position-embeddings', '512'), 
            ('--save', f'{ap}/checkpoints/pretrain/block_base'),  # 模型保存位置
            # ('--load', f'{ap}/checkpoints/pretrain/blocklm-base-blank'),  # 续跑
            ('--resume-dataloader', None),
            ('--train-data', 'bert-base'),
            ('--no-lazy-loader', None),
            ('--tokenizer-type', 'BertWordPieceTokenizer'), 
            ('--tokenizer-model-type', 'bert-base-uncased'), 
            ('--split', '949,50,1'),
            ('--distributed-backend', 'nccl'),
            ('--lr-decay-style', 'cosine'),
            ('--lr-decay-iters', '120000'),
            ('--train-iters', '150000'),  # 迭代几次
            ('--lr-decay-ratio', '0.05'),
            ('--warmup', '.05'),
            ('--fp16', None),  # 用 ds 还需要设置 deepspeed_config 中的 fp16
            # ('--fp32-allreduce', None),
        ])
        if env.get('deepspeed_config') is None:
            suffix = env['deepspeed_config_suffix'] if 'deepspeed_config_suffix' in env else ''
            env['deepspeed_config'] = f'config/config_block_base{suffix}.json'
        return env

class Tasks:
    BATCH_SIZE = '16'
    EPOCH_SINGLE = {
        'copa': '50', 'rte': '50', 'boolq': '20', 'wic': '30', 'cb': '50', 'multirc': '15',
        'wsc_generative': '50', 'wsc': '50', 'record': '5',
    }
    XXLARGE_EPOCH = {
        'copa': '100', 'rte': '50', 'boolq': '24', 'wic': '40', 'cb': '100', 'multirc': '12',
        'wsc_generative': '100', 'wsc': '100', 'record': '3',
    }
    EPOCH_0 = {
        'copa': '0', 'rte': '0', 'boolq': '0', 'wic': '0', 'cb': '0', 'multirc': '0',
        'wsc_generative': '0', 'wsc': '0', 'record': '0',
    }
    EPOCH_SINGLE_ = EPOCH_SINGLE

    @staticmethod
    def copa(env: dict, **kw):
        env['TASK_NAME'] = 'COPA'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '0'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '20'), 
            ('--eval-interval', '1000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1)'
        env['PROMPT_IDS'] = '(1 2)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def rte(env: dict, **kw):
        env['TASK_NAME'] = 'RTE'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '0'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '10000000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def boolq(env: dict, **kw):
        env['TASK_NAME'] = 'BoolQ'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '4'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '10000000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2 3 4 5)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wic(env: dict, **kw):
        env['TASK_NAME'] = 'WiC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '1'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '10000000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def cb(env: dict, **kw):
        env['TASK_NAME'] = 'CB'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '256'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '3'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '10000000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def multirc(env: dict, **kw):
        env['TASK_NAME'] = 'MultiRC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '512'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'), 
            ('--pattern-id', '0'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '10000000'), 
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2 3)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wsc_generative(env: dict, **kw):
        env['TASK_NAME'] = 'WSC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}_generative'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '128'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'), 
            ('--log-interval', '50'), 
            ('--eval-interval', '1000'), 
            ('--eval-iters', '100'),
        ])
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def wsc(env: dict, **kw):
        env['TASK_NAME'] = 'WSC'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}-negative'
        env['MAX_SEQ_LEN'] = '128'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'),
            ('--loss-func', 'mix'),
            ('--wsc-negative', None),
            ('--length-penalty', '1'),
            ('--pattern-id', '2'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0 1 2)'
        env['PROMPT_IDS'] = '(1 2 3)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def record(env: dict, **kw):
        env['TASK_NAME'] = 'ReCoRD'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/{env["TASK_NAME"]}'
        env['MAX_SEQ_LEN'] = '512'
        env['LR_SINGLE'] = '1e-5'
        env['EPOCH_SINGLE'] = Tasks.EPOCH_SINGLE[sys._getframe().f_code.co_name]
        env['TRAIN_ARGS'] = ([
            ('--lr-decay-style', 'linear'), 
            ('--warmup', '0.1'), 
            ('--weight-decay', '1.0e-1'),
            ('--pattern-id', '0'),
        ])
        env['COMMON_ARGS'] = ([
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
        ])
        env['PATTERN_IDS'] = '(0)'
        env['BATCH_SIZE'] = Tasks.BATCH_SIZE
        return env

    @staticmethod
    def zero_lambada(env: dict, **kw):
        env['TASK_NAME'] = 'lambda'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/lambada_test.jsonl'
        env['EVALUATE_ARGS'] = [
            ('--eval-batch-size', '16'), 
            ('--seq-length', '512'), 
        ]
        return env

    @staticmethod
    def zero_lambada_uni(env: dict, **kw):
        env['TASK_NAME'] = 'lambda'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}_uni'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/lambada_test.jsonl'
        env['EVALUATE_ARGS'] = [
            ('--eval-batch-size', '16'), 
            ('--seq-length', '512'), 
            ('--unidirectional', None), 
        ]
        return env

    @staticmethod
    def zero_lm(env: dict, **kw):
        env['TASK_NAME'] = 'language_model'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/bert-large-test.txt'
        env['EVALUATE_ARGS'] = [
            ('--eval-batch-size', '16'), 
            ('--seq-length', '512'), 
            ('--overlapping-eval', '256'), 
        ]
        return env

    @staticmethod
    def zero_lm_uni(env: dict, **kw):
        env['TASK_NAME'] = 'language_model'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}_uni'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/bert-large-test.txt'
        env['EVALUATE_ARGS'] = [
            ('--eval-batch-size', '16'), 
            ('--seq-length', '512'), 
            ('--overlapping-eval', '256'), 
            ('--unidirectional', None), 
        ]
        return env

    @staticmethod
    def zero_wikitext(env: dict, **kw):
        env['TASK_NAME'] = 'wikitext'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/wikitext-103/wiki.test.tokens'
        env['EVALUATE_ARGS'] = [
            ('--eval-batch-size', '16'), 
            ('--seq-length', '512'),  # --max-position-embeddings 能到 1024 就用 1024
            ('--overlapping-eval', '256'), 
        ]
        return env

    @staticmethod
    def seq_blank(env: dict, **kw):
        env['TASK_NAME'] = 'blank'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-blank-{env["MASK_RATIO"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/blank_yahoo'
        env['TRAIN_ARGS'] = [
            ('--epochs', '5'),
            ('--batch-size', '16'),
            ('--lr', '1e-5'),
            ('--lr-decay-style', 'linear'),
            ('--warmup', '0.06'),
            ('--weight-decay', '1.0e-1'),
            ('--label-smoothing', '0.1'),
            ('--blank-maskratio', env["MASK_RATIO"]),
        ]
        env['COMMON_ARGS'] = [
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
            # ('--eval-epoch', '100'),
        ]
        env['TASK_ARGS'] = [
            ('--src-seq-length', '256'),
            ('--tgt-seq-length', '200'),
            ('--min-tgt-length', '0'),
            ('--length-penalty', '1'),
            ('--no-repeat-ngram-size', '3'),
            ('--eval-batch-size', '8'),
        ]
        return env

    @staticmethod
    def seq_cnndm_org(env: dict, **kw):
        env['TASK_NAME'] = 'cnn_dm_original'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/cnn_dm_original'
        env['TRAIN_ARGS'] = [
            ('--epochs', '10'),
            ('--batch-size', '8'),
            ('--lr', '1e-5'),
            ('--lr-decay-style', 'linear'),
            ('--warmup', '0.06'),
            ('--weight-decay', '1.0e-1'),
            ('--label-smoothing', '0.1'),
        ]
        env['COMMON_ARGS'] = [
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
            ('--eval-epoch', '2'),
        ]
        env['TASK_ARGS'] = [
            ('--src-seq-length', '608'),
            ('--tgt-seq-length', '160'),
            ('--min-tgt-length', '55'),
            ('--length-penalty', '0.7'),
            ('--no-repeat-ngram-size', '3'),
            ('--num-beams', '5'),
            ('--select-topk', None),
            ('--eval-batch-size', '1'),
        ]
        return env

    @staticmethod
    def seq_cnndm(env: dict, **kw):
        env['TASK_NAME'] = 'cnn_dm_original'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/cnn_dm'
        env['TRAIN_ARGS'] = [
            ('--epochs', '15'),
            ('--batch-size', '8'),
            ('--lr', '3e-5'),
            ('--lr-decay-style', 'linear'),
            ('--warmup', '0.06'),
            ('--weight-decay', '1.0e-1'),
            ('--label-smoothing', '0.1'),
        ]
        env['COMMON_ARGS'] = [
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
        ]
        env['TASK_ARGS'] = [
            ('--src-seq-length', '608'),
            ('--tgt-seq-length', '160'),
            ('--min-tgt-length', '55'),
            ('--length-penalty', '0.7'),
            ('--no-repeat-ngram-size', '3'),
            ('--num-beams', '5'),
            ('--select-topk', None),
            ('--eval-batch-size', '4'),
        ]
        return env

    @staticmethod
    def seq_xsum(env: dict, **kw):
        env['TASK_NAME'] = 'xsum'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/bbc-summary-data'
        env['TRAIN_ARGS'] = [
            ('--epochs', '6'),
            ('--batch-size', '8'),
            ('--lr', '1e-5'),
            ('--lr-decay-style', 'linear'),
            ('--warmup', '0.06'),
            ('--weight-decay', '1.0e-1'),
            ('--label-smoothing', '0.1'),
        ]
        env['COMMON_ARGS'] = [
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
            ('--eval-epoch', '2'),
        ]
        env['TASK_ARGS'] = [
            ('--src-seq-length', '608'),
            ('--tgt-seq-length', '60'),
            ('--min-tgt-length', '10'),
            ('--length-penalty', '1.'),
            ('--no-repeat-ngram-size', '3'),
            ('--num-beams', '6'),
            ('--select-topk', None),
            ('--eval-batch-size', '1'),
        ]
        return env

    @staticmethod
    def seq_gigaword(env: dict, **kw):
        env['TASK_NAME'] = 'gigaword'
        env['EXPERIMENT_NAME'] = f'{env["MODEL_TYPE"]}-{env["TASK_NAME"]}'
        env['DATA_PATH'] = f'{env["DATA_ROOT"]}/gigaword/org_data'
        env['TRAIN_ARGS'] = [
            ('--epochs', '10'),
            ('--batch-size', '16'),
            ('--lr', '3e-5'),
            ('--lr-decay-style', 'linear'),
            ('--warmup', '0.06'),
            ('--weight-decay', '1.0e-1'),
            ('--label-smoothing', '0.1'),
        ]
        env['COMMON_ARGS'] = [
            ('--save-interval', '10000'),
            ('--log-interval', '50'),
            ('--eval-interval', '1000'),
            ('--eval-iters', '100'),
        ]
        env['TASK_ARGS'] = [
            ('--src-seq-length', '192'),
            ('--tgt-seq-length', '32'),
            ('--min-tgt-length', '0'),
            ('--length-penalty', '0.6'),
            ('--no-repeat-ngram-size', '3'),
            ('--num-beams', '5'),
            ('--select-topk', None),
            ('--eval-batch-size', '4'),
        ]
        return env

class Scripts:
    @staticmethod
    def finetune_superglue(model_f, task_f, env=None, n_gpu=1, save_sub=None, **kw):
        env = {} if env is None else env
        save_sub = save_sub if save_sub else 'finetune'
        env['DATA_ROOT'] = f'{ap}/english_data/superglue'  # 总数据位置
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = env['MODEL_PATH'] + f'/{save_sub}/' + env['TASK_NAME']
        env['N_GPU'] = f'{n_gpu}'  # BATCH_SIZE 均分到几张卡上
        env['PER_GPU_BS'] = str(int(int(env['BATCH_SIZE']) / int(env['N_GPU'])))
        env['EXPERIMENT_NAME'] = env['EXPERIMENT_NAME'] + '-' + datetime.now().strftime('%y%m%d_%H%M%S.%f')
        py_args = ([
            ('--finetune', None),
            ('--cloze-eval', None),
            ('--experiment-name', env['EXPERIMENT_NAME']),
            ('--task', env['TASK_NAME']),
            ('--data-dir', env['DATA_PATH']),
            ('--save', env['SAVE_PATH']),
            ('--seq-length', env['MAX_SEQ_LEN']),
            ('--checkpoint-activations', None),
            ('--eval-batch-size', '16'),
            ('--save-epoch', '100000'),  # 每轮微调保存可能会将 latest_checkpointed_iteration.txt 的 best 覆盖
            *env['MODEL_ARGS'],
            *env['TRAIN_ARGS'],
            *env['COMMON_ARGS'],
            ('--batch-size', env['PER_GPU_BS']),
            ('--epochs', env['EPOCH_SINGLE']),
            ('--lr', env['LR_SINGLE']),
            ('--overwrite', None),
            # ('--num-workers', '0'),  # 不使用多进程数据加载器方便调试
        ])
        return py_args

    @staticmethod
    def pretrain_nvidia(model_pre_f, env=None, **kw):
        env = {} if env is None else env
        model_pre_f(env)
        py_args = ([
            *env['gpt_options'],
            ('--checkpoint-activations', None),
            ('--model-parallel-size', '1'),  # 模型并行数, 常调参数
            # ('--save-interval', '100'),  # 迭代几次保存一次, 默认 5000
        ])
        return py_args

    @staticmethod
    def evaluate_lm(model_f, task_f, env=None, save_sub=None, **kw):
        env = {} if env is None else env
        save_sub = save_sub if save_sub else 'evaluate_lm'
        env['DATA_ROOT'] = f'{ap}/english_data/NLG'  # 总数据位置
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = env['MODEL_PATH'] + f'/{save_sub}/' + env['TASK_NAME']
        env['EXPERIMENT_NAME'] = env['EXPERIMENT_NAME'] + '-' + datetime.now().strftime('%y%m%d_%H%M%S.%f')
        py_args = ([
            ('--finetune', None),
            ('--experiment-name', env['EXPERIMENT_NAME']),
            ('--task', env['TASK_NAME']),
            ('--valid-data', env['DATA_PATH']),
            ('--save', env['SAVE_PATH']),
            ('--checkpoint-activations', None),
            ('--overwrite', None),
            ('--save-epoch', '100000'),
            *env['MODEL_ARGS'],
            *env['EVALUATE_ARGS'],
        ])
        return py_args

    @staticmethod
    def finetune_blank(model_f, task_f, env=None, save_sub=None, **kw):
        env = {} if env is None else env
        save_sub = save_sub if save_sub else 'finetune_blank'
        env['DATA_ROOT'] = f'{ap}/english_data/NLG'  # 总数据位置
        env['MASK_RATIO'] = env['MASK_RATIO'] if 'MASK_RATIO' in env else '0.1'  # 比例
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = env['MODEL_PATH'] + f'/{save_sub}/' + env['TASK_NAME']
        env['EXPERIMENT_NAME'] = env['EXPERIMENT_NAME'] + '-' + datetime.now().strftime('%y%m%d_%H%M%S.%f')
        py_args = ([
            ('--finetune', None),
            ('--experiment-name', env['EXPERIMENT_NAME']),
            ('--task', env['TASK_NAME']),
            ('--data-dir', env['DATA_PATH']),
            ('--save', env['SAVE_PATH']),
            ('--checkpoint-activations', None),
            ('--overwrite', None),
            ('--save-epoch', '100000'),
            *env['MODEL_ARGS'],
            *env['TRAIN_ARGS'],
            *env['COMMON_ARGS'],
            *env['TASK_ARGS'],
        ])
        return py_args

    @staticmethod
    def finetune_seq2seq(model_f, task_f, env=None, save_sub=None, **kw):
        env = {} if env is None else env
        save_sub = save_sub if save_sub else 'finetune_seq2seq'
        env['DATA_ROOT'] = f'{ap}/english_data/NLG'  # 总数据位置
        model_f(env)
        task_f(env)
        env['SAVE_PATH'] = env['MODEL_PATH'] + f'/{save_sub}/' + env['TASK_NAME']
        env['EXPERIMENT_NAME'] = env['EXPERIMENT_NAME'] + '-' + datetime.now().strftime('%y%m%d_%H%M%S.%f')
        py_args = [
            ('--finetune', None),
            ('--experiment-name', env['EXPERIMENT_NAME']),
            ('--task', env['TASK_NAME']),
            ('--data-dir', env['DATA_PATH']),
            ('--save', env['SAVE_PATH']),
            ('--checkpoint-activations', None),
            ('--num-workers', '1'),
            ('--no-load-lr-scheduler', None),
            ('--save-epoch', '100000'),
            *env['MODEL_ARGS'],
            *env['TRAIN_ARGS'],
            *env['COMMON_ARGS'],
            *env['TASK_ARGS'],
            ('--overwrite', None),
        ]
        return py_args


def py_args_to_line(py_args) -> str:
    if type(py_args) == list:
        py_args = OrderedDict(py_args)
    args = []
    for k, v in py_args.items():
        k = k.strip()
        if v in {None, True}:
            args.append(k)
        elif v in {False}:
            continue
        else:
            args.append(f"{k}={v.strip()}")
    return ' '.join(args)


def create_cmd(script, model=None, model_pre=None, task=None, ds=False, gpus='6', py_file=None, env=None, **kw):
    env = {} if env is None else env
    # 前缀
    if ds:
        prefix = ([
            ('NCCL_DEBUG', 'info'),
            ('NCCL_IB_DISABLE', '0'),
            ('NCCL_NET_GDR_LEVEL', '2'),
            ('deepspeed', None),
            ('--master_port', str(random.randint(10000, 60000))),
            ('--include', f'localhost:{gpus}'),  # 占用显卡
            ('--hostfile', ''),
        ])
    else:
        prefix = ([
            ('CUDA_VISIBLE_DEVICES', f'{gpus}'),  # 占用显卡
            ('python', None),
            ('-u', None),
        ])
    # py 文件
    if py_file:
        py = (py_file, None)
    else:
        py = (f"{script.__name__.split('_')[0]}_glm.py", None)
    # 主体变量
    n_gpu = gpus.count(',') + 1  # 不管预训练固定batch size\模型并行\deepspeed问题
    py_args = prefix + [py] + script(model_f=model, task_f=task, model_pre_f=model_pre, env=env, n_gpu=n_gpu, **kw)
    if ds:
        py_args += ([
            ('--deepspeed-activation-checkpointing', None),
            ('--deepspeed', None),
            ('--deepspeed_config', env['deepspeed_config']),
        ])
    return py_args


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def generate_unique():
    return datetime.now().strftime('%y%m%d_%H%M%S.%f') + '_' + str(random.random())[2:]


def auto_tune():
    global ap
    ap = os.path.expanduser('../GLM/data')
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--py_file', type=str, default='finetune_glm.py')
    py_parser.add_argument('--gpus', type=str, default='6')
    py_parser.add_argument('--model', type=str, default='block_tiny6')
    py_parser.add_argument('--model_path', type=str, default=None)  # checkpoints/pretrain/blocklm-base-blank
    py_parser.add_argument('--task_t_load', type=str, default=None)
    py_parser.add_argument('--save_sub', type=str, default=None)
    py_parser.add_argument('--test', action='store_true')
    py_parser.add_argument('--epoch_type', type=str, default='EPOCH_SINGLE')  # EPOCH_SINGLE,XXLARGE_EPOCH,EPOCH_0
    py_parser.add_argument('--tasks', type=str, default='copa,wsc_generative,cb,rte,boolq,wic,wsc,multirc,record')  # 可以大任务放在前面先使用多卡
    py_parser.add_argument('--script', type=str, default='finetune_superglue')  # finetune_superglue,evaluate_lm,finetune_blank,finetune_seq2seq
    py_parser.add_argument('--mask_ratio', type=str, default='0.1')
    py_parser.add_argument('--big_task_gpus', type=str, default=None, help='这个有值就会专门针对显存需求大的任务使用gpus和deepspeed_config')
    py_parser.add_argument('--big_task', type=str, default='record', help='针对这些任务(必须属于参数tasks)使用专门的gpus和deepspeed_config')
    py_parser.add_argument('--big_task_dsc_suffix', type=str, default='_big_task', help='大任务的 deepspeed_config 补充后缀')
    py_parser.add_argument('--show_tune', type=str, default='', help='只用来展示保存的tune文件,而不是运行')
    py_parser.add_argument('--rate_arg_epochs', type=float, default=1., help='倍率,对所有任务的epochs乘以这个倍率')
    # deepspeed_config 重构,会新建一个json文件用于模型调用
    # 多个值用英文分号分隔,与--tasks一一对应; 保证原始deepspeed配置文件中有对应值做类型转换; bool用有值和无值代替True/False
    py_parser.add_argument('--ds_train_micro_batch_size_per_gpu', type=str, default=None, help='')
    py_parser.add_argument('--ds_gradient_accumulation_steps', type=str, default=None, help='')
    py_parser.add_argument('--ds_optimizer__params__lr', type=str, default=None, help='')
    py_parser.add_argument('--rate_ds_train_micro_batch_size_per_gpu', type=float, default=1., help='倍率,配合gpus和ds多个值使用')
    py_parser.add_argument('--rate_ds_gradient_accumulation_steps', type=float, default=1., help='倍率,配合gpus和ds多个值使用')
    # 不同任务微调可以选择特定的模型加载
    for t in [
        'copa', 'wsc_generative', 'cb', 'rte', 'boolq', 'wic', 'wsc', 'multirc', 'record',
        'zero_lambada', 'zero_lambada_uni', 'zero_lm', 'zero_lm_uni', 'zero_wikitext', 
        'seq_blank', 'seq_cnndm_org', 'seq_cnndm', 'seq_xsum', 'seq_gigaword',
    ]:
        py_parser.add_argument(f'--{t}_model_path', type=str, default=None)
    # finetune 基础上再微调
    # py_parser.add_argument('--again_1__tinybert_ft_pre', action='store_true')
    # py_parser.add_argument('--again_2__tinybert_ft_hard', action='store_true')
    # py_parser.add_argument('--again_3__tinybert_ft_pre', action='store_true')
    # py_parser.add_argument('--again_3__tinybert_ft_hard', action='store_true')
    args, args_other = py_parser.parse_known_args()
    # read tune
    if args.show_tune:
        path = os.path.expanduser(args.show_tune)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf8') as r:
                max_output_L = json.load(r)
            show_max_output_L(max_output_L)
            print(args.show_tune)
            return max_output_L
    # 初始化
    if len(sys.argv) < 2:
        args.test = True
    print('args:', vars(args))
    args_again, args_other_ = [], []
    for k in args_other:
        if '=' in k:
            k, v = k.split('=')
        else:
            v = None
        if re.search(f'^--again_\d+__', k):
            args_again.append((k, v))
        else:
            args_other_.append((k, v))
    args_other = args_other_
    args_other_D = {k: v for k, v in args_other}
    print('args_other:', args_other)
    print('args_again:', args_again)
    # args 处理
    big_tasks = set(args.big_task.split(',')) if args.big_task_gpus and args.big_task else set()

    task_t_load = {  # --teacher_load_pretrained
        'base': {
            Tasks.copa: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/COPA/blank-base-COPA-220813_064749',
            Tasks.wsc_generative: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/WSC/blank-base-WSC_generative-220813_065345',
            Tasks.cb: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/CB/blank-base-CB-220813_065532',
            Tasks.rte: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/RTE/blank-base-RTE-220813_065724',
            Tasks.boolq: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/BoolQ/blank-base-BoolQ-220813_070712',
            Tasks.wic: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/WiC/blank-base-WiC-220813_072213',
            Tasks.multirc: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/MultiRC/blank-base-MultiRC-220813_074014',
            Tasks.wsc: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/WSC/blank-base-WSC-220813_073441',
            Tasks.record: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune/ReCoRD/blank-base-ReCoRD-220813_083745',
            Tasks.seq_cnndm_org: f'{ap}/checkpoints/pretrain/blocklm-base-blank/finetune_seq2seq/cnn_dm_original/blank-base-cnn_dm_original-220823_213924',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['base'][t]),
                ('--teacher_num_layers', '12'),
                ('--teacher_hidden_size', '768'),
                ('--teacher_num_attention_heads', '12'),
                ('--teacher_max_position_embeddings', '512'),
                ('--teacher_fp16', None),
            ]
        },
        'large': {
            Tasks.copa: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/COPA/blank-large-COPA-220813_123629',
            Tasks.wsc_generative: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/WSC/blank-large-WSC_generative-220813_125540',
            Tasks.cb: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/CB/blank-large-CB-220813_125843',
            Tasks.rte: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/RTE/blank-large-RTE-220813_130259',
            Tasks.boolq: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/BoolQ/blank-large-BoolQ-220813_133458',
            Tasks.wic: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/WiC/blank-large-WiC-220813_142454',
            Tasks.multirc: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/MultiRC/blank-large-MultiRC-220813_152437',
            Tasks.wsc: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/WSC/blank-large-WSC-220813_150605',
            Tasks.record: f'{ap}/checkpoints/pretrain/blocklm-large-blank/finetune/ReCoRD/blank-large-ReCoRD-220813_190003',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['large'][t]),
                ('--teacher_num_layers', '24'),
                ('--teacher_hidden_size', '1024'),
                ('--teacher_num_attention_heads', '16'),
                ('--teacher_max_position_embeddings', '512'),
            ]
        },
        'base_large': {
            'args': lambda t: [
                ('--mt_load_pretrained', task_t_load['base'][t] + ':' + task_t_load['large'][t]),
                ('--mt_num_layers', '12:24'),
                ('--mt_hidden_size', '768:1024'),
                ('--mt_num_attention_heads', '12:16'),
                ('--mt_max_position_embeddings', '512:512'),
                ('--teacher_fp16', None),
            ]
        },
        'base_base_large': {  # 第一个base是预训练的,用于RL-KD
            'args': lambda t: [
                ('--mt_load_pretrained', f'{ap}/checkpoints/pretrain/blocklm-base-blank:' + task_t_load['base'][t] + ':' + task_t_load['large'][t]),
                ('--mt_num_layers', '12:12:24'),
                ('--mt_hidden_size', '768:768:1024'),
                ('--mt_num_attention_heads', '12:12:16'),
                ('--mt_max_position_embeddings', '512:512:512'),
                ('--teacher_fp16', None),
            ]
        },
        # distill24.1024-18.896_mix2 (logitsdistil) 200M
        'tune_221006_170005.145602': {
            Tasks.copa: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/COPA/blank-base-COPA-221008_084623.965766/ft_kd/COPA/blank-base-COPA-221008_084625.067243',
            Tasks.wsc_generative: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/WSC/blank-base-WSC_generative-221008_090932.817304/ft_kd/WSC/blank-base-WSC_generative-221008_090933.918785',
            Tasks.cb: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/CB/blank-base-CB-221008_091721.813896/ft_kd/CB/blank-base-CB-221008_091722.915296',
            Tasks.rte: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/RTE/blank-base-RTE-221008_092548.958127/ft_kd/RTE/blank-base-RTE-221008_092550.059740',
            Tasks.boolq: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/BoolQ/blank-base-BoolQ-221008_101743.557232/ft_kd/BoolQ/blank-base-BoolQ-221008_101744.658682',
            Tasks.wic: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/WiC/blank-base-WiC-221008_113434.837390/ft_kd/WiC/blank-base-WiC-221008_113435.938793',
            Tasks.multirc: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/MultiRC/blank-base-MultiRC-221008_131055.701624/ft_kd/MultiRC/blank-base-MultiRC-221008_131056.803034',
            Tasks.wsc: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/WSC/blank-base-WSC-221008_124751.748433/ft_kd/WSC/blank-base-WSC-221008_124752.849713',
            Tasks.record: f'{ap}/checkpoints/distill/tiny12/distill24.1024-18.896_mix2/ft_kd/ReCoRD/blank-base-ReCoRD-221006_170005.145677/ft_kd/ReCoRD/blank-base-ReCoRD-221006_170006.247077',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['tune_221006_170005.145602'][t]),
                ('--teacher_num_layers', '18'),
                ('--teacher_hidden_size', '896'),
                ('--teacher_num_attention_heads', '14'),
                ('--teacher_max_position_embeddings', '512'),
                ('--teacher_fp16', None),
            ]
        },
        # distill18.896-12.768_mix2 (logitsdistil) 110M
        'tune_221009_173050.702880': {
            Tasks.copa: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/COPA/blank-base-COPA-221010_185736.644660/ft_kd/COPA/blank-base-COPA-221010_185737.746157',
            Tasks.wsc_generative: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/WSC/blank-base-WSC_generative-221010_191608.607141/ft_kd/WSC/blank-base-WSC_generative-221010_191609.708584',
            Tasks.cb: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/CB/blank-base-CB-221010_192247.742439/ft_kd/CB/blank-base-CB-221010_192248.843763',
            Tasks.rte: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/RTE/blank-base-RTE-221010_193034.715783/ft_kd/RTE/blank-base-RTE-221010_193035.817385',
            Tasks.boolq: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/BoolQ/blank-base-BoolQ-221010_201256.221705/ft_kd/BoolQ/blank-base-BoolQ-221010_201257.323046',
            Tasks.wic: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/WiC/blank-base-WiC-221010_212113.481188/ft_kd/WiC/blank-base-WiC-221010_212114.582578',
            Tasks.multirc: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/MultiRC/blank-base-MultiRC-221010_222612.125780/ft_kd/MultiRC/blank-base-MultiRC-221010_222613.227076',
            Tasks.wsc: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/WSC/blank-base-WSC-221010_221305.055794/ft_kd/WSC/blank-base-WSC-221010_221306.157390',
            Tasks.record: f'{ap}/checkpoints/distill/tiny12/distill18.896-12.768_mix2/ft_kd/ReCoRD/blank-base-ReCoRD-221009_173050.702958/ft_kd/ReCoRD/blank-base-ReCoRD-221009_173051.808541',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['tune_221009_173050.702880'][t]),
                ('--teacher_num_layers', '12'),
                ('--teacher_hidden_size', '768'),
                ('--teacher_num_attention_heads', '12'),
                ('--teacher_max_position_embeddings', '512'),
                ('--teacher_fp16', None),
            ]
        },
        # distill12.768-6.768_mix2 (logitsdistil) 66M
        'tune_221011_101500.766969': {
            Tasks.copa: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/COPA/blank-tiny6-COPA-221012_001702.677294/ft_kd/COPA/blank-tiny6-COPA-221012_001703.778772',
            Tasks.wsc_generative: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/WSC/blank-tiny6-WSC_generative-221012_003106.315605/ft_kd/WSC/blank-tiny6-WSC_generative-221012_003107.416016',
            Tasks.cb: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/CB/blank-tiny6-CB-221012_003646.768917/ft_kd/CB/blank-tiny6-CB-221012_003647.870362',
            Tasks.rte: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/RTE/blank-tiny6-RTE-221012_004334.467392/ft_kd/RTE/blank-tiny6-RTE-221012_004335.568770',
            Tasks.boolq: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/BoolQ/blank-tiny6-BoolQ-221012_011411.535615/ft_kd/BoolQ/blank-tiny6-BoolQ-221012_011412.636984',
            Tasks.wic: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/WiC/blank-tiny6-WiC-221012_015951.235099/ft_kd/WiC/blank-tiny6-WiC-221012_015952.335994',
            Tasks.multirc: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/MultiRC/blank-tiny6-MultiRC-221012_024748.779231/ft_kd/MultiRC/blank-tiny6-MultiRC-221012_024749.879988',
            Tasks.wsc: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/WSC/blank-tiny6-WSC-221012_023841.651092/ft_kd/WSC/blank-tiny6-WSC-221012_023842.751981',
            Tasks.record: f'{ap}/checkpoints/distill/tiny6/distill12.768-6.768_mix2/ft_kd/ReCoRD/blank-tiny6-ReCoRD-221011_101500.767113/ft_kd/ReCoRD/blank-tiny6-ReCoRD-221011_101501.868044',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['tune_221011_101500.766969'][t]),
                ('--teacher_num_layers', '6'),
                ('--teacher_hidden_size', '768'),
                ('--teacher_num_attention_heads', '12'),
                ('--teacher_max_position_embeddings', '512'),
                ('--teacher_fp16', None),
            ]
        },
        # distill18.896-12.768-6.768_b (logitsdistil) 66M
        'tune_221012_143057.727192': {
            Tasks.copa: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/COPA/blank-tiny6-COPA-221013_081544.749969/ft_kd/COPA/blank-tiny6-COPA-221013_081545.851413',
            Tasks.wsc_generative: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/WSC/blank-tiny6-WSC_generative-221013_083034.373427/ft_kd/WSC/blank-tiny6-WSC_generative-221013_083035.474919',
            Tasks.cb: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/CB/blank-tiny6-CB-221013_083705.043801/ft_kd/CB/blank-tiny6-CB-221013_083706.145277',
            Tasks.rte: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/RTE/blank-tiny6-RTE-221013_084314.623689/ft_kd/RTE/blank-tiny6-RTE-221013_084315.725175',
            Tasks.boolq: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/BoolQ/blank-tiny6-BoolQ-221013_091628.327639/ft_kd/BoolQ/blank-tiny6-BoolQ-221013_091629.429409',
            Tasks.wic: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/WiC/blank-tiny6-WiC-221013_100514.802376/ft_kd/WiC/blank-tiny6-WiC-221013_100515.903949',
            Tasks.multirc: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/MultiRC/blank-tiny6-MultiRC-221013_105714.177239/ft_kd/MultiRC/blank-tiny6-MultiRC-221013_105715.278701',
            Tasks.wsc: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/WSC/blank-tiny6-WSC-221013_104706.918815/ft_kd/WSC/blank-tiny6-WSC-221013_104708.020043',
            Tasks.record: f'{ap}/checkpoints/distill/tiny6/distill18.896-12.768-6.768_b/ft_kd/ReCoRD/blank-tiny6-ReCoRD-221012_143057.727276/ft_kd/ReCoRD/blank-tiny6-ReCoRD-221012_143058.828645',
            'args': lambda t: [
                ('--teacher_load_pretrained', task_t_load['tune_221012_143057.727192'][t]),
                ('--teacher_num_layers', '6'),
                ('--teacher_hidden_size', '768'),
                ('--teacher_num_attention_heads', '12'),
                ('--teacher_max_position_embeddings', '512'),
                ('--teacher_fp16', None),
            ]
        },
        # 用于 distill24.1024_18.896_12.768-6.768_avg1 (logitsdistil)
        'distill24.1024_18.896_12.768-6.768_avg1': {
            'args': lambda t: [
                ('--mt_load_pretrained', ':'.join([
                    task_t_load['tune_221009_173050.702880'][t], 
                    task_t_load['tune_221006_170005.145602'][t], 
                    task_t_load['large'][t]])),
                ('--mt_num_layers', '12:18:24'),
                ('--mt_hidden_size', '768:896:1024'),
                ('--mt_num_attention_heads', '12:14:16'),
                ('--mt_max_position_embeddings', '512:512:512'),
                ('--teacher_fp16', None),
            ]
        },
        # 用于 distill24.1024_18.896_12.768-6.768_rl_kd
        'distill24.1024_18.896_12.768-6.768_rl_kd': {
            'args': lambda t: [
                ('--mt_load_pretrained', ':'.join([
                    f'{ap}/checkpoints/pretrain/blocklm-base-blank',
                    task_t_load['tune_221009_173050.702880'][t], 
                    task_t_load['tune_221006_170005.145602'][t], 
                    task_t_load['large'][t]])),
                ('--mt_num_layers', '12:12:18:24'),
                ('--mt_hidden_size', '768:768:896:1024'),
                ('--mt_num_attention_heads', '12:12:14:16'),
                ('--mt_max_position_embeddings', '512:512:512:512'),
                ('--teacher_fp16', None),
            ]
        },
    }

    max_output_L = []
    max_output_path = f"{ap}/tmp/tune_{datetime.now().strftime('%y%m%d_%H%M%S.%f')}.json"
    ensure_directory_exists(max_output_path)
    print(str(datetime.now()), 'max_output_path:', max_output_path, '\n')
    custom_tmp_result_f = lambda: f"{ap}/tmp/result_{datetime.now().strftime('%y%m%d_%H%M%S.%f')}.json"
    Tasks.EPOCH_SINGLE = getattr(Tasks, args.epoch_type)
    rate_ds_D = {k[5:]: v for k, v in vars(args).items() if k[:8] == 'rate_ds_'}  # ds_ 倍率
    rate_arg_D = {'--' + k[9:]: v for k, v in vars(args).items() if k[:9] == 'rate_arg_'}  # arg_ 倍率
    for tni, (tn, task) in enumerate([(i, getattr(Tasks, i)) for i in args.tasks.split(',')]):
        task_model_path = getattr(args, f'{tn}_model_path') if hasattr(args, f'{tn}_model_path') else None
        if task_model_path:
            task_model_path = os.path.join(ap, task_model_path)
        elif args.model_path:
            task_model_path = os.path.join(ap, args.model_path)
        else:
            task_model_path = None
        if '--student_model' in args_other_D and args.save_sub is None:
            save_sub = f'ft_{args_other_D["--student_model"]}'
        else:
            save_sub = args.save_sub
        create_cmd_paras = {
            'script': getattr(Scripts, args.script),
            'model': getattr(Models, args.model),
            'task': task,
            'ds': True,
            'gpus': args.big_task_gpus if tn in big_tasks else args.gpus,
            'py_file': args.py_file,
            'env': {
                'MODEL_PATH': task_model_path,
                'MASK_RATIO': args.mask_ratio,
                **({'deepspeed_config_suffix': args.big_task_dsc_suffix} if tn in big_tasks else {})
            },
            'save_sub': save_sub,
        }
        py_args_L = []
        py_args = create_cmd(**create_cmd_paras) + args_other
        if args.task_t_load:
            print('distill')
            teacher_args = task_t_load[args.task_t_load]['args'](task)
            py_args_L.append(py_args + teacher_args)
            create_cmd_paras['env']['MODEL_PATH'] = os.path.join(OrderedDict(py_args)['--save'], OrderedDict(py_args)['--experiment-name'])
            for again in [f'again_{i}__' for i in range(10)]:
                again_args = [('--' + k.split('__', 1)[1], v) for k, v in args_again if re.search(f'^--{again}', k)]
                if len(again_args) == 0:
                    continue
                print('distill:', again)
                # args_other 出现和 again_args 相同参数则同时删除这两个参数
                same_S = set(args_other) & set(again_args)
                args_other_ = [i for i in args_other if i not in same_S]
                again_args = [i for i in again_args if i not in same_S]
                for same in same_S:
                    print('del', same)
                # 修改 teacher_args
                teacher_args_ = teacher_args
                again_args, again_args_ = [], again_args
                for k, v in again_args_:
                    if k == '--task_t_load':
                        teacher_args_ = task_t_load[v]['args'](task)
                    else:
                        again_args.append((k, v))
                time.sleep(1.1)
                py_args_pre = create_cmd(**create_cmd_paras) + teacher_args_ + args_other_ + again_args + [('--custom_first_eval', None)]
                py_args_L.append(py_args_pre)
            print()
        else:
            py_args_L.append(py_args)
        # cmds 处理
        cmds = []
        for py_args in py_args_L:
            for i, (k, v) in enumerate(py_args):
                if k in {'--deepspeed_config'} and v and os.path.exists(v):
                    with open(v, 'r', encoding='utf8') as r:
                        deepspeed_config = json.load(r)
                    restructure = False
                    for dsk, dsv in vars(args).items():
                        if dsk[:3] != 'ds_' or dsv is None:
                            continue
                        key = dsk[3:].split('__')
                        origin_dsv = get(key, deepspeed_config)
                        if ';' in dsv:
                            assert len(dsv.split(';')) == len(args.tasks.split(',')), '任务和ds_数量不对应'
                            dsv = dsv.split(';')[tni]
                        dsv = type(origin_dsv)(dsv)
                        if dsk in rate_ds_D:
                            dsv *= rate_ds_D[dsk]
                            dsv = type(origin_dsv)(dsv)
                        if origin_dsv == dsv:
                            continue
                        print(f'deepspeed_config: {key}: {origin_dsv} → {put(key, deepspeed_config, dsv)}')
                        restructure = True
                    if restructure:
                        config_path = 'tmp_deepspeed_config'
                        if not os.path.exists(config_path):
                            os.mkdir(config_path)
                        config_path = os.path.join(config_path, generate_unique() + '.json')
                        py_args[i] = (py_args[i][0], config_path)
                        with open(config_path, 'w', encoding='utf8') as w:
                            json.dump(deepspeed_config, w, ensure_ascii=False, indent=2, sort_keys=True)
                if k in rate_arg_D:
                    if k in {'--epochs'}:
                        v_ = str(int(rate_arg_D[k] * int(v)))
                        if v != v_:
                            print(f'{k}: {v} → {v_}')
                            py_args[i] = (py_args[i][0], v_)
            cmds.append(py_args_to_line(py_args))
        # 运行与捕获
        for cmd in cmds:
            custom_tmp_result = custom_tmp_result_f()
            cmd += ' --custom_tmp_result=' + custom_tmp_result
            print(cmd, '\n')
            if not args.test:
                sleep_t = 30
                while True:
                    try:
                        os.system(cmd)
                        print(str(datetime.now()), 'max_output_path:', max_output_path)
                        with open(custom_tmp_result, 'r', encoding='utf8') as r:
                            max_output = json.load(r)
                        assert max_output['args']['epochs'] == max_output['epoch'] + 1 or max_output['epoch'] == -1,\
                            f"epoch未跑完:{max_output['epoch']},{sleep_t}秒后重新运行相同的命令!"
                    except:
                        traceback.print_exc()
                        print(f'错误, {sleep_t}秒后重新运行相同的命令!')
                        time.sleep(sleep_t)
                        continue
                    # 处理输出
                    max_output_L.append(max_output)
                    max_output_L[-1]['*cmd'] = cmd
                    with open(max_output_path, 'w', encoding='utf8') as w:
                        json.dump(max_output_L, w, ensure_ascii=False, indent=2)
                    break
    # output
    show_max_output_L(max_output_L)
    print(str(datetime.now()), 'max_output_path:', max_output_path)
    return max_output_L


def show_max_output_L(max_output_L):
    metrics = ['f1a', 'f1', 'f1-macro', 'em', 'acc', 'accuracy', 'ppl', 'rouge-1', 'rouge-2', 'rouge-l']
    all_lines_ = OrderedDict([
        ('task', []),
        ('metric', []),
        ('result', []),
        ('epoch', []),
        ('all_epoch', []),
        ('final_epoch', []),
    ])  # 结果放在每一行
    all_lines_L = []  # 每组任务一个
    same_task_num = 0  # 相似的任务出现几次
    last_task = None  # 上一次任务是什么
    for max_output in max_output_L:
        print(max_output['*cmd'])
        print(max_output['args']['save'])
        max_value = -1
        max_score_dict_ = {}
        score_dict_ = {}
        for max_score_dict in max_output['max_score_dict'].values():
            score_dict = {k.lower(): v for k, v in max_score_dict['score_dict'].items() if k.lower() in metrics}
            n = 0
            for m in metrics:
                if m in score_dict:
                    n += 1
                    if n > (2 if m[:5] != 'rouge' else 3):
                        del score_dict[m]
            s = sum(score_dict.values())
            if s > max_value:
                max_value = s
                max_score_dict_ = max_score_dict
                score_dict_ = score_dict
        out_name = []
        out = []
        for m in metrics:
            if m in score_dict_:
                out.append(score_dict_[m])
                out_name.append(m)
        # 尝试分段对应
        if last_task == (max_output['args']['task'], max_output['args']['wsc_negative']):
            same_task_num += 1
        else:
            last_task = (max_output['args']['task'], max_output['args']['wsc_negative'])
            same_task_num = 0
        if same_task_num > len(all_lines_L) - 1:
            all_lines_L.append(copy.deepcopy(all_lines_))
        all_lines = all_lines_L[same_task_num]
        all_lines['task'] += len(out) * [max_output['args']['task']]
        all_lines['metric'] += out_name
        all_lines['result'] += out
        all_lines['epoch'] += len(out) * [max_score_dict_['epoch']]
        all_lines['all_epoch'] += len(out) * [max_output['args']['epochs']]
        all_lines['final_epoch'] += len(out) * [max_output['epoch']]
        # 输出处理
        out_name += ['epoch', 'all_epoch', 'final_epoch']
        out += [max_score_dict_['epoch'], max_output['args']['epochs'], max_output['epoch']]
        if max_output['args']['epochs'] != max_output['epoch'] + 1 and max_output['epoch'] != -1:
            print('error!')
        print('\t', '\t'.join(out_name))
        print('*1\t', '\t'.join([str(i) for i in out]))
        print('*100\t', '\t'.join([str(i * 100 if i is not None and 0 < i <= 1 else i) for i in out]))
    # 尝试分段对应
    for seg_i, all_lines in enumerate(all_lines_L):
        print()
        print(f'第{seg_i}组整合结果到一行:')
        for k, v in all_lines.items():
            if k == 'result':
                print('*1\t', '\t'.join([str(i) for i in v]))
                print('*100\t', '\t'.join([str(i * 100 if i is not None and 0 < i <= 1 else i) for i in v]))
            else:
                print(f'{k}\t', '\t'.join([str(i) for i in v]))


if __name__ == '__main__':
    auto_tune()
    if len(sys.argv) < 2:
        model = model_pre = task = None
        print()
        for script, model, model_pre, task, ds in [
            # ('finetune_superglue', 'block_tiny6', '', 'copa', False),
            # ('pretrain_nvidia', '', 'block_base', '', True),
            # ('finetune_seq2seq', 'model_blocklm_base', '', 'seq_cnndm_org', True),
        ] + [  # 模版
            *[('finetune_superglue', 'model_blocklm_10B', '', i, True) for i in 'copa,wsc_generative,cb,rte,boolq,wic,wsc,multirc,record'.split(',')],
            *[('finetune_seq2seq', 'model_blocklm_10B', '', i, True) for i in 'seq_cnndm_org,seq_xsum,seq_gigaword'.split(',')],
        ]:
            print('###' + f' task: {task}, model: {model}, model_pre: {model_pre}, script: {script}, ds: {ds}')
            if model in {'model_blocklm_10B'}:
                Tasks.EPOCH_SINGLE = Tasks.XXLARGE_EPOCH
            else:
                Tasks.EPOCH_SINGLE = Tasks.EPOCH_SINGLE_
            py_args = create_cmd(
                getattr(Scripts, script), 
                getattr(Models, model) if model else None, 
                getattr(Models_pre, model_pre) if model_pre else None, 
                getattr(Tasks, task) if task else None, 
                ds=ds)
            print(py_args_to_line(py_args))
            print()
