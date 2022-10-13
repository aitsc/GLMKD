import sys, os
sys.path.append(os.getcwd())

import data_utils
from tqdm import tqdm
from collections import Counter
from data_utils.tokenization import make_tokenizer
import json
from tasks.superglue.dataset import PROCESSORS
from arguments import get_args
import numpy as np
from tasks.superglue.pvp import PVPS
from tensorboardX import SummaryWriter
import shutil


args = get_args()
args.tokenizer_type = 'BertWordPieceTokenizer'
args.tokenizer_model_type = 'bert-base-uncased'
args.corpora_path = '../GLM/data/pretrain/bertbase/wikibook_glm.txt'
args.superglue_path = '../GLM/data/english_data/superglue'
args.draw_path = f'../GLM/data/tensorboard/draw/{args.tokenizer_type}_{args.tokenizer_model_type}'
args.save_vocab_path = f'.pytorch_pretrained_bert/{args.tokenizer_type}_{args.tokenizer_model_type}.txt'

# tokenizer
tokenizer = make_tokenizer(args.tokenizer_type, None, args.tokenizer_path, args.vocab_size,
                            args.tokenizer_model_type, add_block_symbols=True, cache_dir=None,
                            add_sentinel_token=0, add_task_mask=False,
                            add_decoder_mask=False,
                            fix_command_token=False)
save_token_id = {i.Id for i in tokenizer._command_tokens}
pvp_token = set()
for pvp in PVPS.values():  # 获取 VERBALIZER
    for v in ['VERBALIZER', 'VERBALIZER_A', 'VERBALIZER_B']:
        if not hasattr(pvp, v):
            continue
        bar = getattr(pvp, v)
        if isinstance(getattr(pvp, v), dict):
            bar = bar.values()
        for tokens in bar:
            for token in tokens:
                pvp_token.add(token.strip())
                pvp_token.add(token.strip().lower())
print(f'pvp_token({len(pvp_token)}):', sorted(pvp_token))
pvp_token_no_id = set()  # 无法 tokenizer 的token词
for t in pvp_token:
    try:
        save_token_id.add(tokenizer.TokenToId(t))
    except:
        pvp_token_no_id.add(t)
print(f'pvp_token_no_id({len(pvp_token_no_id)}):', sorted(pvp_token_no_id))
print(f'这些token不能省略({len(save_token_id)}):', sorted(save_token_id))
unk_id = tokenizer.get_command('unk').Id
print('unk_id:', unk_id)

def get_pt_token_num(path=args.corpora_path):
    token_num_path = os.path.join(data_utils.lazy_loader.get_lazy_path(path), 'token_num.json')
    if os.path.exists(token_num_path):
        with open(token_num_path, 'r', encoding='utf8') as r:
            token_num = json.load(r)
    else:
        pre_tokenize = True  # 是否为token化之后的数据
        map_fn = (lambda x: x.tolist()) if pre_tokenize else None
        text = data_utils.LazyLoader(
            path,
            data_type='text', 
            map_fn=map_fn, 
            mem_map=True,
            is_array=pre_tokenize,
            load_memory=True,  # 是否全部加载到内存
            half_load=False,  # 是否只加载一半
        )
        print('数据集token数量:', len(text.file))
        token_num = {str(k): int(v) for k, v in Counter(text.file).items()}
        with open(token_num_path, 'w', encoding='utf8') as w:
            json.dump(token_num, w)
    print('数据集token种数:', len(token_num))
    return token_num  # {'token':num,..}

# 预训练语料
token_num = get_pt_token_num()  # {'token_id':num,..}
[token_num.setdefault(str(i), 0) for i in range(tokenizer.num_tokens)]  # 填充0次出现的token
token_num_sort = sorted(token_num.items(), key=lambda t:t[1], reverse=True)  # 排序
save_token_id_L = [str(i[0]) for i in token_num_sort if int(i[0]) in save_token_id]  # ['token_id',..]
other_token_id_L = [str(i[0]) for i in token_num_sort if int(i[0]) not in save_token_id]
sum_num = sum(token_num.values())
# 写入文件
head = [
    ('origin_no', '原顺序这一行id应该是哪一行,从0开始'),
    ('sort_id', '排序后的id'),
    ('token', 'sort_id 对应的 token'),
    ('number', '语料中出现的次数'),
    ('frequency', '语料中出现的频率'),
]
with open(args.save_vocab_path, 'w', encoding='utf8') as w:
    w.write('\t'.join([i[0] for i in head]) + '\n')
    id_no_D = {v: k for k, v in enumerate(save_token_id_L + other_token_id_L)}  # {'token_id':序号,..}
    for i, id in enumerate(save_token_id_L + other_token_id_L):
        out = [str(id_no_D[str(i)])]  # origin_no
        out += [id, tokenizer.IdToToken(int(id)), str(token_num[id]), '%.3e' % (token_num[id] / sum_num)]
        w.write('\t'.join(out) + '\n')


def get_ft_token_num(data_dir, processor, tokenizer, args, split='train', token_num_sort=None):
    # 获取微调语料的数据
    token_num_path = os.path.join(data_dir, f'{split}_token_num.json')
    if os.path.exists(token_num_path):
        with open(token_num_path, 'r', encoding='utf8') as r:
            return json.load(r)
    if split == 'train':
        examples = processor.get_train_examples(data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(data_dir)
    elif split == 'test':
        examples = processor.get_test_examples(data_dir)
    sample = [processor.encode(i, tokenizer, args.seq_length, args) for i in examples]
    tokens = np.concatenate([i['text'].flatten() for i in sample], axis=0)
    c_id = {i.Id for i in tokenizer._command_tokens}  #  去除 command_tokens
    token_num_ = {str(k): 0 if int(k) in c_id else int(v) for k, v in Counter(tokens).items()}
    if token_num_sort:
        token_num_ = [(k, token_num_[k] if k in token_num_ else 0) for k, v in token_num_sort]
    with open(token_num_path, 'w', encoding='utf8') as w:
        json.dump(token_num_, w)
    return token_num_  # {'token':num,..} or [('token',num),..]


def draw_superglue(task_path=args.superglue_path):
    y_num_L = [[t[1] for t in token_num_sort]]  # [[num,..],..]
    x_token = [t[0] for t in token_num_sort]  # ['token_id',..]
    x_no_token = {v: k for k, v in enumerate(x_token)}  # {'token_id':序号,..}
    ylabel_L = ['Pretrain']  #  ['name',..]
    args.seq_length = args.max_position_embeddings = 512
    # 微调语料
    for task in tqdm(['WiC', 'RTE', 'CB', 'WSC', 'BoolQ', 'COPA', 'MultiRC', 'ReCoRD'], '微调数据'):
        processor = PROCESSORS[task.lower()](args)
        data_dir = os.path.join(task_path, task)
        for split in ['train', 'dev', 'test']:
            token_num_ = get_ft_token_num(data_dir, processor, tokenizer, args, split=split, token_num_sort=token_num_sort)
            y_num_L.append([t[1] for t in token_num_])
            ylabel_L.append(task + f'_{split}')
    # 预训练 + 微调
    for ii, (y_num, ylabel) in tqdm(enumerate(zip(y_num_L, ylabel_L)), 'writer'):
        y_num = np.array(y_num)  # 数量
        y_num_norm = y_num / y_num.sum()  # 归一化, 百分比
        y_num_cumsum = np.cumsum(y_num, axis=-1)  # 数量累计
        y_num_norm_cumsum = np.cumsum(y_num_norm, axis=-1)  # 归一化累计
        y_num_save_tokens = y_num[..., [x_no_token[i] for i in save_token_id_L]]  # 保留词的数量
        if ylabel == 'Pretrain':
            log_dir=f'{args.draw_path}/{ylabel}'
        else:
            log_dir=f'{args.draw_path}/superglue/{ylabel}'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir, purge_step=0)
        [writer.add_scalar('y/num', j, i) for i, j in enumerate(y_num)]
        [writer.add_scalar('y/num_norm', j, i) for i, j in enumerate(y_num_norm)]
        [writer.add_scalar('y/num_cumsum', j, i) for i, j in enumerate(y_num_cumsum)]
        [writer.add_scalar('y/num_norm_cumsum', j, i) for i, j in enumerate(y_num_norm_cumsum)]
        [writer.add_scalar('y/num_save_tokens', j, i) for i, j in enumerate(y_num_save_tokens)]
        if ii == 0:
            [writer.add_text('x/token', f'{j} {tokenizer.IdToToken(int(j))}', i) for i, j in enumerate(x_token)]
            [writer.add_text('x/token_save', f'{j} {tokenizer.IdToToken(int(j))}', i) for i, j in enumerate(save_token_id_L)]

# tensorboard --port 9726 --logdir ../GLM/data/tensorboard/draw
draw_superglue()
