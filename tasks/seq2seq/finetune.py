# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Race."""
import torch
import mpu
import json
import functools
from tasks.eval_utils import accuracy_func_provider
from finetune_glm import finetune
from pretrain_glm import get_batch
from collections import OrderedDict
from tasks.seq2seq.dataset import Seq2SeqDataset, BlankLMDataset, ExtractionDataset
from tasks.seq2seq.evaluate import rouge_metric, DecoderEvaluater, BlankLMEvaluater
from tasks.superglue.evaluate import squad_exact_match, squad_f1
from mpu import hook_model
from distill.distill_model import unpacking_student_model
from tsc_base import merge_dict
from distill.prepare import get_teachers_hook, mt_repeat_operation, NoneWith
from distill.tools import distill_random_data
from train_utils import backward_step

global_tokenizer = None


def seq2seq_forward_step(data, model, args, timers, mems, teacher_models=None, is_eval=False, optimizer=None):
    # Get the batch.
    if timers is not None:
        timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    if timers is not None:
        timers('batch generator').stop()

    repeat_f = lambda data_: seq2seq_forward_step_(
        *data_,
        data, model, args, timers, mems, teacher_models=teacher_models,
    )
    ret = distill_random_data(args, [tokens], [labels, loss_mask, attention_mask, position_ids], 0, cancel=is_eval)
    args.forward_repeat_current_n = 0
    loss, mems = repeat_f(ret[0] + ret[1])[:2]
    if args.forward_repeat_num and args.ignore_first_backward_gard:
        assert args.gradient_accumulation_steps == 1
        timers('sub_backward').start()
        backward_step(optimizer, model, loss, args, timers)
        timers('sub_backward').stop()
        loss = 0.
        
    if args.forward_repeat_num and not is_eval:
        for i in range(args.forward_repeat_num):
            args.forward_repeat_current_n = i + 1
            ret = distill_random_data(args, [tokens], [labels, loss_mask, attention_mask, position_ids], i + 1)
            loss = loss + repeat_f(ret[0] + ret[1])[0]
        args.forward_repeat_current_n = 0
    
    if args.forward_repeat_num and args.ignore_first_backward_gard:
        if args.deepspeed:
            model.zero_grad()
            model.optimizer.zero_grad()
        else:
            optimizer.zero_grad()
    return loss, mems, 'bert'


def seq2seq_forward_step_(tokens, labels, loss_mask, attention_mask, position_ids,
    data, model, args, timers, mems, teacher_models=None):
    """Forward step."""

    is_distill = teacher_models is not None and len(teacher_models) > 0
    student_model = unpacking_student_model(model)
    s_inter_vars, t_inter_vars_L = [], []
    if is_distill:
        s_hook = student_model.get_student_hook()
        t_hook_L = get_teachers_hook(args, student_model)
        t_inter_vars_L = [[] for _ in range(len(t_hook_L))]
        s_hook_op = student_model.get_student_hook_op(teacher_models=teacher_models)
        t_hook_op_L = get_teachers_hook(args, student_model, is_op=True, teacher_models=teacher_models)
    else:
        t_hook_L = s_hook = t_hook_op_L = s_hook_op = None
        teacher_models = []

    # Forward model.
    logits, *mems = hook_model(s_hook, s_inter_vars, model, tokens, position_ids, attention_mask, *mems, hook_op=s_hook_op)
    # loss
    def get_loss(logits):
        # logits, loss_mask = logits[:, args.src_seq_length:], loss_mask[:, args.src_seq_length:]
        # target_ids = target_ids[:, args.src_seq_length:]
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
        if args.label_smoothing > 0.0:
            epsilon = args.label_smoothing
            smooth_loss = -torch.nn.functional.log_softmax(logits, dim=-1).mean(dim=-1)
            losses = (1 - epsilon) * losses + epsilon * smooth_loss
        # The loss is not normalized for fair comparison
        loss_batch = (losses * loss_mask).sum(-1)
        loss = loss_batch.sum() / loss_mask.sum()
        loss_batch = loss_batch / loss_mask.sum(-1)
        return loss, loss_batch
    loss, loss_batch = get_loss(logits)

    if is_distill:
        with NoneWith() if args.mt_has_grad else torch.no_grad():
            t_out_L = mt_repeat_operation(
                zip(t_hook_L, t_inter_vars_L, teacher_models, t_hook_op_L),
                lambda h, i, m, h_op: hook_model(h, i, m, tokens, position_ids, attention_mask, *mems, hook_op=h_op),
                lambda ret: {'logits': ret[0], 'mems': ret[1:]},
                [int(i) for i in args.mt_disable_operation.split(':')],
                [{'logits': 0., 'mems': 0.}],
            )
        if args.mt_has_loss:
            t_out_L = [merge_dict([i, j]) for i, j in zip(mt_repeat_operation(
                t_out_L,
                lambda logits, **k: get_loss(logits),
                lambda ret: {'loss': ret[0], 'loss_batch': ret[1]},
                [int(i) for i in args.mt_disable_operation.split(':')],
                [{'loss': 0., 'loss_batch': 0.}],
            ), t_out_L)]
        loss = student_model.multi_teacher_model.compute(
            teacher_models = teacher_models,
            t_hook_L = t_hook_L,
            t_inter_vars_L = t_inter_vars_L,
            t_out_L = t_out_L,
            student_model = student_model,
            s_hook = s_hook,
            s_inter_vars = s_inter_vars,
            s_out = {'logits': logits, 'loss': loss, 'loss_batch': loss_batch},
            loss_mask = loss_mask,
            labels = labels,
        )
    return loss, mems, 'bert'


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    if args.task.lower() == 'blank':
        train_dataset = BlankLMDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    elif args.task.lower() == 'extraction':
        train_dataset = ExtractionDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    else:
        train_dataset = Seq2SeqDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    global global_tokenizer
    global_tokenizer = tokenizer
    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Provide metrics callback function."""

    def single_dataset_provider(split):
        if args.task.lower() == 'blank':
            return BlankLMDataset(args, split=split, tokenizer=tokenizer)
        elif args.task.lower() == 'extraction':
            return ExtractionDataset(args, split=split, tokenizer=tokenizer)
        else:
            return Seq2SeqDataset(args, split=split, tokenizer=tokenizer)

    if args.task.lower() in ['blank', 'extraction']:
        evaluater = BlankLMEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        metric_dict = {}
    else:
        evaluater = DecoderEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        if args.tokenizer_type == "BertWordPieceTokenizer":
            dataset = 'cnn_dm'
        elif args.task.lower() == 'gigaword':
            dataset = 'gigaword'
        else:
            dataset = 'cnn_dm_org'
        if args.task.lower() in ['squad', 'squad_v1']:
            metric_dict = {"EM": squad_exact_match, "F1": squad_f1}
        else:
            metric_dict = OrderedDict({"rouge-1": functools.partial(rouge_metric, metric="rouge-1", dataset=dataset),
                                       "rouge-2": functools.partial(rouge_metric, metric="rouge-2", dataset=dataset),
                                       "rouge-l": functools.partial(rouge_metric, metric="rouge-l", dataset=dataset)})

    def output_func(predictions, examples, output_file):
        if args.task.lower() in ['squad', 'squad_v1']:
            with open(output_file, "w", encoding='utf-8') as output:
                res = {}
                for prediction, example in zip(predictions, examples):
                    idx = example.idx
                    if prediction.lower().replace(' ', '') == 'n/a':
                        prediction = ''
                    if idx not in res or res[idx] == '':
                        res[idx] = prediction
                json.dump(res, output)
            with open(output_file + ".refs", "w", encoding='utf-8') as output:
                for prediction, example in zip(predictions, examples):
                    res = {'id': example.idx, 'pred': prediction, 'gold': example.meta['answers']}
                    output.write(json.dumps(res) + '\n')
            return
        with open(output_file + ".hyps", "w", encoding='utf-8') as output:
            for prediction in predictions:
                output.write(prediction)
                output.write("\n")
        with open(output_file + ".refs", "w", encoding='utf-8') as output:
            for example in examples:
                output.write(example.meta["ref"])
                output.write("\n")
        if args.task.lower() == 'squad_generation':
            with open(output_file + ".source", "w", encoding='utf-8') as output:
                for example in examples:
                    output.write(example.text_a.replace("\n", " ") + " Answer: " + example.meta["answer"])
                    output.write("\n")

    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False)


def main(args, ft=finetune):
    if args.src_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.src_seq_length
    if args.task.lower() in ['cnn_dm', 'cnn_dm_original', 'gigaword', 'blank', 'squad_generation', 'xsum',
                             'squad', 'squad_v1', 'extraction', 'cmrc']:
        args.custom_logits_paralle = True
        ft(args, train_valid_datasets_provider, {}, end_of_epoch_callback_provider=metrics_func_provider,
                 forward_step=seq2seq_forward_step)
    else:
        raise NotImplementedError(args.task)
