import sys, os
sys.path.append(os.getcwd())

from datetime import datetime
import random

import torch.distributed
from filelock import FileLock
import torch

from contextlib import ExitStack
from configure_data import configure_data, prepare_tokenizer, build_multi_task_dataset
import mpu
import pathlib
from distill.prepare import get_args, get_teacher_model, get_teachers_hook, mt_repeat_operation, glm_wrap, mt_model_load, NoneWith, truncate_teacher_as_student, build_map_vocab_for_student

from train_utils import setup_model_and_optimizer, load_pretrained
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_and_save_args
from utils import print_rank_0, get_distributed_formatted_time
from utils import get_sample_writer, get_log_dir, get_hostname
import torch.distributed as dist
from pretrain_glm import get_batch, evaluate_and_print_results, initialize_distributed, set_random_seed, get_train_val_test_data, train
from distill.distill_model import unpacking_student_model
from distill.tools import distill_random_data
from mpu import hook_model
from tsc_base import merge_dict

tokenizer = None


def forward_step(data_iterator, model, args, timers, mems, teacher_models=None, is_eval=False):
    # Get the batch.
    timers('batch generator').start()
    timers('data loader').start()
    rand = random.Random(args.iteration * mpu.get_data_parallel_world_size() + mpu.get_data_parallel_rank())
    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    # print_rank_0("data iterator")
    timers('data loader').stop()
    # (16,561) (16,561) (16,561) (16) (16,2,561)
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    timers('batch generator').stop()

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'

    repeat_f = lambda data_: forward_step_(
        *data_,
        data_iterator, model, args, timers, mems, teacher_models=teacher_models,
    )
    ret = distill_random_data(args, [tokens], [labels, loss_mask, attention_mask, position_ids], 0, cancel=is_eval)
    args.forward_repeat_current_n = 0
    loss, mems = repeat_f(ret[0] + ret[1])[:2]
        
    if args.forward_repeat_num and not is_eval:
        for i in range(args.forward_repeat_num):
            args.forward_repeat_current_n = i + 1
            ret = distill_random_data(args, [tokens], [labels, loss_mask, attention_mask, position_ids], i + 1)
            loss = loss + repeat_f(ret[0] + ret[1])[0]
        args.forward_repeat_current_n = 0
    return loss, mems, mode


def forward_step_(tokens, labels, loss_mask, attention_mask, position_ids,
    data_iterator, model, args, timers, mems, teacher_models=None):
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
    logits, *mems = hook_model(s_hook, s_inter_vars, model, tokens, position_ids, attention_mask, *mems, hook_op=s_hook_op)

    def compute_loss(logits):
        losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
        loss_batch = (losses * loss_mask).sum(-1)
        loss = loss_batch.sum()
        if loss_mask.sum().item() > 0:
            loss = loss / loss_mask.sum()
            loss_batch = loss_batch / loss_mask.sum(-1)
        return loss, loss_batch
    loss, loss_batch = compute_loss(logits)

    if is_distill:
        student_model.add_summary('Train/hard_loss', loss)
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
                lambda logits, **k: compute_loss(logits),
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
    return loss, mems


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    args.mem_length = args.mem_length if args.transformer_xl else 0
    # Pytorch distributed.
    initialize_distributed(args)
    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + get_distributed_formatted_time()
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    global tokenizer
    tokenizer = prepare_tokenizer(args)
    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)
    multi_train_data, multi_val_data = None, None
    if args.multi_task_ratio > 0.0:
        multi_train_data, multi_val_data = build_multi_task_dataset(args, tokenizer)
    if train_data is not None and not args.custom_sample_shape:
        args.custom_sample_shape = ','.join((str(i) for i in train_data.dataset[0]['text'].shape))

    # Model, optimizer, and learning rate.
    args.custom_logits_paralle = True
    teacher_models=get_teacher_model(args)
    glm_wrap_ = lambda **k: glm_wrap(**k, teacher_models=teacher_models)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, glm_wrap=glm_wrap_)
    if args.load_pretrained:
        print_rank_0("> Load only pre-trained model parameters...")
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            load_pretrained(model, args.load_pretrained, args)
    is_load1 = mt_model_load(model, args.mt_model_load)
    is_load2 = truncate_teacher_as_student(model, teacher_models, args)
    build_map_vocab_for_student(model, teacher_models, args, tokenizer)
    if (is_load1 or is_load2 or args.load_pretrained) and args.fp16 and optimizer is not None:
        if args.deepspeed:
            optimizer.refresh_fp32_params()
        else:
            optimizer._model_params_to_master_params()
    if args.load is not None:
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        if args.no_load_optim and args.fp16 and optimizer is not None:
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    else:
        args.iteration = 0
    torch.distributed.barrier()
    if args.switch_linear:
        lr_scheduler.switch_linear(args)

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        args.log_dir = None
        if args.train_iters > 0:
            args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
            summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration, args=args)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)
        student_model = unpacking_student_model(model)
        if student_model is not None:
            student_model.summary_writer = summary_writer

    # Resume data loader if necessary.
    if args.resume_dataloader:
        print_rank_0("Resume dataloader")
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
        if val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        if multi_train_data is not None:
            multi_train_data.batch_sampler.start_iter = int(args.iteration * args.multi_task_ratio) % len(
                multi_train_data)
        if multi_val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters * args.multi_task_ratio
            multi_val_data.batch_sampler.start_iter = start_iter_val % len(multi_val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if multi_train_data is not None:
        multi_train_iterator = iter(multi_train_data)
    else:
        multi_train_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
    if multi_val_data is not None:
        multi_val_iterator = iter(multi_val_data)
    else:
        multi_val_iterator = None

    # TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)

                # stack.callback(save_on_exit, args, model, optimizer, lr_scheduler)
                iteration, skipped = train(model, optimizer,
                                           lr_scheduler,
                                           (train_data_iterator, multi_train_iterator),
                                           (val_data_iterator, multi_val_iterator),
                                           timers, args, summary_writer=summary_writer, teacher_models=teacher_models, forward_step_func=forward_step)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, (val_data_iterator, multi_val_iterator),
                                                  model, args, timers, verbose=False, forward_step_func=forward_step)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, (test_data_iterator, None),
                                   model, args, timers, verbose=True, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
