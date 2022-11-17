import sys, os
sys.path.append(os.getcwd())

import json
from tasks.data_utils import build_data_loader, FakeDataloader
from utils import get_sample_writer, get_log_dir, print_and_save_args
from distill.prepare import get_args, get_teacher_model, get_teachers_hook, mt_repeat_operation, glm_wrap, mt_model_load, NoneWith, truncate_teacher_as_student
from filelock import FileLock
import pretrain_glm
from pretrain_glm import initialize_distributed, set_random_seed, get_batch
import pathlib
import mpu

import torch
import torch.utils.data
from configure_data import prepare_tokenizer
from distill.distill_model import unpacking_student_model
from distill.tools import distill_random_data

from utils import print_rank_0
from utils import Timers
from train_utils import setup_model_and_optimizer, load_pretrained, backward_step
from utils import load_checkpoint, save_checkpoint
from configure_data import make_data_loader
from finetune_glm import _train, _build_train_valid_dataloaders, process_batch, mix_forward_step
from mpu import hook_model
from tsc_base import merge_dict

tokenizer = None


def finetune_forward_step(batch, model, args, timers, mems, teacher_models=None, is_eval=False, optimizer=None):
    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    # COPA: {'text': [16, 2, 256], 'label': [16], 'mask': [16, 2], 'position': [16, 2, 2, 256], 'target': [16, 2, 256], 'logit_mask': [16, 2, 256]}
    data = process_batch(batch_, args)
    timers('batch generator').stop()
    
    shuffle_objs = [data['text']] + ([data['dec_text']] if 'dec_text' in data else [])
    ret = distill_random_data(args, shuffle_objs, [data], 0, cancel=is_eval)
    tokens, data_ = ret[0], ret[1][0]
    data_['text'] = tokens[0]
    if len(tokens) > 1:
        data_['dec_text'] = tokens[1]

    repeat_f = lambda data_: finetune_forward_step_(
        data_, batch, model, args, timers, mems, teacher_models=teacher_models,
    )
    args.forward_repeat_current_n = 0
    loss, mems = repeat_f(data_)[:2]
    if args.forward_repeat_num and args.ignore_first_backward_gard:
        assert args.gradient_accumulation_steps == 1
        timers('sub_backward').start()
        backward_step(optimizer, model, loss, args, timers)
        timers('sub_backward').stop()
        loss = 0.
        
    if args.forward_repeat_num and not is_eval:
        for i in range(args.forward_repeat_num):
            args.forward_repeat_current_n = i + 1
            ret = distill_random_data(args, shuffle_objs, [data], i + 1)
            tokens, data_ = ret[0], ret[1][0]
            data_['text'] = tokens[0]
            if len(tokens) > 1:
                data_['dec_text'] = tokens[1]
            loss = loss + repeat_f(data_)[0]
        args.forward_repeat_current_n = 0
    
    if args.forward_repeat_num and args.ignore_first_backward_gard:
        if args.deepspeed:
            model.zero_grad()
            model.optimizer.zero_grad()
        else:
            optimizer.zero_grad()
    return loss, mems, 'bert'


def finetune_forward_step_(data, batch, model, args, timers, mems, teacher_models=None):
    """Simple forward step with cross-entropy loss."""

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
    if args.pretrained_bert:
        tokens, types, labels, attention_mask = data['text'], data['types'], data['label'], data['padding_mask']
    elif args.cloze_eval:
        tokens, labels, position_ids = data['text'], data['label'], data['position']
        attention_mask = data['mask']
        if not args.fast_decode:
            target_ids, logit_mask = data['target'], data['logit_mask']
            m_in, m_kw = [tokens, position_ids, attention_mask, target_ids, logit_mask], {}
            if args.continuous_prompt:
                m_kw['prompt_pos'] = data["prompt_pos"]
        else:
            dec_input_ids, dec_position_ids, dec_attention_mask = data['dec_text'], data['dec_position'], data[
                'dec_mask']
            dec_target_ids, dec_logit_mask = data['dec_target'], data['dec_logit_mask']
    else:
        tokens, labels, position_ids, attention_mask = data['text'], data['label'], data['position'], data['mask']
        m_in = [tokens, position_ids, attention_mask]
        m_kw = {}
        
    def get_logits(h, i, m, h_op, no_grad=False):
        # 必须是 model 直出的结果, 防止和测试过程中的不一致
        if args.pretrained_bert:
            logits = model(tokens, token_type_ids=types, attention_mask=attention_mask, checkpoint_activations=True)
        elif args.cloze_eval:
            if not args.fast_decode:
                with torch.no_grad() if no_grad else NoneWith():
                    result = hook_model(h, i, m, *m_in, **m_kw, hook_op=h_op)
                if not args.multi_token:
                    logits, lm_logits, *mems = result
                else:
                    logits, *mems = result
            else:
                logits, *mems = model(tokens, position_ids, attention_mask, dec_input_ids, dec_position_ids,
                                    dec_attention_mask, dec_target_ids, dec_logit_mask)
        else:
            with torch.no_grad() if no_grad else NoneWith():
                logits, *mems = hook_model(h, i, m, *m_in, **m_kw, hook_op=h_op)
        return logits, mems
    logits, mems = get_logits(s_hook, s_inter_vars, model, s_hook_op)

    # loss
    def get_loss(logits):
        if args.adapet:
            batch_size, num_classes = logits.size()[:2]
            label_mask = torch.ones(batch_size, num_classes, device=logits.device)
            label_mask.scatter_(1, labels.unsqueeze(1), -1.0)
            if "loss_mask" in data:
                loss_mask = data["loss_mask"]
                label_mask = label_mask * loss_mask
            loss_batch = logits.contiguous().float() * label_mask
            loss = loss_batch.sum() / batch_size
        else:
            if "segment_id" in data:
                from torch_scatter import scatter_sum
                if "loss_mask" in data:
                    logits = logits * data["loss_mask"]
                logits = scatter_sum(logits, data["segment_id"], dim=1)
            elif "loss_mask" in data:
                loss_mask = data["loss_mask"]
                logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
            if args.loss_func == "cross_entropy":
                # Cross-entropy loss.
                loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                loss_batch = loss_func(logits.contiguous().float(), labels)
                loss = loss_batch.mean()
            elif args.loss_func == "hinge":
                correct_logits = logits[range(logits.size(0)), labels]
                hinge_loss = 1 + logits - correct_logits.unsqueeze(1)
                hinge_loss[hinge_loss < 0.0] = 0.0
                loss_batch = hinge_loss.sum(dim=1) - 1.0
                loss = loss_batch.mean()
            elif args.loss_func == "generative" or args.loss_func == "mix":
                batch_size = logits.size(0)
                loss = - logits[range(batch_size), labels]
                if args.loss_func == "mix":
                    loss_func = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss + loss_func(logits.contiguous().float(), labels)
                loss_batch = loss
                loss = loss.mean()
            else:
                raise NotImplementedError
        return loss, loss_batch
    loss, loss_batch = get_loss(logits)

    if is_distill:
        student_model.add_summary('Train/hard_loss', loss)
        t_out_L = mt_repeat_operation(
            zip(t_hook_L, t_inter_vars_L, teacher_models, t_hook_op_L),
            lambda h, i, m, h_op: get_logits(h, i, m, h_op, no_grad=not args.mt_has_grad),
            lambda ret: {'logits': ret[0], 'mems': ret[1]},
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
            labels = labels,
        )
    return loss, mems, 'bert'


def finetune(args, train_valid_datasets_provider, model_kwargs, forward_step=finetune_forward_step,
             end_of_epoch_callback_provider=None):
    """Main finetune function used across all tasks."""
    global tokenizer
    timers = Timers()
    tokenizer = prepare_tokenizer(args)
    pretrain_glm.tokenizer = tokenizer
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    train_dataloader, valid_dataloader = None, None
    train_block_dataloader, valid_block_dataloader = None, None
    if train_valid_datasets_provider is not None and args.epochs > 0:
        if mpu.get_model_parallel_rank() == 0:
            train_dataset, valid_dataset = train_valid_datasets_provider(args, tokenizer)
            train_dataloader, valid_dataloader = _build_train_valid_dataloaders(train_dataset, valid_dataset, args)
            if args.no_validation:
                valid_dataloader = None
            train_iters = torch.cuda.LongTensor([len(train_dataloader)])
        else:
            train_iters = torch.cuda.LongTensor([0])
        torch.distributed.broadcast(train_iters, mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        if mpu.get_model_parallel_rank() != 0:
            args.train_iters_per_epoch = train_iters[0].item()
            args.train_iters = args.epochs * args.train_iters_per_epoch

            train_dataloader = FakeDataloader(args.train_iters_per_epoch)
            if args.no_validation:
                valid_dataloader = None
            else:
                valid_dataloader = FakeDataloader(None)
        if args.block_lm_ratio > 0.0:
            if mpu.get_model_parallel_rank() == 0:
                train_block_dataset, valid_block_dataset = train_valid_datasets_provider(args, tokenizer,
                                                                                         pattern_text=True)
                train_block_dataloader = make_data_loader(train_block_dataset, tokenizer,
                                                          args.batch_size * mpu.get_data_parallel_world_size(),
                                                          args.train_iters, args, shuffle=True,
                                                          block_collate=True)
                valid_block_dataloader = make_data_loader(valid_block_dataset, tokenizer,
                                                          args.batch_size * mpu.get_data_parallel_world_size(), (
                                                                  args.train_iters // args.eval_interval + 1) * args.eval_iters,
                                                          args, shuffle=True, block_collate=True)
            else:
                train_block_dataloader = FakeDataloader(args.train_iters)
                valid_block_dataloader = FakeDataloader(None)
            train_block_dataloader, valid_block_dataloader = iter(train_block_dataloader), iter(valid_block_dataloader)

    timers('train/valid/test dataset/dataloder').stop()
    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback, end_of_train_callback = None, None
    if end_of_epoch_callback_provider is not None:
        if train_valid_datasets_provider is not None and args.epochs > 0 and not args.no_validation:
            end_of_epoch_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=False)
        end_of_train_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=False)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    teacher_models=get_teacher_model(args, **model_kwargs)
    glm_wrap_ = lambda **k: glm_wrap(**k, teacher_models=teacher_models)
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, **model_kwargs, glm_wrap=glm_wrap_)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.load_pretrained and not args.pretrained_bert:
        task_tokens = None
        if args.continuous_prompt and args.prompt_init:
            if mpu.get_model_parallel_rank() == 0:
                dataset = train_dataloader.dataset
                processor, pvp = dataset.processor, dataset.pvp
                task_tokens = []
                for label in processor.get_labels():
                    verbalizer = pvp.verbalize(label)[0]
                    verbalizer_ids = tokenizer.EncodeAsIds(verbalizer).tokenization
                    task_tokens += verbalizer_ids
                print_rank_0("Task tokens: " + tokenizer.DecodeIds(task_tokens))
                num_task_tokens = len(task_tokens)
            else:
                num_task_tokens, task_tokens = 0, []
            num_task_tokens = torch.cuda.LongTensor([num_task_tokens])
            torch.distributed.broadcast(num_task_tokens, mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            num_task_tokens = num_task_tokens.item()
            if num_task_tokens > 0:
                if mpu.get_model_parallel_rank() == 0:
                    task_tokens = torch.cuda.LongTensor(task_tokens)
                else:
                    task_tokens = torch.empty(num_task_tokens, device=torch.cuda.current_device(), dtype=torch.long)
                torch.distributed.broadcast(task_tokens, mpu.get_model_parallel_src_rank(),
                                            group=mpu.get_model_parallel_group())
                task_tokens = task_tokens.tolist()
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            load_pretrained(model, args.load_pretrained, args, task_tokens=task_tokens)
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16 and optimizer is not None:
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    is_load1 = mt_model_load(model, args.mt_model_load)
    is_load2 = truncate_teacher_as_student(model, teacher_models, args)
    if (is_load1 or is_load2) and args.fp16 and optimizer is not None:
        if args.deepspeed:
            optimizer.refresh_fp32_params()
        else:
            optimizer._model_params_to_master_params()
    if args.load is not None:
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=args.no_deepspeed_load)
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16 and optimizer is not None:
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    torch.distributed.barrier()
    timers('pretrained checkpoint').stop()
    args.iteration = 0
    summary_writer = None
    if torch.distributed.get_rank() == 0:
        args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
        if os.path.exists(os.path.join(args.log_dir, "test_results.json")) and args.load is None and not args.overwrite:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.log_dir))
        summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration, args=args)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)
        student_model = unpacking_student_model(model)
        if student_model is not None:
            student_model.summary_writer = summary_writer

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    score_dict = None
    if train_dataloader is not None and args.epochs > 0:
        if args.block_lm_ratio > 0.0:
            forward_step = mix_forward_step
        best_iteration = _train(model, optimizer, lr_scheduler, forward_step,
                                (train_dataloader, train_block_dataloader), (valid_dataloader, valid_block_dataloader),
                                end_of_epoch_callback, args, timers,
                                summary_writer=summary_writer, teacher_models=teacher_models)
        if end_of_train_callback is not None and best_iteration is not None:
            with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
                args.load = os.path.join(args.save, "best")
                load_checkpoint(model, optimizer, lr_scheduler, args, no_load_optim=True, no_deepspeed=True)
                args.load = None
        torch.distributed.barrier()
        if end_of_train_callback is not None:
            score_dict = end_of_train_callback(model, epoch=-1, output_predictions=True)
    # Or just evaluate.
    else:
        if end_of_train_callback is not None:
            print_rank_0('evaluation only mode, setting epoch to -1')
            score_dict = end_of_train_callback(model, epoch=-1, output_predictions=True)
    if score_dict is not None and torch.distributed.get_rank() == 0:
        score_dict.update({"type": "test"})
        with open(os.path.join(args.log_dir, "test_results.json"), "w") as output:
            output.write(json.dumps(score_dict) + "\n")

    print_rank_0('done :-)')


if __name__ == '__main__':
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    assert args.finetune

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    from tasks.superglue.dataset import PROCESSORS

    superglue_tasks = list(PROCESSORS.keys())
    if args.task.lower() in superglue_tasks:
        from tasks.superglue.finetune import main
    elif args.task.lower() in ['lambda', 'wikitext', 'language_model']:
        from tasks.language_model.finetune import main
    elif args.task.lower() in ['cnn_dm', 'cnn_dm_original', 'gigaword', 'blank', 'squad_generation', 'squad',
                               'squad_v1', 'xsum', 'extraction', 'cmrc']:
        from tasks.seq2seq.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    main(args, finetune)
