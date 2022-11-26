# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""GPT-2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import mpu
from model.prompt import PromptSpell
from utils import print_rank_0
from mpu import hook_model, hook_return, hook_child, hook_add


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GLMModel(torch.nn.Module):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0,
                 map_vocab_size=None,
                 ib_hidden_size=None,
                 ib_mode=False,
                 ib_ffn_num=1,
                 ib_word_emb=None,
                 ):

        super(GLMModel, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.map_vocab_size = map_vocab_size

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        if ib_mode and ib_word_emb:
            self.ib_emb_conv = mpu.ColumnParallelLinear(ib_word_emb * 3, hidden_size,
                                                        gather_output=True,
                                                        init_method=init_method)
            self.ib_word_emb = ib_word_emb
        if map_vocab_size:
            self.word_embeddings = mpu.VocabParallelEmbedding(
                map_vocab_size, hidden_size, init_method=init_method)
            # 初始化映射, 无法直接使用, 需要外部赋值或加载已有数据
            self.map_vocab_paras = {
                # origin_id等价于origin_pos, 用途: 原始词表id映射到本地词表位置; 本地词表嵌入还原到原始词表嵌入顺序
                'origin_id_to_target_pos': torch.ones(vocab_size, dtype=torch.int64) * map_vocab_size,
                # True表示该词在词表中,False表示该词是映射词
                'origin_id_mask_map': torch.ones(vocab_size, dtype=torch.bool),
                # 位置 [map_vocab_size,vocab_size) 之间的token是映射的token
                'target_pos_to_origin_id': torch.ones(vocab_size, dtype=torch.int64) * map_vocab_size,
            }
        else:
            self.word_embeddings = mpu.VocabParallelEmbedding(
                vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       relative_encoding=relative_encoding,
                                                       block_position_encoding=block_position_encoding,
                                                       output_dim=ib_hidden_size if ib_mode else None,
                                                       ib_mode=ib_mode,
                                                       ib_ffn_num=ib_ffn_num)
        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None, hook=None, hook_op=None):
        inter_vars = []
        # Embeddings.
        if hook_op and 'input_ids' in hook_op and callable(hook_op['input_ids']):
            input_ids = hook_op['input_ids'](input_ids=input_ids)
        input_ids = self.map_input_ids(input_ids)
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        if hasattr(self, 'ib_emb_conv') and hasattr(self, 'ib_word_emb'):
            words_embeddings = words_embeddings[..., :self.ib_word_emb]
            words_embeddings = torch.cat([
                F.pad(words_embeddings[:, 1:], (0, 0, 0, 1, 0, 0)),
                words_embeddings,
                F.pad(words_embeddings[:, :-1], (0, 0, 1, 0, 0, 0))
            ], axis=2)
            words_embeddings = self.ib_emb_conv(words_embeddings)
        hook_add(hook, inter_vars, 'words_embeddings', words_embeddings)
        hook_add(hook, inter_vars, 'position_ids', position_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        transformer_output = hook_model(hook_child(hook, 'transformer'), inter_vars, self.transformer,
                                        embeddings, position_ids, attention_mask, mems,
                                        return_memory=return_memory, detach_memory=detach_memory,
                                        hook_op=hook_child(hook_op, 'transformer'))
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        if self.output_predict:
            # Parallel logits.
            logits_parallel = self.map_logits_linear(logits)
            hook_add(hook, inter_vars, 'logits_parallel', logits_parallel)

            if self.parallel_output:
                ret = (logits_parallel, *outputs)
            else:
                ret = (mpu.gather_from_model_parallel_region(logits_parallel), *outputs)
        else:
            ret = (logits, *outputs)
        return hook_return(hook, inter_vars, ret)
    
    def map_input_ids(self, input_ids):
        # 词表映射到转换的 word_embeddings
        if not self.map_vocab_size:
            return input_ids
        origin_id_to_target_pos = self.map_vocab_paras['origin_id_to_target_pos']
        return F.embedding(input_ids, origin_id_to_target_pos.unsqueeze(-1)).squeeze(-1)

    def map_logits_linear(self, logits):
        # 还原原始 word_embeddings 进行线性变换得到 logits_parallel
        logits_parallel = mpu.copy_to_model_parallel_region(logits)
        if self.map_vocab_size:
            origin_id_to_target_pos = self.map_vocab_paras['origin_id_to_target_pos']
            if mpu.get_model_parallel_world_size() == 1:
                weight = F.embedding(origin_id_to_target_pos, self.word_embeddings.weight)
                logits = F.linear(logits_parallel, weight)
                return logits
                # 切片会导致 backward 慢很多
                # logits = F.linear(logits_parallel, self.word_embeddings.weight)
                # return logits[..., origin_id_to_target_pos]
            else:
                origin_id_to_target_pos = mpu.scatter_to_model_parallel_region(origin_id_to_target_pos)
                weight = self.word_embeddings(origin_id_to_target_pos)
                return F.linear(logits_parallel, weight)
        else:
            return F.linear(logits_parallel, self.word_embeddings.weight)


class GLMModel_empty(torch.nn.Module):
    # 空模型用于占位
    def __init__(self, glm_model=None, vocab_size=None, hidden_size=None, parallel_output=True, output_predict=True):
        super().__init__()
        if glm_model is not None:
            parallel_output = glm_model.parallel_output
            output_predict = glm_model.output_predict
            hidden_size = glm_model.hidden_size
            vocab_size = glm_model.vocab_size
        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, input_ids, *inputs, hook=None, **kw):
        inter_vars = []
        if self.output_predict:
            if self.parallel_output:
                dim = mpu.divide(self.vocab_size, mpu.get_model_parallel_world_size())
            else:
                dim = self.vocab_size
        else:
            dim = self.hidden_size
        return hook_return(hook, inter_vars, (torch.ones([*input_ids.size(), dim], device=input_ids.device),))


class EncoderDecoder(torch.nn.Module):
    """Seq2Seq Transformer Model
    The output of the forward method are the logits (parallel or serial depending on the `parallel_output` flag).
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 output_predict=True
                 ):
        super(EncoderDecoder, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.encoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers)
        self.decoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers,
                                                   use_decoder_layer=True)

    def forward(self, source_ids, target_ids, source_position_ids, target_position_ids, source_mask, target_mask):
        # Embeddings.
        source_embeddings = self.word_embeddings(source_ids)
        target_embeddings = self.word_embeddings(target_ids)

        # Transformer.
        encoder_output, _ = self.encoder(source_embeddings, source_position_ids, source_mask)
        decoder_output, _ = self.decoder(target_embeddings, target_position_ids, target_mask)
        if self.output_predict:
            # Parallel logits.
            output_parallel = mpu.copy_to_model_parallel_region(decoder_output)
            logits_parallel = F.linear(output_parallel, self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel,)

            return (mpu.gather_from_model_parallel_region(logits_parallel),)
        else:
            return (decoder_output,)


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
