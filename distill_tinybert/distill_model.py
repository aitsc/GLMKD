import sys, os
sys.path.append(os.getcwd())

import torch
from model import GLMModel
import mpu
import torch.nn.functional as F


class GLMStudent(torch.nn.Module):  # tinybert
    def __init__(self, language_model: GLMModel, args):
        super().__init__()
        self.model = language_model
        self.fit_dense = torch.nn.Linear(args.hidden_size, args.teacher_hidden_size)

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        if kwargs['distill_hook'] is not None:
            # {'embeddings':,'layers':{0:{'layernorm_output':,'attention_scores':,..},..},'output':,..}
            dhs = kwargs['distill_hook']['transformer']
            for k, v in dhs['layers'].items():
                v['layernorm_output'] = self.fit_dense(v['layernorm_output'])
            dhs['output'] = self.fit_dense(dhs['output'])
        return outputs


class HookModel(torch.nn.Module):  # tinybert
    def __init__(self, layers_per_block=2):
        super().__init__()
        self.layers_per_block = layers_per_block

    def forward(self, distill_hook_student, distill_hook_teacher):
        dhs = distill_hook_student['transformer']
        dht = distill_hook_teacher['transformer']
        get_layer_out_f = lambda dh, out: [i[1][out] for i in sorted(dh['layers'].items())]
        outputs = 0.
        # attentions
        student_reps = get_layer_out_f(dhs, 'attention_scores')
        teacher_reps = get_layer_out_f(dht, 'attention_scores')
        new_teacher_reps = [teacher_reps[i * self.layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_reps.distill_hook = True
            teacher_reps.distill_hook = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        mpu.reduce_from_model_parallel_region(outputs)
        # emb + hidden_states
        student_reps = get_layer_out_f(dhs, 'layernorm_output') + [dhs['output']]
        teacher_reps = get_layer_out_f(dht, 'layernorm_output') + [dht['output']]
        new_teacher_reps = [teacher_reps[i * self.layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            student_reps.distill_hook = True
            teacher_reps.distill_hook = True
            outputs += F.mse_loss(student_rep, teacher_rep)
        return outputs
