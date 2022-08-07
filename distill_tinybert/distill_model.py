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
        logits, *mems = self.model(*inputs, **kwargs)
        if 'is_distill' in kwargs and kwargs['is_distill']:
            # [[[layernorm_output,[attention_scores]],..],output]
            inter_vars = logits
            for v in inter_vars[0]:
                v[0] = self.fit_dense(v[0])
            inter_vars[1] = self.fit_dense(inter_vars[1])
            return inter_vars, *mems 
        return logits, *mems

    @staticmethod
    def compute_loss(s_inter_vars, t_inter_vars, layers_per_block=2):
        outputs = 0.
        # attentions
        student_reps = [i[1][0] for i in s_inter_vars[0]]
        teacher_reps = [i[1][0] for i in t_inter_vars[0]]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            outputs += F.mse_loss(student_rep, teacher_rep)
        mpu.reduce_from_model_parallel_region(outputs)
        # emb + hidden_states
        student_reps = [i[0] for i in s_inter_vars[0]] + [s_inter_vars[1]]
        teacher_reps = [i[0] for i in t_inter_vars[0]] + [t_inter_vars[1]]
        new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(len(student_reps))]
        for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
            outputs += F.mse_loss(student_rep, teacher_rep)
        return outputs
