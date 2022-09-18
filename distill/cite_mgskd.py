# MGSKD/src/loss.py 20220918
# https://github.com/LC97-pku/MGSKD/blob/main/src/loss.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SampleLoss(nn.Module):
    def __init__(self, n_relation_heads):
        super(SampleLoss, self).__init__()
        self.n_relation_heads = n_relation_heads

    def mean_pooling(self, hidden, attention_mask):
        hidden *= attention_mask.unsqueeze(-1).to(hidden)
        hidden = torch.sum(hidden, dim=1)
        hidden /= torch.sum(attention_mask, dim=1).unsqueeze(-1).to(hidden)
        return hidden

    def cal_angle_multihead(self, hidden):
        batch, dim = hidden.shape
        hidden = hidden.view(batch, self.n_relation_heads, -1).permute(1, 0, 2)
        norm_ = F.normalize(hidden.unsqueeze(1) - hidden.unsqueeze(2), p=2, dim=-1)
        angle = torch.einsum('hijd, hidk->hijk', norm_, norm_.transpose(-2, -1))
        return angle

    def forward(self, s_rep, t_rep, attention_mask):
        s_rep = self.mean_pooling(s_rep, attention_mask)
        t_rep = self.mean_pooling(t_rep, attention_mask)
        with torch.no_grad():
            t_angle = self.cal_angle_multihead(t_rep)
        s_angle = self.cal_angle_multihead(s_rep)
        loss = F.smooth_l1_loss(s_angle.view(-1), t_angle.view(-1), reduction='elementwise_mean')
        return loss


def expand_gather(input, dim, index):
    size = list(input.size())
    size[dim] = -1
    return input.gather(dim, index.expand(*size))


class TokenPhraseLoss(nn.Module):
    def __init__(self, n_relation_heads, k1, k2):
        super(TokenPhraseLoss, self).__init__()
        self.n_relation_heads = n_relation_heads
        self.k1 = k1
        self.k2 = k2

    def forward(self, s_rep, t_rep, attention_mask):
        attention_mask_extended = torch.einsum('bl, bp->blp', attention_mask, attention_mask)
        attention_mask_extended = attention_mask_extended.unsqueeze(1).repeat(1, self.n_relation_heads, 1, 1).float()
        s_pair, s_global_topk, s_local_topk = self.cal_pairinteraction_multihead(s_rep, attention_mask_extended,
                                                                                 self.n_relation_heads,
                                                                                 k1=min(self.k1, s_rep.shape[1]),
                                                                                 k2=min(self.k2, s_rep.shape[1]))
        with torch.no_grad():
            t_pair, t_global_topk, t_local_topk = self.cal_pairinteraction_multihead(t_rep, attention_mask_extended,
                                                                                     self.n_relation_heads,
                                                                                     k1=min(self.k1, t_rep.shape[1]),
                                                                                     k2=min(self.k2, t_rep.shape[1]))
        loss_pair = F.mse_loss(s_pair.view(-1), t_pair.view(-1), reduction='sum') / torch.sum(attention_mask_extended)
        s_angle, s_mask = self.calculate_tripletangleseq_multihead(s_rep, attention_mask, 1,
                                                                   t_global_topk, t_local_topk)
        with torch.no_grad():
            t_angle, t_mask = self.calculate_tripletangleseq_multihead(t_rep, attention_mask, 1,
                                                                       t_global_topk, t_local_topk)
        loss_triplet = F.smooth_l1_loss(s_angle.view(-1), t_angle.view(-1), reduction='sum') / torch.sum(s_mask)
        return loss_pair + loss_triplet

    def cal_pairinteraction_multihead(self, hidden, attention_mask_extended, n_relation_heads, k1, k2):
        batch, seq_len, dim = hidden.shape
        hidden = hidden.view(batch, seq_len, n_relation_heads, -1).permute(0, 2, 1, 3)
        scores = torch.matmul(hidden, hidden.transpose(-1, -2))
        scores = scores / math.sqrt(dim // n_relation_heads)
        scores = scores * attention_mask_extended
        scores_out = scores
        attention_mask_extended_add = (1.0 - attention_mask_extended) * -10000.0
        scores = scores + attention_mask_extended_add
        scores = F.softmax(scores, dim=-1)
        scores *= attention_mask_extended
        global_score = scores.sum(2).sum(1)
        global_topk = global_score.topk(k1, dim=1)[1]
        local_score = scores.sum(1)
        mask = torch.ones_like(local_score)
        mask[:, range(mask.shape[-2]), range(mask.shape[-1])] = 0.
        local_score = local_score * mask
        local_topk = local_score.topk(k2, dim=2)[1]
        index_ = global_topk.unsqueeze(-1)
        local_topk = expand_gather(local_topk, 1, index_)
        return scores_out, global_topk, local_topk

    def calculate_tripletangleseq_multihead(self, hidden, attention_mask, n_relation_heads, global_topk, local_topk):
        '''
            hidden: batch, len, dim
            attention_mask: batch, len
            global_topk: batch, k1
            local_topk: batch, k1, k2
        '''
        batch, seq_len, dim = hidden.shape
        hidden = hidden.view(batch, seq_len, n_relation_heads, -1).permute(0, 2, 1, 3)
        index_ = global_topk.unsqueeze(1).unsqueeze(-1)
        index_ = index_.repeat(1, n_relation_heads, 1, 1)
        hidden1 = expand_gather(hidden, 2, index_)
        sd = (hidden1.unsqueeze(3) - hidden.unsqueeze(2))
        index_ = local_topk.unsqueeze(1).repeat(1, n_relation_heads, 1, 1).unsqueeze(-1)
        sd = expand_gather(sd, 3, index_)
        norm_sd = F.normalize(sd, p=2, dim=-1)
        angle = torch.einsum('bhijd, bhidk->bhijk', norm_sd,norm_sd.transpose(-2, -1))
        attention_mask1 = attention_mask.gather(-1, global_topk)
        attention_mask_extended = attention_mask1.unsqueeze(2) + attention_mask.unsqueeze(1)
        attention_mask_extended = attention_mask_extended.unsqueeze(1).repeat(1, n_relation_heads, 1, 1)
        attention_mask_extended = attention_mask_extended.unsqueeze(-1)
        index_ = local_topk.unsqueeze(1).repeat(1, n_relation_heads, 1, 1).unsqueeze(-1)
        attention_mask_extended = expand_gather(attention_mask_extended, 3, index_)
        attention_mask_extended = (torch.einsum('bhijd, bhidk->bhijk', attention_mask_extended.float(),
                                                attention_mask_extended.transpose(-2, -1).float()) == 4).float()
        mask = angle.ne(0).float()
        mask[:, :, :, range(mask.shape[-2]), range(mask.shape[-1])] = 0.
        attention_mask_extended = attention_mask_extended * mask
        angle = angle * attention_mask_extended
        return angle, attention_mask_extended


def get_phrase_reps(hidden, phrase_pos, seq_lens, skip_nophrase=True, add_token=False):
    seq_lens_new = []
    attention_mask_new = []
    all_reps = []
    hidden_dim = hidden.shape[-1]
    for case_hidden, pos_tuple, length in zip(hidden, phrase_pos, seq_lens):
        if skip_nophrase and len(pos_tuple) == 0:
            continue
        case_hidden_new = []
        pos_new = []
        pos_new.append(0)
        for loc_a, loc_b in pos_tuple:
            pos_new.append(loc_a)
            pos_new.append(loc_b)
        pos_new.append(length.item())
        for i in range(len(pos_new) - 1):
            if i % 2 == 0:
                if add_token:
                    if pos_new[i + 1] - pos_new[i] > 0:
                        case_hidden_new.append(case_hidden[pos_new[i]:pos_new[i + 1]])
            else:
                case_hidden_new.append(torch.mean(case_hidden[pos_new[i]:pos_new[i + 1]], dim=0, keepdim=True))
        assert len(case_hidden_new) == len(pos_tuple)
        if not add_token:
            if len(case_hidden_new) < 3:
                continue
        case_hidden_new = torch.cat(case_hidden_new, dim=0)
        seq_lens_new.append(case_hidden_new.shape[0])
        all_reps.append(case_hidden_new)
    if len(seq_lens_new) == 0:
        return None, None
    max_len = max(seq_lens_new)
    hidden_new = []
    for i in range(len(seq_lens_new)):
        attention_mask_new.append([1] * seq_lens_new[i] + [0] * (max_len - seq_lens_new[i]))
        case_hidden_new = torch.cat([all_reps[i], all_reps[i][0].new(max_len - seq_lens_new[i], hidden_dim).fill_(0.)],
                                    dim=0)
        hidden_new.append(case_hidden_new)
    hidden_new = torch.stack(hidden_new, dim=0)
    attention_mask_new = torch.LongTensor(attention_mask_new).to(hidden_new.device)
    return hidden_new, attention_mask_new



class MGSKDLoss(nn.Module):
    def __init__(self, n_relation_heads=64, k1=20, k2=20, M=2, weights=(4., 1., 1.)):
        super(MGSKDLoss, self).__init__()
        self.n_relation_heads = n_relation_heads
        self.k1 = k1
        self.k2 = k2
        self.M = M
        self.w1, self.w2, self.w3 = weights
        self.sample_loss = SampleLoss(n_relation_heads)
        self.tokenphrase_loss = TokenPhraseLoss(n_relation_heads, k1, k2)

    # def forward(self, s_reps, t_reps, attention_mask, phrase_poses, input_lengths):
    #     token_loss = 0.
    #     phrase_loss = 0.
    #     sample_loss = 0.
    #     for layer_id, (s_rep, t_rep) in enumerate(zip(s_reps, t_reps)):
    #         if layer_id < self.M:
    #             token_loss += self.tokenphrase_loss(s_rep, t_rep, attention_mask)
    #             student_phrase_rep, mask = get_phrase_reps(s_rep, phrase_poses, input_lengths, skip_nophrase=True,
    #                                                        add_token=False)
    #             teacher_phrase_rep, mask = get_phrase_reps(t_rep, phrase_poses, input_lengths, skip_nophrase=True,
    #                                                        add_token=False)
    #             if student_phrase_rep is not None:
    #                 phrase_loss += self.tokenphrase_loss(student_phrase_rep, teacher_phrase_rep, mask)
    #         else:
    #             sample_loss += self.sample_loss(s_rep, t_rep, attention_mask)
    #     loss = self.w1 * sample_loss + self.w2 * token_loss + self.w3 * phrase_loss
    #     return loss, (sample_loss, token_loss, phrase_loss)

    # fast
    def forward(self, s_reps, t_reps, attention_mask, phrase_poses, input_lengths):
        token_loss = 0.
        phrase_loss = 0.
        sample_loss = 0.
        # phrase
        student_phrase_rep, mask = get_phrase_reps(torch.cat(s_reps[:self.M], dim=-1), phrase_poses,
                                                   input_lengths, skip_nophrase=True, add_token=False)
        teacher_phrase_rep, mask = get_phrase_reps(torch.cat(t_reps[:self.M], dim=-1), phrase_poses,
                                                   input_lengths, skip_nophrase=True, add_token=False)
        if student_phrase_rep is not None:
            student_phrase_reps = torch.split(student_phrase_rep, 768, dim=-1)
            teacher_phrase_reps = torch.split(teacher_phrase_rep, 768, dim=-1)
            for s_phrase_rep, t_phrase_rep in zip(student_phrase_reps, teacher_phrase_reps):
                phrase_loss += self.tokenphrase_loss(s_phrase_rep, t_phrase_rep, mask)
        for layer_id, (s_rep, t_rep) in enumerate(zip(s_reps, t_reps)):
            if layer_id < self.M:
                token_loss += self.tokenphrase_loss(s_rep, t_rep, attention_mask)
            else:
                sample_loss += self.sample_loss(s_rep, t_rep, attention_mask)
        loss = self.w1 * sample_loss + self.w2 * token_loss + self.w3 * phrase_loss
        return loss, (sample_loss, token_loss, phrase_loss)