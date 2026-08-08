
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MotivationLSTM(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4,
                 dropout: float = 0.1, attn: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attn = attn

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        if attn:
            self.W_out = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W_last = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if lengths.max() <= 0:
            return torch.zeros(B, self.hidden_dim, device=x.device)

        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        sorted_x = x[sorted_idx]

        non_zero_mask = sorted_lengths > 0
        if non_zero_mask.sum() == 0:
            return torch.zeros(B, self.hidden_dim, device=x.device)

        valid_x = sorted_x[non_zero_mask]
        valid_lengths = sorted_lengths[non_zero_mask]

        packed = nn.utils.rnn.pack_padded_sequence(
            valid_x, valid_lengths.cpu().clamp(min=1), batch_first=True
        )
        out_packed, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]

        if self.attn:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
            T = outputs.shape[1]
            ar = torch.arange(T, device=x.device).unsqueeze(0)
            mask = ar < valid_lengths.unsqueeze(1)
            e = self.v(torch.tanh(self.W_out(outputs) +
                                  self.W_last(h_last).unsqueeze(1))).squeeze(-1)
            e = e.masked_fill(~mask, float('-inf'))
            a = torch.softmax(e, dim=1).unsqueeze(-1)
            context = (a * outputs).sum(dim=1)
            z_valid = self.combine(torch.cat([context, h_last], dim=-1))
        else:
            z_valid = h_last

        z = torch.zeros(B, self.hidden_dim, device=x.device)
        z[sorted_idx[non_zero_mask]] = z_valid
        return self.layer_norm(self.output_proj(z))


class FusionGate(nn.Module):

    def __init__(self, hidden_dim: int, num_views: int = 4):
        super().__init__()
        self.num_views = num_views
        self.gate_proj = nn.Linear(hidden_dim * num_views, num_views)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, views) -> torch.Tensor:
        combined = torch.cat(views, dim=-1)
        gate_weights = torch.softmax(self.gate_proj(combined), dim=-1)  # (batch, num_views)

        z_global = 0
        for i, v in enumerate(views):
            z_global = z_global + gate_weights[:, i:i + 1] * v

        z_global = self.layer_norm(self.output_proj(z_global))
        return z_global


class UPSTAR(nn.Module):

    def __init__(self, num_items: int, item_embed_dim: int = 128,
                 lstm_hidden_dim: int = 128, lstm_layers: int = 4,
                 dropout: float = 0.1, use_full_seq: bool = True,
                 cosine_scoring: bool = False, tie_output: bool = False,
                 use_repeat: bool = True, attn: bool = True):
        super().__init__()
        self.num_items = num_items
        self.item_embed_dim = item_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.use_full_seq = use_full_seq
        self.cosine_scoring = cosine_scoring
        self.tie_output = tie_output
        self.use_repeat = use_repeat

        
        self.s_model = MotivationLSTM(item_embed_dim, lstm_hidden_dim, lstm_layers, dropout, attn=attn)
        self.e_model = MotivationLSTM(item_embed_dim, lstm_hidden_dim, lstm_layers, dropout, attn=attn)
        self.o_model = MotivationLSTM(item_embed_dim, lstm_hidden_dim, lstm_layers, dropout, attn=attn)
        if use_full_seq:
            self.g_model = MotivationLSTM(item_embed_dim, lstm_hidden_dim, lstm_layers, dropout, attn=attn)

        num_views = 4 if use_full_seq else 3
        self.fusion_gate = FusionGate(lstm_hidden_dim, num_views=num_views)

        self.score_proj = nn.Linear(lstm_hidden_dim, item_embed_dim)
        self.out_embedding = nn.Embedding(num_items, item_embed_dim)
        nn.init.normal_(self.out_embedding.weight, std=0.05)
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))
        self.repeat_scale = nn.Parameter(torch.tensor(0.5413))
        self.dropout = nn.Dropout(dropout)

    def _repeat_bias(self, full_items: torch.Tensor,
                     full_lens: torch.Tensor) -> torch.Tensor:
        B, S = full_items.shape
        device = full_items.device
        pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S).float()
        valid = (pos < full_lens.unsqueeze(1).float())
        recency = (pos + 1.0) * valid
        bias = torch.zeros(B, self.num_items, device=device)
        bias.scatter_add_(1, full_items, recency)
        bias = bias / (bias.amax(dim=1, keepdim=True) + 1e-8)
        bias[:, 0] = 0.0
        return bias

    def _score(self, z: torch.Tensor, item_repr: torch.Tensor) -> torch.Tensor:
        q = self.score_proj(self.dropout(z))  
        W = item_repr if self.tie_output else self.out_embedding.weight
        if self.cosine_scoring:
            q = F.normalize(q, dim=-1)
            W = F.normalize(W, dim=-1)
            scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits = scale * torch.matmul(q, W.t()) + self.item_bias
        else:
            logits = torch.matmul(q, W.t()) + self.item_bias
        return logits

    def forward(self, item_representations: torch.Tensor,
                full_items: torch.Tensor, full_lens: torch.Tensor,
                stab_items: torch.Tensor, stab_lens: torch.Tensor,
                expl_items: torch.Tensor, expl_lens: torch.Tensor,
                other_items: torch.Tensor, other_lens: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        item_emb = item_representations
        if not self.tie_output:
            item_emb = item_emb + self.out_embedding.weight
        h_stab = item_emb[stab_items]
        h_expl = item_emb[expl_items]
        h_other = item_emb[other_items]

        z_stab = self.s_model(h_stab, stab_lens)
        z_expl = self.e_model(h_expl, expl_lens)
        z_other = self.o_model(h_other, other_lens)

        views = [z_stab, z_expl, z_other]
        z_full = None
        if self.use_full_seq:
            h_full = item_emb[full_items]
            z_full = self.g_model(h_full, full_lens)
            views.append(z_full)

        y_stab = self._score(z_stab, item_representations)
        y_expl = self._score(z_expl, item_representations)
        y_other = self._score(z_other, item_representations)

        z_global = self.fusion_gate(views)
        y_global = self._score(z_global, item_representations)

        if self.use_repeat:
            y_global = y_global + F.softplus(self.repeat_scale) * \
                self._repeat_bias(full_items, full_lens)

        outputs = {
            'y_stab': y_stab,
            'y_expl': y_expl,
            'y_other': y_other,
            'y_global': y_global,
            'z_stab': z_stab,
            'z_expl': z_expl,
            'z_other': z_other,
        }
        if z_full is not None:
            outputs['y_full'] = self._score(z_full, item_representations)
            outputs['z_full'] = z_full
        return outputs

    def predict(self, item_representations: torch.Tensor,
                full_items: torch.Tensor, full_lens: torch.Tensor,
                stab_items: torch.Tensor, stab_lens: torch.Tensor,
                expl_items: torch.Tensor, expl_lens: torch.Tensor,
                other_items: torch.Tensor, other_lens: torch.Tensor,
                ) -> torch.Tensor:
        outputs = self.forward(item_representations, full_items, full_lens,
                               stab_items, stab_lens, expl_items, expl_lens,
                               other_items, other_lens)
        scores = outputs['y_global']
        scores = scores.clone()
        scores[:, 0] = float('-inf')
        return scores


class DualTeacherStudentLoss(nn.Module):

    def __init__(self, lambda_: float = 0.7, tau_s: float = 0.5,
                 tau_e: float = 0.5, aux_ce_weight: float = 0.3,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_ = lambda_
        self.tau_s = tau_s
        self.tau_e = tau_e
        self.aux_ce_weight = aux_ce_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_stab = outputs['y_stab']
        y_expl = outputs['y_expl']
        y_other = outputs['y_other']
        y_global = outputs['y_global']

        loss_global = self.ce_loss(y_global, targets)
        loss_stab = self.ce_loss(y_stab, targets)
        loss_expl = self.ce_loss(y_expl, targets)
        loss_other = self.ce_loss(y_other, targets)
        aux = loss_stab + loss_expl + loss_other
        if 'y_full' in outputs:
            aux = aux + self.ce_loss(outputs['y_full'], targets)

        ce_loss = loss_global + self.aux_ce_weight * aux

        ts_stab_loss = self._kl_divergence(y_stab, y_global, self.tau_s)
        ts_expl_loss = self._kl_divergence(y_expl, y_global, self.tau_e)
        ts_loss = ts_stab_loss + ts_expl_loss

        total_loss = ce_loss + self.lambda_ * ts_loss

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'ts_stab_loss': ts_stab_loss,
            'ts_expl_loss': ts_expl_loss,
            'ts_loss': ts_loss,
        }

    def _kl_divergence(self, teacher_logits: torch.Tensor,
                        student_logits: torch.Tensor,
                        temperature: float) -> torch.Tensor:
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_log_prob, teacher_prob.detach(),
                           reduction='batchmean') * (temperature ** 2)
        return kl_loss
