

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BipartiteGNNEncoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Transformation weights
        self.W_item = nn.Linear(input_dim, hidden_dim)
        self.W_neigh = nn.Linear(input_dim, hidden_dim)
        self.W_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, item_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        
        h_self = self.W_item(item_features)

        
        degree = adj_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj_matrix / degree

        
        h_neigh = torch.matmul(adj_norm, item_features)
        h_neigh = self.W_neigh(h_neigh)

        
        h_combined = torch.cat([h_self, h_neigh], dim=-1)
        h_out = self.W_combine(h_combined)
        h_out = self.activation(h_out)
        h_out = self.norm(h_out)

        return h_out


class STBEstimator:
    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 alpha: float = 0.4, epsilon_x: float = 0.1,
                 epsilon_a: float = 0.1, pgd_steps: int = 10,
                 lr: float = 1e-3, epochs: int = 50,
                 rho: int = 50, beta: int = 40,
                 device: str = 'cpu'):
        
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.epsilon_x = epsilon_x
        self.epsilon_a = epsilon_a
        self.pgd_steps = pgd_steps
        self.lr = lr
        self.epochs = epochs
        self.rho = rho
        self.beta = beta
        self.device = device

        self.encoder = BipartiteGNNEncoder(input_dim, hidden_dim).to(device)

    def _build_copurchase_adj(self, session_items: List[int],
                               session_timestamps: List[int],
                               num_items: int) -> torch.Tensor:

        adj = torch.zeros(num_items, num_items, device=self.device)

        
        time_to_items = {}
        for item, t in zip(session_items, session_timestamps):
            if t not in time_to_items:
                time_to_items[t] = []
            time_to_items[t].append(item)

        
        for t, items in time_to_items.items():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    adj[items[i], items[j]] = 1.0
                    adj[items[j], items[i]] = 1.0

        
        for i in range(1, len(session_items)):
            adj[session_items[i - 1], session_items[i]] = 1.0
            adj[session_items[i], session_items[i - 1]] = 1.0

        return adj

    def _train_encoder(self, item_features: torch.Tensor,
                        adj_matrix: torch.Tensor):
        
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.encoder.train()

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            h = self.encoder(item_features, adj_matrix)

            h_norm = F.normalize(h, dim=-1)
            pred_adj = torch.matmul(h_norm, h_norm.t())
            pred_adj = torch.sigmoid(pred_adj)

            target = (adj_matrix > 0).float()
            loss = F.binary_cross_entropy(pred_adj, target)

            loss += 1e-5 * torch.norm(h, p=2)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.debug(f"STB Encoder Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

    def _pgd_perturbation(self, item_features: torch.Tensor,
                           adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.encoder.eval()
        num_items = item_features.shape[0]

        with torch.no_grad():
            h_orig = self.encoder(item_features, adj_matrix)

        delta_a = torch.zeros_like(adj_matrix, requires_grad=True)
        delta_x = torch.zeros_like(item_features, requires_grad=True)

        for step in range(self.pgd_steps):
            perturbed_adj = torch.clamp(adj_matrix + delta_a, 0, 1)
            perturbed_features = item_features + delta_x

            h_pert = self.encoder(perturbed_features, perturbed_adj)

            loss = -F.mse_loss(h_pert, h_orig.detach())
            loss.backward()

            with torch.no_grad():
                if delta_a.grad is not None:
                    delta_a_new = delta_a - self.epsilon_a * delta_a.grad.sign()
                    delta_a_new = self._project_adj_perturbation(
                        delta_a_new, adj_matrix, self.alpha, num_items)
                    delta_a.copy_(delta_a_new)

                if delta_x.grad is not None:
                    delta_x_new = delta_x - self.epsilon_x * delta_x.grad.sign()
                    delta_x_new = torch.clamp(delta_x_new, -self.epsilon_x, self.epsilon_x)
                    delta_x.copy_(delta_x_new)

            if delta_a.grad is not None:
                delta_a.grad.zero_()
            if delta_x.grad is not None:
                delta_x.grad.zero_()

        perturbed_adj = torch.clamp(adj_matrix + delta_a.detach(), 0, 1)
        perturbed_features = item_features + delta_x.detach()

        return perturbed_adj, perturbed_features

    def _project_adj_perturbation(self, delta_a: torch.Tensor,
                                    adj: torch.Tensor, alpha: float,
                                    num_items: int) -> torch.Tensor:

        budget = alpha * num_items
        norm = torch.norm(delta_a, p='fro')
        if norm > budget:
            delta_a = delta_a * budget / norm

        perturbed = adj + delta_a
        perturbed = torch.clamp(perturbed, 0, 1)
        delta_a = perturbed - adj

        return delta_a

    def compute_stb(self, item_features: np.ndarray, sessions: List[dict],
                     num_items: int) -> List[dict]:
        item_features_t = torch.FloatTensor(item_features).to(self.device)

        global_adj = torch.zeros(num_items, num_items, device=self.device)
        for session in sessions:
            items = session['items']
            timestamps = session.get('timestamps', list(range(len(items))))
            session_adj = self._build_copurchase_adj(items, timestamps, num_items)
            global_adj = torch.clamp(global_adj + session_adj, 0, 1)

        self._train_encoder(item_features_t, global_adj)

        updated_sessions = []

        for sess_idx, session in enumerate(sessions):
            items = session['items']
            timestamps = session.get('timestamps', list(range(len(items))))

            session_adj = self._build_copurchase_adj(items, timestamps, num_items)

            combined_adj = torch.clamp(0.5 * global_adj + 0.5 * session_adj, 0, 1)

            self.encoder.eval()
            with torch.no_grad():
                h_orig = self.encoder(item_features_t, combined_adj)

            pert_adj, pert_features = self._pgd_perturbation(item_features_t, combined_adj)
            with torch.no_grad():
                h_pert = self.encoder(pert_features, pert_adj)

            h_orig_norm = F.normalize(h_orig, dim=-1)
            h_pert_norm = F.normalize(h_pert, dim=-1)
            stb_scores = (h_orig_norm * h_pert_norm).sum(dim=-1)  # (num_items,)

            item_stb = [stb_scores[item].item() for item in items]

            stb_labels = self._classify_motivation(item_stb)

            updated_session = session.copy()
            updated_session['stb_labels'] = stb_labels
            updated_session['stb_scores'] = item_stb
            updated_sessions.append(updated_session)

            if (sess_idx + 1) % 500 == 0:
                logger.info(f"Processed {sess_idx + 1}/{len(sessions)} sessions")

        return updated_sessions

    def _classify_motivation(self, stb_scores: List[float]) -> List[int]:
        n = len(stb_scores)
        if n == 0:
            return []

        sorted_indices = np.argsort(stb_scores)[::-1] 

        n_stable = max(1, int(n * self.rho / 100))
        n_expl = max(1, int(n * self.beta / 100))

        labels = [2] * n

        for i in range(min(n_stable, n)):
            labels[sorted_indices[i]] = 0

        for i in range(min(n_expl, n)):
            labels[sorted_indices[-(i + 1)]] = 1

        return labels

    def _build_global_adj_fast(self, sessions: List[dict],
                               num_items: int) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_set = set()
        item_frequency = torch.zeros(num_items, device=self.device)

        for session in sessions:
            items = session['items']
            timestamps = session.get('timestamps', list(range(len(items))))

            time_to_items = {}
            for item, t in zip(items, timestamps):
                time_to_items.setdefault(t, []).append(item)
                item_frequency[item] += 1

            for t_items in time_to_items.values():
                for i in range(len(t_items)):
                    for j in range(i + 1, len(t_items)):
                        a, b = t_items[i], t_items[j]
                        edge_set.add((a, b))
                        edge_set.add((b, a))

            for i in range(1, len(items)):
                edge_set.add((items[i - 1], items[i]))
                edge_set.add((items[i], items[i - 1]))

        global_adj = torch.zeros(num_items, num_items, device=self.device)
        if edge_set:
            edges = torch.tensor(list(edge_set), dtype=torch.long, device=self.device)
            global_adj[edges[:, 0], edges[:, 1]] = 1.0

        return global_adj, item_frequency

    def compute_stb_repurchase(self, sessions: List[dict], num_items: int,
                               freq_weight: float = 0.5) -> List[dict]:
        total = np.zeros(num_items, dtype=np.float64)
        users = np.zeros(num_items, dtype=np.float64)
        for s in sessions:
            seen = set()
            for it in s['items']:
                total[it] += 1.0
                if it not in seen:
                    users[it] += 1.0
                    seen.add(it)

        repurchase = total / np.maximum(users, 1.0)

        def _z(x):
            return (x - x.mean()) / (x.std() + 1e-8)

        stb = _z(repurchase) + freq_weight * _z(np.log1p(total))
        stb[total == 0] = stb.min() - 1.0 

        updated_sessions = []
        for session in sessions:
            items = session['items']
            item_stb = [float(stb[it]) for it in items]
            stb_labels = self._classify_motivation(item_stb)
            us = session.copy()
            us['stb_labels'] = stb_labels
            us['stb_scores'] = item_stb
            updated_sessions.append(us)
        logger.info("Repurchase STB computation complete.")
        return updated_sessions

    def compute_stb_consistency(self, item_features: np.ndarray,
                                sessions: List[dict], num_items: int,
                                alpha: float = 0.7) -> List[dict]:
        F_feat = item_features.astype(np.float32)
        D = F_feat.shape[1]

        total = sum(len(s['items']) for s in sessions)
        ctx = np.zeros((total, D), dtype=np.float32)
        iid = np.zeros(total, dtype=np.int64)
        pos = 0
        for s in sessions:
            items = s['items']
            ts = s.get('timestamps', list(range(len(items))))
            L = len(items)
            t2pos = {}
            for p, t in enumerate(ts):
                t2pos.setdefault(t, []).append(p)
            for p in range(L):
                neigh = [items[q] for q in t2pos[ts[p]] if q != p]
                if p > 0:
                    neigh.append(items[p - 1])
                if p < L - 1:
                    neigh.append(items[p + 1])
                if neigh:
                    ctx[pos] = F_feat[neigh].mean(axis=0)
                else:
                    ctx[pos] = F_feat[items[p]]
                iid[pos] = items[p]
                pos += 1

        ctxn = ctx / (np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-8)
        centroid = np.zeros((num_items, D), dtype=np.float32)
        np.add.at(centroid, iid, ctxn)
        count = np.bincount(iid, minlength=num_items).astype(np.float32)
        centroid = centroid / np.maximum(count, 1)[:, None]
        centroidn = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8)

        cons_occ = (ctxn * centroidn[iid]).sum(axis=1)
        consistency = np.zeros(num_items, dtype=np.float32)
        np.add.at(consistency, iid, cons_occ)
        consistency = consistency / np.maximum(count, 1)

        def _z(x):
            m = x.mean()
            sd = x.std()
            return (x - m) / (sd + 1e-8)

        log_freq = np.log1p(count)
        stb = alpha * _z(consistency) + (1.0 - alpha) * _z(log_freq)
        stb[count == 0] = stb.min() - 1.0

        updated_sessions = []
        for session in sessions:
            items = session['items']
            item_stb = [float(stb[it]) for it in items]
            stb_labels = self._classify_motivation(item_stb)
            us = session.copy()
            us['stb_labels'] = stb_labels
            us['stb_scores'] = item_stb
            updated_sessions.append(us)

        return updated_sessions

    def compute_stb_fast(self, item_features: np.ndarray, sessions: List[dict],
                          num_items: int) -> List[dict]:
        item_features_t = torch.FloatTensor(item_features).to(self.device)

        global_adj, item_frequency = self._build_global_adj_fast(sessions, num_items)

        self._train_encoder(item_features_t, global_adj)
        self.encoder.eval()
        with torch.no_grad():
            h_orig = self.encoder(item_features_t, global_adj)

        pert_adj, pert_features = self._pgd_perturbation(item_features_t, global_adj)
        with torch.no_grad():
            h_pert = self.encoder(pert_features, pert_adj)

        h_orig_norm = F.normalize(h_orig, dim=-1)
        h_pert_norm = F.normalize(h_pert, dim=-1)
        global_stb = (h_orig_norm * h_pert_norm).sum(dim=-1)

        freq_norm = item_frequency / (item_frequency.max() + 1e-8)

        combined_stb = 0.7 * global_stb + 0.3 * freq_norm

        updated_sessions = []
        for session in sessions:
            items = session['items']
            item_stb = [combined_stb[item].item() for item in items]
            stb_labels = self._classify_motivation(item_stb)

            updated_session = session.copy()
            updated_session['stb_labels'] = stb_labels
            updated_session['stb_scores'] = item_stb
            updated_sessions.append(updated_session)

        return updated_sessions
