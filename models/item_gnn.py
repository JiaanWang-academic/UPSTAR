
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ItemGNN(nn.Module):

    def __init__(self, num_items: int, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, gnn_type: str = 'sage',
                 dropout: float = 0.1, num_subclass: int = 0,
                 item_subclass: Optional[np.ndarray] = None):
        super().__init__()
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout

        
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)

        
        self.use_subclass = num_subclass > 0 and item_subclass is not None
        if self.use_subclass:
            self.subclass_embedding = nn.Embedding(num_subclass, hidden_dim, padding_idx=0)
            self.register_buffer('item_subclass',
                                 torch.as_tensor(item_subclass, dtype=torch.long))

        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            if gnn_type == 'sage':
                self.gnn_layers.append(SAGEConv(in_dim, out_dim))
            elif gnn_type == 'gat':
                self.gnn_layers.append(GATConv(in_dim, out_dim, heads=1))
            elif gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(in_dim, out_dim))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.normal_(self.item_embedding.weight, std=0.02)

    def forward(self, item_features: torch.Tensor,
                edge_index: torch.Tensor,
                item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        x = self.input_proj(item_features)

        all_ids = torch.arange(self.num_items, device=item_features.device)
        embed = self.item_embedding(all_ids)
        x = x + embed

        if self.use_subclass:
            x = x + self.subclass_embedding(self.item_subclass)

        for i, (gnn_layer, ln) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            x_new = gnn_layer(x, edge_index)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x_new = ln(x_new)
            x = x + x_new  

        
        final_repr = self.output_proj(torch.cat([x, embed], dim=-1))

        if item_ids is not None:
            return final_repr[item_ids]
        return final_repr

    def get_item_representations(self, item_features: torch.Tensor,
                                  edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.forward(item_features, edge_index)


class ItemGraphBuilder:

    def __init__(self, num_items: int, device: str = 'cpu'):
        self.num_items = num_items
        self.device = device
        self.edge_index = None
        self.edge_weight = None

    def build_from_sessions(self, sessions: list,
                            max_basket_clique: int = 30) -> torch.Tensor:
        import numpy as _np
        edge_set = set()
        for session in sessions:
            items = session['items']
            timestamps = session.get('timestamps', list(range(len(items))))

            day_groups = []
            cur_t, cur = None, []
            t2items = {}
            for it, t in zip(items, timestamps):
                t2items.setdefault(t, [])
                if it not in t2items[t]:
                    t2items[t].append(it)
            ordered_days = sorted(t2items.keys())

            
            for t in ordered_days:
                basket = t2items[t]
                if len(basket) > max_basket_clique:
                    basket = list(_np.random.choice(basket, max_basket_clique, replace=False))
                for a in range(len(basket)):
                    for b in range(a + 1, len(basket)):
                        edge_set.add((basket[a], basket[b]))
                        edge_set.add((basket[b], basket[a]))

            
            for d in range(1, len(ordered_days)):
                prev = t2items[ordered_days[d - 1]]
                nxt = t2items[ordered_days[d]]
                for a in prev:
                    for b in nxt:
                        edge_set.add((a, b))

        if not edge_set:
            self.edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
            return self.edge_index

        edges = list(edge_set)
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]

        self.edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        return self.edge_index

    def get_edge_index(self) -> torch.Tensor:
        if self.edge_index is None:
            raise RuntimeError("Graph not built yet. Call build_from_sessions first.")
        return self.edge_index
