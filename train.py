"""
Main training and evaluation script for UPSTAR.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

from models.stb_estimator import STBEstimator
from models.item_gnn import ItemGNN, ItemGraphBuilder
from models.upstar import UPSTAR, DualTeacherStudentLoss
from data.dataset import (DataProcessor, SessionDataset, get_dataloader,
                           build_item_graph, kfold_split, collate_fn)
from utils.config import get_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_device(device_str: str) -> torch.device:
    """Get computing device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EMA:

    def __init__(self, modules, decay: float = 0.999):
        self.decay = decay
        self.params = [p for m in modules for p in m.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self.params]
        self.backup = None

    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self):
        self.backup = [p.detach().clone() for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.copy_(s)

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for b, p in zip(self.backup, self.params):
            p.copy_(b)
        self.backup = None


def train_epoch(model: UPSTAR, item_gnn: ItemGNN, dataloader,
                item_features: torch.Tensor, edge_index: torch.Tensor,
                optimizer: optim.Optimizer, loss_fn: DualTeacherStudentLoss,
                device: torch.device, log_interval: int = 10,
                ema: 'EMA' = None, base_lr: float = None,
                warmup_steps: int = 0, start_step: int = 0,
                gnn_refresh: int = 1) -> Tuple[Dict[str, float], int]:
    
    model.train()
    item_gnn.train()

    total_losses = {'total_loss': 0, 'ce_loss': 0, 'ts_loss': 0}
    num_batches = 0
    global_step = start_step
    cached_item_repr = None

    for batch_idx, batch in enumerate(dataloader):
        
        if warmup_steps and base_lr and global_step < warmup_steps:
            warm_lr = base_lr * float(global_step + 1) / float(warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = warm_lr

        
        targets = batch['targets'].to(device)
        input_items = batch['input_items'].to(device)
        session_lens = batch['session_lens'].to(device)
        stab_items = batch['stab_items'].to(device)
        expl_items = batch['expl_items'].to(device)
        other_items = batch['other_items'].to(device)
        stab_lens = batch['stab_lens'].to(device)
        expl_lens = batch['expl_lens'].to(device)
        other_lens = batch['other_lens'].to(device)

        
        if cached_item_repr is None or (batch_idx % max(gnn_refresh, 1) == 0):
            item_repr = item_gnn(item_features, edge_index)
            cached_item_repr = item_repr.detach()
        else:
            item_repr = cached_item_repr

        
        outputs = model(item_repr, input_items, session_lens,
                        stab_items, stab_lens,
                        expl_items, expl_lens, other_items, other_lens)

        
        losses = loss_fn(outputs, targets)
        total_loss = losses['total_loss']

        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(item_gnn.parameters()), max_norm=5.0)
        optimizer.step()
        if ema is not None:
            ema.update()
        global_step += 1

        
        total_losses['total_loss'] += losses['total_loss'].item()
        total_losses['ce_loss'] += losses['ce_loss'].item()
        total_losses['ts_loss'] += losses['ts_loss'].item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.info(f"  Batch {batch_idx + 1}: "
                        f"Loss={losses['total_loss'].item():.4f}, "
                        f"CE={losses['ce_loss'].item():.4f}, "
                        f"TS={losses['ts_loss'].item():.4f}")

    
    for key in total_losses:
        total_losses[key] /= max(num_batches, 1)

    return total_losses, global_step





def train_upstar(args):
    
    
    device = get_device(args.device)
    set_seed(args.seed)
    logger.info(f"Using device: {device}")

    
    logger.info(f"Loading dataset: {args.dataset}")
    processor = DataProcessor(args.data_dir, args.dataset, args.min_session_len)
    sessions, num_items, item_features = processor.load_and_preprocess()
    logger.info(f"Loaded {len(sessions)} sessions with {num_items} items")

    
    if item_features is None:
        item_features = np.random.randn(num_items, args.item_embed_dim).astype(np.float32)
    item_features_t = torch.FloatTensor(item_features).to(device)
    input_dim = item_features.shape[1]

    
    item_subclass, num_subclass = None, 0
    if not args.no_category:
        from data.dataset import load_item_categories
        cat_csv = os.path.join(args.data_dir, args.dataset, args.category_csv)
        if os.path.exists(cat_csv) and getattr(processor, 'item2idx', None):
            item_subclass, num_subclass = load_item_categories(
                cat_csv, processor.item2idx, num_items)
        else:
            logger.info(f"Category CSV not found at {cat_csv}; skipping category features.")

    
    logger.info("=" * 60)
    logger.info("Step 1: Computing STB labels for purchase motivation identification")
    logger.info("=" * 60)

    stb_estimator = STBEstimator(
        input_dim=input_dim,
        hidden_dim=args.stb_hidden_size,
        alpha=args.perturbation_alpha,
        epsilon_x=args.perturbation_epsilon,
        epsilon_a=args.pgd_epsilon,
        pgd_steps=args.pgd_steps,
        lr=args.stb_lr,
        epochs=args.stb_epochs,
        rho=args.rho,
        beta=args.beta,
        device=str(device),
    )

    
    import pickle
    stb_cache_path = os.path.join(args.data_dir, args.dataset,
                                  f'stb_sessions_{args.stb_method}_rho{args.rho}_beta{args.beta}.pkl')
    if (not args.recompute_stb) and os.path.exists(stb_cache_path):
        logger.info(f"Loading cached STB labels from {stb_cache_path}")
        with open(stb_cache_path, 'rb') as f:
            sessions = pickle.load(f)
    else:
        if args.stb_method == 'repurchase':
            sessions = stb_estimator.compute_stb_repurchase(sessions, num_items)
        elif args.stb_method == 'consistency':
            sessions = stb_estimator.compute_stb_consistency(item_features, sessions, num_items)
        else:  # 'fast' (original, degenerate -> popularity)
            sessions = stb_estimator.compute_stb_fast(item_features, sessions, num_items)
        with open(stb_cache_path, 'wb') as f:
            pickle.dump(sessions, f)
        logger.info(f"Saved STB labels cache to {stb_cache_path}")
    logger.info("STB computation complete.")

    
    stab_count = sum(sum(1 for l in s['stb_labels'] if l == 0) for s in sessions)
    expl_count = sum(sum(1 for l in s['stb_labels'] if l == 1) for s in sessions)
    other_count = sum(sum(1 for l in s['stb_labels'] if l == 2) for s in sessions)
    total_items = stab_count + expl_count + other_count
    logger.info(f"Motivation distribution: "
                f"Stable={stab_count}({stab_count / total_items * 100:.1f}%), "
                f"Exploratory={expl_count}({expl_count / total_items * 100:.1f}%), "
                f"Other={other_count}({other_count / total_items * 100:.1f}%)")

    
    logger.info("=" * 60)
    logger.info(f"Step 2: {args.n_folds}-fold Cross-Validation Training")
    logger.info("=" * 60)

    for fold_idx, (train_sessions, test_sessions) in enumerate(
            kfold_split(sessions, args.n_folds, args.seed)):

        logger.info(f"\n{'=' * 40}")
        logger.info(f"Fold {fold_idx + 1}/{args.n_folds}")
        logger.info(f"Train: {len(train_sessions)}, Test: {len(test_sessions)}")
        logger.info(f"{'=' * 40}")

        
        graph_builder = ItemGraphBuilder(num_items, str(device))
        edge_index = graph_builder.build_from_sessions(train_sessions)

        
        item_gnn = ItemGNN(
            num_items=num_items,
            input_dim=input_dim,
            hidden_dim=args.item_embed_dim,
            num_layers=args.gnn_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            num_subclass=num_subclass,
            item_subclass=item_subclass,
        ).to(device)

        model = UPSTAR(
            num_items=num_items,
            item_embed_dim=args.item_embed_dim,
            lstm_hidden_dim=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            dropout=args.dropout,
            use_full_seq=not args.disable_full_seq,
            cosine_scoring=args.cosine,
            tie_output=args.tie_output,
            use_repeat=not args.no_repeat,
            attn=not args.no_attn,
        ).to(device)

        
        if not args.no_pop_bias_init:
            freq = np.ones(num_items, dtype=np.float64)
            for s in train_sessions:
                for it in s['items']:
                    freq[it] += 1.0
            freq[0] = 1.0  # padding
            log_freq = np.log(freq)
            log_freq = log_freq - log_freq.mean()
            with torch.no_grad():
                model.item_bias.copy_(torch.tensor(log_freq, dtype=torch.float32,
                                                   device=device))

        
        loss_fn = DualTeacherStudentLoss(
            lambda_=args.lambda_,
            tau_s=args.tau_s,
            tau_e=args.tau_e,
            aux_ce_weight=args.aux_ce_weight,
            label_smoothing=args.label_smoothing,
        )

        
        optimizer = optim.Adam(
            list(model.parameters()) + list(item_gnn.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        train_loader = get_dataloader(train_sessions, num_items,
                                       args.batch_size, shuffle=True,
                                       aug_max=args.aug_max,
                                       shuffle_basket=not args.no_basket_shuffle)

        # EMA of weights
        ema = EMA([model, item_gnn], decay=args.ema_decay) if args.ema_decay > 0 else None

        global_step = 0

        for epoch in range(args.epochs):
            start_time = time.time()

            train_losses, global_step = train_epoch(
                model, item_gnn, train_loader, item_features_t, edge_index,
                optimizer, loss_fn, device, args.log_interval,
                ema=ema, base_lr=args.lr, warmup_steps=args.warmup_steps,
                start_step=global_step, gnn_refresh=args.gnn_refresh)

            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s) - "
                        f"Loss: {train_losses['total_loss']:.4f}, "
                        f"CE: {train_losses['ce_loss']:.4f}, "
                        f"TS: {train_losses['ts_loss']:.4f}")

        # Save final model
        os.makedirs(args.save_dir, exist_ok=True)
        if ema is not None:
            ema.apply_shadow()
        torch.save({
            'model_state_dict': model.state_dict(),
            'item_gnn_state_dict': item_gnn.state_dict(),
            'epoch': args.epochs,
        }, os.path.join(args.save_dir, f'model_fold{fold_idx}.pt'))
        if ema is not None:
            ema.restore()

        logger.info(f"Fold {fold_idx + 1} training complete.")

    logger.info("All folds training complete.")


if __name__ == '__main__':
    args = get_args()
    train_upstar(args)
