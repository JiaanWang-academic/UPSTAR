"""
Configuration management for UPSTAR.
"""

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='UPSTAR: User Purchase Motivation-Aware Product Recommender System')

    # Data settings
    parser.add_argument('--dataset', type=str, default='tafeng',
                        choices=['tafeng', 'ijcai15', 'cetailer'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help='Directory of data files')
    parser.add_argument('--min_session_len', type=int, default=3,
                        help='Minimum session length')
    parser.add_argument('--category_csv', type=str,
                        default='ta_feng_all_months_merged.csv')
    parser.add_argument('--no_category', action='store_true',
                        help='Disable category (PRODUCT_SUBCLASS) embedding features')

    # STB estimation settings
    parser.add_argument('--stb_method', type=str, default='repurchase',
                        choices=['repurchase', 'consistency', 'fast'],
                        help="STB measure: 'repurchase' (habitual re-purchase "
                             "intensity, default & best), 'consistency' (context "
                             "consistency), 'fast' (original global-PGD, degenerate)")
    parser.add_argument('--rho', type=int, default=50,
                        help='Percentage of items classified as stable preference (%%)')
    parser.add_argument('--beta', type=int, default=40,
                        help='Percentage of items classified as exploratory intent (%%)')
    parser.add_argument('--stb_hidden_size', type=int, default=512,
                        help='Hidden size for STB GNN encoder')
    parser.add_argument('--stb_lr', type=float, default=1e-3,
                        help='Learning rate for STB estimation')
    parser.add_argument('--perturbation_alpha', type=float, default=0.4,
                        help='Perturbation budget for adjacency matrix')
    parser.add_argument('--perturbation_epsilon', type=float, default=0.1,
                        help='Perturbation budget for features')
    parser.add_argument('--pgd_epsilon', type=float, default=0.1,
                        help='PGD step size')
    parser.add_argument('--pgd_steps', type=int, default=10,
                        help='Number of PGD steps')
    parser.add_argument('--stb_epochs', type=int, default=50,
                        help='Number of epochs for STB estimation')

    parser.add_argument('--item_embed_dim', type=int, default=128,
                        help='Item embedding/representation dimension')
    parser.add_argument('--gnn_layers', type=int, default=1,
                        help='Number of GNN layers for item representation')
    parser.add_argument('--gnn_type', type=str, default='sage',
                        choices=['sage', 'gat', 'gcn'],
                        help='Type of GNN for item representation learning')

    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                        help='Hidden size for LSTM networks')
    parser.add_argument('--lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--disable_full_seq', action='store_true',
                        help='Disable the full-sequence (ordering-preserving) '
                             'global LSTM branch')
    parser.add_argument('--cosine', action='store_true',
                        help='Use cosine (L2-normalized) scoring with a learnable '
                             'temperature instead of plain dot-product scoring')
    parser.add_argument('--tie_output', action='store_true',
                        help='Score against the GNN item reps (paper-faithful, '
                             'tied). Default uses a dedicated learnable output '
                             'embedding, which is far stronger as a classifier.')
    parser.add_argument('--no_repeat', action='store_true',
                        help='Disable the repeat-aware (stable-preference) scoring '
                             'bonus for items already seen in the session')
    parser.add_argument('--no_attn', action='store_true',
                        help='Disable NARM/STAMP-style attention pooling in the '
                             'LSTM encoders (fall back to last-hidden pooling)')
    parser.add_argument('--no_basket_shuffle', action='store_true',
                        help='Disable within-day (basket) shuffling augmentation. '
                             'By default same-day items are treated as an unordered '
                             'basket and permuted each epoch.')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for UPSTAR model (paper uses 3e-4; '
                             '1e-3 converges much faster with the decoupled head)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lambda_', type=float, default=0.7,
                        help='Lambda for dual teacher-student loss')
    parser.add_argument('--tau_s', type=float, default=0.5,
                        help='Temperature for stable preference teacher')
    parser.add_argument('--tau_e', type=float, default=0.5,
                        help='Temperature for exploratory intent teacher')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing for cross-entropy losses')
    parser.add_argument('--aux_ce_weight', type=float, default=0.3,
                        help='Weight of the auxiliary CE on sub-model heads')
    parser.add_argument('--warmup_steps', type=int, default=200,
                        help='Linear LR warmup steps (0 disables warmup)')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay for evaluation weights (0 disables EMA)')
    parser.add_argument('--gnn_refresh', type=int, default=4,
                        help='Recompute full-graph item representations every N '
                             'batches (near-lossless speedup; 1 = every batch)')
    parser.add_argument('--no_pop_bias_init', action='store_true',
                        help='Disable popularity-prior initialization of the '
                             'item scoring bias')

    parser.add_argument('--aug_max', type=int, default=20,
                        help='Max augmented (prefix -> next-item) targets per '
                             'training session. <=0 keeps only the last item.')
    # STB caching
    parser.add_argument('--recompute_stb', action='store_true',
                        help='Force recomputation of STB labels (ignore cache)')

    # Cross-validation
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of folds for cross-validation')

    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto/cpu/cuda/mps)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log interval in batches')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Directory to save model checkpoints')

    return parser.parse_args()
