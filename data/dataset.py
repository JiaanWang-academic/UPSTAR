

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pickle
import logging

logger = logging.getLogger(__name__)


def load_item_categories(csv_path: str, item2idx: dict, num_items: int):
    
    import csv as _csv
    prod2sub = {}
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        for row in reader:
            prod2sub[row['PRODUCT_ID']] = row['PRODUCT_SUBCLASS']

    sub2idx = {}
    item_subclass = np.zeros(num_items, dtype=np.int64)
    matched = 0
    for prod, idx in item2idx.items():
        sub = prod2sub.get(str(prod))
        if sub is None:
            continue
        if sub not in sub2idx:
            sub2idx[sub] = len(sub2idx) + 1
        item_subclass[idx] = sub2idx[sub]
        matched += 1
    num_subclass = len(sub2idx) + 1
    return item_subclass, num_subclass


def _shuffle_within_day(items, labels, times):
    
    import random
    groups = {}
    order = []
    for it, lb, t in zip(items, labels, times):
        if t not in groups:
            groups[t] = []
            order.append(t)
        groups[t].append((it, lb))
    out_items, out_labels = [], []
    for t in order:
        g = groups[t]
        if len(g) > 1:
            random.shuffle(g)
        for it, lb in g:
            out_items.append(it)
            out_labels.append(lb)
    return out_items, out_labels


class SessionDataset(Dataset):
    

    def __init__(self, sessions: List[dict], num_items: int, max_len: int = 200,
                 aug_max: int = 20, shuffle_basket: bool = False):
        self.num_items = num_items
        self.max_len = max_len
        self.aug_max = aug_max
        self.shuffle_basket = shuffle_basket
        self.samples = self._build_samples(sessions)

    def _build_samples(self, sessions: List[dict]) -> List[dict]:
        samples = []
        for session in sessions:
            items = session['items']
            labels = session.get('stb_labels', [2] * len(items))
            times = session.get('timestamps', list(range(len(items))))
            n = len(items)
            if n < 2:
                continue

            
            if self.aug_max is not None and self.aug_max > 0:
                t_start = max(1, n - self.aug_max)
            else:
                t_start = n - 1

            for t in range(t_start, n):
                lo = max(0, t - self.max_len)
                samples.append({
                    'input_items': items[lo:t],
                    'input_labels': labels[lo:t],
                    'input_times': times[lo:t],
                    'target': items[t],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        input_items = s['input_items']
        input_labels = s['input_labels']

        if self.shuffle_basket:
            input_items, input_labels = _shuffle_within_day(
                input_items, input_labels, s['input_times'])

        stab_items = [it for it, lb in zip(input_items, input_labels) if lb == 0]
        expl_items = [it for it, lb in zip(input_items, input_labels) if lb == 1]
        other_items = [it for it, lb in zip(input_items, input_labels) if lb == 2]

        return {
            'input_items': input_items,
            'target': s['target'],
            'stab_items': stab_items,
            'expl_items': expl_items,
            'other_items': other_items,
            'session_len': len(input_items),
        }


def collate_fn(batch):
    max_len = max(b['session_len'] for b in batch)
    max_stab = max(len(b['stab_items']) for b in batch) if batch else 1
    max_expl = max(len(b['expl_items']) for b in batch) if batch else 1
    max_other = max(len(b['other_items']) for b in batch) if batch else 1

    # Ensure at least length 1
    max_stab = max(max_stab, 1)
    max_expl = max(max_expl, 1)
    max_other = max(max_other, 1)

    batch_size = len(batch)

    input_items = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    stab_items = torch.zeros(batch_size, max_stab, dtype=torch.long)
    expl_items = torch.zeros(batch_size, max_expl, dtype=torch.long)
    other_items = torch.zeros(batch_size, max_other, dtype=torch.long)
    session_lens = torch.zeros(batch_size, dtype=torch.long)
    stab_lens = torch.zeros(batch_size, dtype=torch.long)
    expl_lens = torch.zeros(batch_size, dtype=torch.long)
    other_lens = torch.zeros(batch_size, dtype=torch.long)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        sl = b['session_len']
        input_items[i, :sl] = torch.tensor(b['input_items'], dtype=torch.long)
        targets[i] = b['target']
        session_lens[i] = sl
        mask[i, :sl] = True

        s_len = len(b['stab_items'])
        e_len = len(b['expl_items'])
        o_len = len(b['other_items'])

        if s_len > 0:
            stab_items[i, :s_len] = torch.tensor(b['stab_items'], dtype=torch.long)
        stab_lens[i] = max(s_len, 0)

        if e_len > 0:
            expl_items[i, :e_len] = torch.tensor(b['expl_items'], dtype=torch.long)
        expl_lens[i] = max(e_len, 0)

        if o_len > 0:
            other_items[i, :o_len] = torch.tensor(b['other_items'], dtype=torch.long)
        other_lens[i] = max(o_len, 0)

    return {
        'input_items': input_items,
        'targets': targets,
        'stab_items': stab_items,
        'expl_items': expl_items,
        'other_items': other_items,
        'session_lens': session_lens,
        'stab_lens': stab_lens,
        'expl_lens': expl_lens,
        'other_lens': other_lens,
        'mask': mask,
    }


def get_dataloader(sessions, num_items, batch_size, shuffle=True, max_len=200,
                   aug_max=20, shuffle_basket=False):
    dataset = SessionDataset(sessions, num_items, max_len, aug_max=aug_max,
                             shuffle_basket=shuffle_basket)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0)


class DataProcessor:

    def __init__(self, data_dir: str, dataset: str, min_session_len: int = 3):
        self.data_dir = data_dir
        self.dataset = dataset
        self.min_session_len = min_session_len
        self.item2idx = {}
        self.idx2item = {}
        self.num_items = 0
        self.item_features = None
        self.sessions = []

    def load_and_preprocess(self) -> Tuple[List[dict], int, Optional[np.ndarray]]:
        data_path = os.path.join(self.data_dir, self.dataset)

        if self.dataset == 'tafeng':
            return self._process_tafeng(data_path)
        elif self.dataset == 'ijcai15':
            return self._process_ijcai15(data_path)
        elif self.dataset == 'cetailer':
            return self._process_cetailer(data_path)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def _process_tafeng(self, data_path: str):
        # Try to load preprocessed data first
        cache_path = os.path.join(data_path, 'processed_sessions.pkl')
        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        # Try JSONL format first
        jsonl_path = os.path.join(data_path, 'tafeng.jsonl')
        if os.path.exists(jsonl_path):
            return self._load_jsonl(jsonl_path, data_path)

        # Fallback to CSV format
        raw_path = os.path.join(data_path, 'transactions.csv')
        if not os.path.exists(raw_path):
            return self._generate_synthetic_data(data_path)

        df = pd.read_csv(raw_path)
        return self._build_sessions(df, data_path, user_col='customer_id',
                                     item_col='product_id', time_col='timestamp',
                                     cat_col='product_subclass')

    def _load_jsonl(self, jsonl_path: str, data_path: str):
        logger.info(f"Loading JSONL data from {jsonl_path}")

        all_items = set()
        raw_sessions = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                user_id = data['user']
                session_entries = data['session']
                items = [entry['item'] for entry in session_entries]
                timestamps = [entry['time'] for entry in session_entries]
                all_items.update(items)
                raw_sessions.append({
                    'user_id': user_id,
                    'raw_items': items,
                    'timestamps': timestamps,
                })

        
        sorted_items = sorted(all_items)
        self.item2idx = {item: idx + 1 for idx, item in enumerate(sorted_items)}
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.num_items = len(sorted_items) + 1  # +1 for padding

        
        sessions = []
        for raw in raw_sessions:
            items = [self.item2idx[it] for it in raw['raw_items']]
            timestamps = raw['timestamps']

            if len(items) >= self.min_session_len:
                sessions.append({
                    'user_id': raw['user_id'],
                    'items': items,
                    'timestamps': timestamps,
                    'stb_labels': [2] * len(items),  # placeholder, computed later by STB
                })

        self.sessions = sessions

        
        logger.info("Building item features from co-occurrence statistics...")
        self.item_features = self._build_item_features_from_sessions(sessions)

        
        self._save_cache(data_path)

        logger.info(f"Loaded {len(sessions)} sessions, {self.num_items} items from JSONL")
        return sessions, self.num_items, self.item_features

    def _build_item_features_from_sessions(self, sessions: List[dict],
                                            feature_dim: int = 64) -> np.ndarray:
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import svds

        
        cooccur = lil_matrix((self.num_items, self.num_items), dtype=np.float32)

        for session in sessions:
            items = session['items']
            timestamps = session['timestamps']

            
            time_to_items = defaultdict(list)
            for item, t in zip(items, timestamps):
                time_to_items[t].append(item)

            for t_items in time_to_items.values():
                for i in range(len(t_items)):
                    for j in range(i + 1, len(t_items)):
                        cooccur[t_items[i], t_items[j]] += 1
                        cooccur[t_items[j], t_items[i]] += 1

            for i in range(1, len(items)):
                cooccur[items[i - 1], items[i]] += 0.5
                cooccur[items[i], items[i - 1]] += 0.5

        cooccur_csr = cooccur.tocsr()
        actual_dim = min(feature_dim, min(cooccur_csr.shape) - 2)

        if actual_dim < 1:
            logger.warning("Co-occurrence matrix too small for SVD, using random features")
            return np.random.randn(self.num_items, feature_dim).astype(np.float32) * 0.01

        try:
            U, S, _ = svds(cooccur_csr.astype(np.float64), k=actual_dim)
            item_features = U * np.sqrt(S)[np.newaxis, :]
            if item_features.shape[1] < feature_dim:
                padding = np.zeros((self.num_items, feature_dim - item_features.shape[1]),
                                   dtype=np.float32)
                item_features = np.concatenate([item_features, padding], axis=1)
            item_features = item_features.astype(np.float32)
        except Exception as e:
            logger.warning(f"SVD failed: {e}, using random features")
            item_features = np.random.randn(self.num_items, feature_dim).astype(np.float32) * 0.01

        
        norms = np.linalg.norm(item_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_features = item_features / norms

        return item_features

    def _process_ijcai15(self, data_path: str):
        
        cache_path = os.path.join(data_path, 'processed_sessions.pkl')
        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        
        jsonl_path = os.path.join(data_path, 'ijcai15.jsonl')
        if os.path.exists(jsonl_path):
            return self._load_jsonl(jsonl_path, data_path)

        raw_path = os.path.join(data_path, 'transactions.csv')
        if not os.path.exists(raw_path):
            logger.info(f"Raw data not found at {raw_path}. Generating synthetic data for demonstration.")
            return self._generate_synthetic_data(data_path)

        df = pd.read_csv(raw_path)
        return self._build_sessions(df, data_path, user_col='user_id',
                                     item_col='item_id', time_col='timestamp',
                                     cat_col='category')

    def _process_cetailer(self, data_path: str):
        
        cache_path = os.path.join(data_path, 'processed_sessions.pkl')
        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        
        jsonl_path = os.path.join(data_path, 'cetailer.jsonl')
        if os.path.exists(jsonl_path):
            return self._load_jsonl(jsonl_path, data_path)

        raw_path = os.path.join(data_path, 'transactions.csv')
        if not os.path.exists(raw_path):
            logger.info(f"Raw data not found at {raw_path}. Generating synthetic data for demonstration.")
            return self._generate_synthetic_data(data_path)

        df = pd.read_csv(raw_path)
        return self._build_sessions(df, data_path, user_col='user_id',
                                     item_col='item_id', time_col='timestamp',
                                     cat_col='category')

    def _build_sessions(self, df: pd.DataFrame, data_path: str,
                         user_col: str, item_col: str, time_col: str,
                         cat_col: str = None):
        
        
        df = df.sort_values([user_col, time_col])

        
        unique_items = df[item_col].unique()
        self.item2idx = {item: idx + 1 for idx, item in enumerate(unique_items)}  # 0 for padding
        self.idx2item = {v: k for k, v in self.item2idx.items()}
        self.num_items = len(unique_items) + 1  # +1 for padding

        
        if cat_col and cat_col in df.columns:
            categories = df[[item_col, cat_col]].drop_duplicates()
            unique_cats = categories[cat_col].unique()
            cat2idx = {cat: idx for idx, cat in enumerate(unique_cats)}
            num_cats = len(unique_cats)

            self.item_features = np.zeros((self.num_items, num_cats), dtype=np.float32)
            for _, row in categories.iterrows():
                item_idx = self.item2idx[row[item_col]]
                cat_idx = cat2idx[row[cat_col]]
                self.item_features[item_idx, cat_idx] = 1.0
        else:
            self.item_features = np.random.randn(self.num_items, 64).astype(np.float32)

        
        sessions = []
        for user_id, group in df.groupby(user_col):
            items = [self.item2idx[it] for it in group[item_col].values]
            timestamps = group[time_col].values.tolist()

            if len(items) >= self.min_session_len:
                sessions.append({
                    'user_id': user_id,
                    'items': items,
                    'timestamps': timestamps,
                    'stb_labels': [2] * len(items),  # placeholder, will be computed later
                })

        self.sessions = sessions

        
        self._save_cache(data_path)

        logger.info(f"Loaded {len(sessions)} sessions, {self.num_items} items")
        return sessions, self.num_items, self.item_features

    def _generate_synthetic_data(self, data_path: str, n_users: int = 5000,
                                  n_items: int = 3000, n_cats: int = 50,
                                  avg_session_len: int = 20):
        os.makedirs(data_path, exist_ok=True)

        np.random.seed(42)
        self.num_items = n_items + 1  

        
        item_cats = np.random.randint(0, n_cats, n_items)
        self.item_features = np.zeros((self.num_items, n_cats), dtype=np.float32)
        for i in range(n_items):
            self.item_features[i + 1, item_cats[i]] = 1.0

        
        sessions = []
        for user_id in range(n_users):
            session_len = max(self.min_session_len,
                              np.random.poisson(avg_session_len))
            session_len = min(session_len, 100)

            
            preferred_cats = np.random.choice(n_cats, size=3, replace=False)
            items_in_preferred = [i + 1 for i in range(n_items) if item_cats[i] in preferred_cats]
            other_items = [i + 1 for i in range(n_items) if item_cats[i] not in preferred_cats]

            items = []
            for t in range(session_len):
                
                r = np.random.random()
                if r < 0.5 and items_in_preferred:
                    items.append(np.random.choice(items_in_preferred))
                elif r < 0.9 and other_items:
                    items.append(np.random.choice(other_items))
                else:
                    items.append(np.random.randint(1, n_items + 1))

            sessions.append({
                'user_id': user_id,
                'items': items,
                'timestamps': list(range(len(items))),
                'stb_labels': [2] * len(items),  # placeholder
            })

        self.sessions = sessions

        
        self._save_cache(data_path)

        return sessions, self.num_items, self.item_features

    def _save_cache(self, data_path: str):
        
        os.makedirs(data_path, exist_ok=True)
        cache = {
            'sessions': self.sessions,
            'num_items': self.num_items,
            'item_features': self.item_features,
            'item2idx': self.item2idx,
            'idx2item': self.idx2item,
        }
        with open(os.path.join(data_path, 'processed_sessions.pkl'), 'wb') as f:
            pickle.dump(cache, f)

    def _load_cache(self, cache_path: str):
        
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        self.sessions = cache['sessions']
        self.num_items = cache['num_items']
        self.item_features = cache['item_features']
        self.item2idx = cache['item2idx']
        self.idx2item = cache['idx2item']
        logger.info(f"Loaded cached data: {len(self.sessions)} sessions, {self.num_items} items")
        return self.sessions, self.num_items, self.item_features


def build_item_graph(sessions: List[dict], num_items: int) -> Tuple[np.ndarray, np.ndarray]:
    
    edge_count = defaultdict(int)
    for session in sessions:
        items = session['items']
        for i in range(1, len(items)):
            edge_count[(items[i - 1], items[i])] += 1

    if not edge_count:
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    edges = list(edge_count.keys())
    weights = [edge_count[e] for e in edges]

    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)

    return edge_index, edge_weight


def build_item_time_graph(sessions: List[dict], num_items: int) -> dict:
    item_time_edges = []
    time_set = set()

    for session in sessions:
        items = session['items']
        timestamps = session.get('timestamps', list(range(len(items))))

        for item, t in zip(items, timestamps):
            item_time_edges.append((item, t))
            time_set.add(t)

    
    time2idx = {t: idx for idx, t in enumerate(sorted(time_set))}
    num_time_nodes = len(time2idx)

    
    time_to_items = defaultdict(list)
    for item, t in item_time_edges:
        time_to_items[time2idx[t]].append(item)

    copurchased_edges = []
    for t, items in time_to_items.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                copurchased_edges.append((items[i], items[j]))
                copurchased_edges.append((items[j], items[i]))

    
    mapped_edges = [(item, time2idx[t]) for item, t in item_time_edges]

    return {
        'item_time_edges': mapped_edges,
        'num_time_nodes': num_time_nodes,
        'copurchased_edges': copurchased_edges,
        'time_to_items': dict(time_to_items),
    }


def kfold_split(sessions: List[dict], n_folds: int = 10, seed: int = 42):
    np.random.seed(seed)
    indices = np.random.permutation(len(sessions))
    fold_size = len(sessions) // n_folds

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(sessions)
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        train_sessions = [sessions[i] for i in train_idx]
        test_sessions = [sessions[i] for i in test_idx]
        yield train_sessions, test_sessions
