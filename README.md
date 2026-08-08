# UPSTAR

Implementation of paper "A User Purchase Motivation-Aware Product Recommender System"

> **A User Purchase Motivation-Aware Product Recommender System**.
> Jiarong Xu, Jiaan Wang, Hongzhe Zhang and Tian Lu.
> *Information Systems Research (ISR)*, 2026.
> DOI: [10.1287/isre.2024.1028](https://doi.org/10.1287/isre.2024.1028)


Retailers struggle to align *what* they recommend with *why* customers buy. This work builds a multi-level framework of product purchase motivations and identifies two actionable motivations rooted in users' inherent interests: **stable preference** and **exploratory intent**.

To operationalize them, we propose **STB**, a data-efficient measure that infers which motivation drives each purchase using only transaction sequences and product attributes (no surveys or auxiliary data). Building on STB, we develop **UPSTAR**, a motivation-aware recommender that separates a user's behavior into stable-preference and exploratory subsequences, models them with dedicated encoders, and fuses their signals for next-item prediction. Experiments on three real-world e-commerce datasets (**Ta-Feng**, **Cetailer**, **IJCAI-15**) show that UPSTAR improves recommendation accuracy and substantially strengthens the system's ability to surface genuinely exploratory products, promoting product discovery and cross-category sales.

## Method Overview

```
transactions ──▶ STB estimator ──▶ per-item motivation labels {stable / exploratory / other}
                                             │
item co-purchase graph ──▶ Item GNN ──▶ item representations
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     │  S-model / E-model / O-model (+ global LSTM)  │  motivation-specific
                     └───────────────────────┬───────────────────────┘  sequence encoders
                                ▼
                                       Fusion Gate  ──▶ next-item scores
                                             ▲
                        Dual Teacher-Student loss (S & E teachers → global student)
```

## Installation

```bash
git clone https://github.com/JiaanWang-academic/UPSTAR && cd UPSTAR
pip install -r requirements.txt
```

Requires Python >= 3.9, PyTorch >= 2.0 and PyTorch Geometric >= 2.3 (CUDA / MPS / CPU all supported).

## Data Preparation

Put the data as a JSONL file at `data/<dataset>/<dataset>.jsonl` (e.g. `data/tafeng/tafeng.jsonl`),
one user per line:

```json
{"user": "u1", "session": [{"item": "p1", "time": 0}, {"item": "p2", "time": 0}, {"item": "p3", "time": 5}]}
```

`item` is the product id and `time` is the (integer) transaction day/timestamp — items sharing the
same `time` are treated as one basket. Sessions shorter than `--min_session_len` are dropped, item
ids are re-indexed (0 reserved for padding), and initial item features are built from co-occurrence
statistics via truncated SVD. The preprocessed result is cached as `data/<dataset>/processed_sessions.pkl`.

For Ta-Feng, `ta_feng_all_months_merged.csv` (see `--category_csv`) is additionally used to attach
`PRODUCT_SUBCLASS` category embeddings; use `--no_category` to disable it.

## Usage

```bash
# Default: Ta-Feng, repurchase-based STB, 10-fold CV
python train.py

# Other datasets / STB variants
python train.py --dataset ijcai15 --stb_method consistency
python train.py --dataset cetailer --recompute_stb

# Paper-faithful configuration
python train.py --lr 3e-4 --lstm_layers 4 --tie_output --cosine
```

STB labels are cached at
`data/<dataset>/stb_sessions_<method>_rho<rho>_beta<beta>.pkl`; pass `--recompute_stb` to refresh
them. Checkpoints are written to `--save_dir` as `model_fold{k}.pt`.


## File Structure

```
UPSTAR/
├── train.py                  # Training loop (k-fold CV, no evaluation)
├── models/
│   ├── stb_estimator.py      # STB measure & purchase motivation identification
│   ├── item_gnn.py           # In/cross-session item representation learning
│   └── upstar.py             #  S/E/O models, Fusion Gate, Dual Teacher-Student Loss
├── data/
│   └── dataset.py            # Data loading, session construction, sequence augmentation
├── utils/
│   └── config.py             # Hyperparameter configuration
└── requirements.txt          # Python dependencies
```


## Citation

```bibtex
@article{xu2026upstar,
  title   = {A User Purchase Motivation-Aware Product Recommender System},
  author  = {Xu, Jiarong and Wang, Jiaan and Zhang, Hongzhe and others},
  journal = {Information Systems Research},
  year    = {2026},
  doi     = {10.1287/isre.2024.1028}
}
```
