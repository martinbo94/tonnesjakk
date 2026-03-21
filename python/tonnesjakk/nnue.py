"""
NNUE Training for Tonnesjakk

HalfPail NNUE training pipeline for Tonnesjakk. Uses dual-perspective sparse
features inspired by Stockfish's HalfKP architecture.

Key features:
- Random opening moves for diversity (prevents all games being identical)
- Outcome-based labeling with temporal discounting
- HalfPail dual-perspective sparse feature encoding
- Export to JSON for Rust inference
- Balanced dataset validation

Usage:
    python -m tonnesjakk.nnue --load-data data.bin --epochs 50  # Train on data
    python -m tonnesjakk.nnue --games 10000 --save-data data.bin  # Generate data
    python -m tonnesjakk.nnue --compare a.json heuristic --depth 6  # Compare
"""

import json
import math
import multiprocessing
import random
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Try to import Rust-accelerated HalfPail decoders
try:
    from tonnesjakk import decode_halfpail as _rust_decode_halfpail
    from tonnesjakk import decode_halfpail_batch as _rust_decode_batch
except ImportError:
    _rust_decode_halfpail = None
    _rust_decode_batch = None

# Constants
BOARD_SIZE = 6
NUM_PIECE_TYPES = 4  # WhiteBarrel, BlackBarrel, WhitePail, BlackPail

# Feature sizes
BASE_FEATURES = BOARD_SIZE * BOARD_SIZE * NUM_PIECE_TYPES  # 144 (piece positions)
# Relational features (20 total):
#   [0-3]  White barrel distances to goal (4 values, normalized 0-1, closest first)
#   [4-7]  Black barrel distances to goal (4 values, normalized 0-1, closest first)
#   [8]    White barrels scored (normalized 0-1)
#   [9]    Black barrels scored (normalized 0-1)
#   [10]   White pail placed (0 or 1)
#   [11]   Black pail placed (0 or 1)
#   [12]   Current player (+1 white, -1 black)
#   [13]   White immediate threats (barrels 1 step from scoring, /4)
#   [14]   Black immediate threats (barrels 1 step from scoring, /4)
#   [15]   Score differential (white_scored - black_scored) / 4, range -1 to +1
#   [16]   White barrels on board / 4
#   [17]   Black barrels on board / 4
#   [18]   White pail blocking count / 4
#   [19]   Black pail blocking count / 4
RELATIONAL_FEATURES = 20
INPUT_SIZE = BASE_FEATURES + RELATIONAL_FEATURES  # 164

# Score normalization: use tanh(score / SCORE_SCALING) instead of linear clip.
# This prevents information loss for extreme scores.
# 600 means: +600cp → tanh(1) ≈ 0.76, +1200cp → 0.96, +3000cp → 1.00
SCORE_SCALING = 600.0

# HalfPail feature architecture constants
# Inspired by Stockfish's HalfKP: piece positions contextualized by pail position.
# Each perspective sees: bucket (pail position) × square × piece_type_reduced
HALFPAIL_BUCKETS = 37       # 36 pail squares + 1 for "no pail placed"
HALFPAIL_PIECE_TYPES = 3    # 0=friendly barrel, 1=enemy barrel, 2=enemy pail
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE  # 36
HALFPAIL_FEATURES_PER_BUCKET = NUM_SQUARES * HALFPAIL_PIECE_TYPES  # 36 * 3 = 108
HALFPAIL_FEATURES = HALFPAIL_BUCKETS * HALFPAIL_FEATURES_PER_BUCKET  # 37 * 108 = 3996
HALFPAIL_DENSE = 20         # Dense features: all 20 relational features from training data


# =============================================================================
# HalfPail Feature Computation
# =============================================================================

def decode_board_from_dense164(row: np.ndarray) -> dict:
    """Decode piece positions and game state from a 164-feature dense row.

    The first 144 features are one-hot piece positions (36 squares × 4 piece types).
    The remaining 20 are relational features from which we extract scored counts,
    pail placement flags, current player, and barrels on board.

    Returns:
        dict with keys: white_barrels (list of sq), black_barrels, white_pail (sq or None),
        black_pail (sq or None), white_scored, black_scored, current_player (+1/-1),
        white_barrels_on_board, black_barrels_on_board
    """
    base = row[:144].reshape(36, 4)

    white_barrels = []
    black_barrels = []
    white_pail = None
    black_pail = None

    for sq in range(36):
        if base[sq, 0] > 0.5:
            white_barrels.append(sq)
        if base[sq, 1] > 0.5:
            black_barrels.append(sq)
        if base[sq, 2] > 0.5:
            white_pail = sq
        if base[sq, 3] > 0.5:
            black_pail = sq

    # Extract from relational features (indices 144+)
    rel = row[144:]
    white_scored = round(rel[8] * 4.0)   # feature[8] = white_scored / 4
    black_scored = round(rel[9] * 4.0)   # feature[9] = black_scored / 4
    current_player = 1.0 if rel[12] > 0 else -1.0  # feature[12] = +1 or -1

    return {
        'white_barrels': white_barrels,
        'black_barrels': black_barrels,
        'white_pail': white_pail,
        'black_pail': black_pail,
        'white_scored': int(white_scored),
        'black_scored': int(black_scored),
        'current_player': current_player,
        'white_barrels_on_board': len(white_barrels),
        'black_barrels_on_board': len(black_barrels),
        'rel': rel,  # all 20 relational features for HalfPail dense input
    }


def halfpail_feature_index(bucket: int, sq: int, piece_type: int) -> int:
    """Compute HalfPail sparse feature index.

    Args:
        bucket: pail square (0-35) or 36 if no pail placed
        sq: piece square (0-35)
        piece_type: 0=friendly barrel, 1=enemy barrel, 2=enemy pail

    Returns:
        Index in range [0, 3996)
    """
    return bucket * HALFPAIL_FEATURES_PER_BUCKET + sq * HALFPAIL_PIECE_TYPES + piece_type


def board_to_halfpail_indices(board_dict: dict) -> tuple:
    """Compute HalfPail sparse feature indices for both perspectives.

    For each perspective (white/black):
    - bucket = own pail square (or 36 if not placed)
    - Encode all pieces EXCEPT own pail:
        Type 0: own barrels ("friendly")
        Type 1: opponent's barrels ("enemy")
        Type 2: opponent's pail ("enemy pail")

    Also returns the 20 dense relational features.

    Returns:
        (white_indices, black_indices, dense_6)
        where indices are lists of ints in [0, 3996)
        and dense_6 is a list of 6 floats
    """
    wb = board_dict['white_barrels']
    bb = board_dict['black_barrels']
    wp = board_dict['white_pail']
    bp = board_dict['black_pail']
    ws = board_dict['white_scored']
    bs = board_dict['black_scored']

    # White perspective: bucket = white pail position
    w_bucket = wp if wp is not None else 36
    white_indices = []
    for sq in wb:
        white_indices.append(halfpail_feature_index(w_bucket, sq, 0))  # friendly barrel
    for sq in bb:
        white_indices.append(halfpail_feature_index(w_bucket, sq, 1))  # enemy barrel
    if bp is not None:
        white_indices.append(halfpail_feature_index(w_bucket, bp, 2))  # enemy pail

    # Black perspective: bucket = black pail position
    b_bucket = bp if bp is not None else 36
    black_indices = []
    for sq in bb:
        black_indices.append(halfpail_feature_index(b_bucket, sq, 0))  # friendly barrel
    for sq in wb:
        black_indices.append(halfpail_feature_index(b_bucket, sq, 1))  # enemy barrel
    if wp is not None:
        black_indices.append(halfpail_feature_index(b_bucket, wp, 2))  # enemy pail

    # Dense features: all 20 relational features from the 164-dim row
    if 'rel' in board_dict:
        dense = board_dict['rel'].tolist() if hasattr(board_dict['rel'], 'tolist') else list(board_dict['rel'])
    else:
        # Fallback: compute the 6 basic features, pad to 20
        dense = [0.0] * HALFPAIL_DENSE
        dense[0] = 1.0 - min(wb[0], 5) / 5.0 if wb else 0.0  # approximate barrel distances
        dense[4] = 1.0 - min(bb[0], 5) / 5.0 if bb else 0.0
        dense[8] = ws / 4.0
        dense[9] = bs / 4.0
        dense[10] = 1.0 if wp is not None else 0.0
        dense[11] = 1.0 if bp is not None else 0.0
        dense[12] = board_dict['current_player']
        dense[15] = (ws - bs) / 4.0
        dense[16] = board_dict['white_barrels_on_board'] / 4.0
        dense[17] = board_dict['black_barrels_on_board'] / 4.0

    return white_indices, black_indices, dense


def test_halfpail_decoding(n_positions: int = 100):
    """Round-trip test: generate positions → encode 164 → decode → compute HalfPail indices.

    Verifies that board decoding and index computation work correctly.
    """
    from tonnesjakk import Board, Engine
    import random

    print(f"Testing HalfPail decoding on {n_positions} positions...")
    engine = Engine()
    errors = 0

    for i in range(n_positions):
        board = Board()
        # Play random moves to get diverse positions
        for _ in range(random.randint(0, 20)):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        if board.check_winner():
            continue

        # Encode to 164-dim tensor
        board_array = board.to_array()
        is_white = "White" in repr(board.current_player)
        current_player = 1 if is_white else -1
        tensor = board_to_tensor(
            board_array,
            white_scored=board.white_scored,
            black_scored=board.black_scored,
            current_player=current_player
        )
        row = tensor.numpy()

        # Decode back
        decoded = decode_board_from_dense164(row)

        # Verify piece positions match
        expected_wb = []
        expected_bb_list = []
        expected_wp = None
        expected_bp = None
        for r in range(6):
            for c in range(6):
                sq = r * 6 + c
                val = board_array[r][c]
                if val == 1:
                    expected_wb.append(sq)
                elif val == -1:
                    expected_bb_list.append(sq)
                elif val == 2:
                    expected_wp = sq
                elif val == -2:
                    expected_bp = sq

        if sorted(decoded['white_barrels']) != sorted(expected_wb):
            print(f"  ERROR at pos {i}: white barrels mismatch")
            errors += 1
            continue
        if sorted(decoded['black_barrels']) != sorted(expected_bb_list):
            print(f"  ERROR at pos {i}: black barrels mismatch")
            errors += 1
            continue
        if decoded['white_pail'] != expected_wp:
            print(f"  ERROR at pos {i}: white pail mismatch")
            errors += 1
            continue
        if decoded['black_pail'] != expected_bp:
            print(f"  ERROR at pos {i}: black pail mismatch")
            errors += 1
            continue

        # Compute HalfPail indices
        w_idx, b_idx, dense = board_to_halfpail_indices(decoded)

        # Verify index ranges
        for idx in w_idx + b_idx:
            if idx < 0 or idx >= HALFPAIL_FEATURES:
                print(f"  ERROR at pos {i}: index {idx} out of range [0, {HALFPAIL_FEATURES})")
                errors += 1
                break

        # Verify dense features
        if len(dense) != HALFPAIL_DENSE:
            print(f"  ERROR at pos {i}: dense has {len(dense)} features, expected {HALFPAIL_DENSE}")
            errors += 1

        # Verify active feature count is reasonable (barrels + pails on board)
        n_pieces = len(decoded['white_barrels']) + len(decoded['black_barrels'])
        if decoded['white_pail'] is not None:
            n_pieces += 1
        if decoded['black_pail'] is not None:
            n_pieces += 1
        # Each perspective encodes all pieces except own pail
        # White perspective: own barrels + enemy barrels + enemy pail (if placed)
        expected_w = len(decoded['white_barrels']) + len(decoded['black_barrels'])
        if decoded['black_pail'] is not None:
            expected_w += 1
        if len(w_idx) != expected_w:
            print(f"  ERROR at pos {i}: white has {len(w_idx)} indices, expected {expected_w}")
            errors += 1

    if errors == 0:
        print(f"  All {n_positions} positions passed!")
    else:
        print(f"  {errors} errors found!")
    return errors == 0


# =============================================================================
# HalfPail Neural Network
# =============================================================================

class HalfPailNNUE(nn.Module):
    """
    HalfPail NNUE with dual-perspective sparse features.

    Architecture:
        White sparse indices → EmbeddingBag(3996, H1, sum) + bias → ReLU → acc_white
        Black sparse indices → EmbeddingBag(3996, H1, sum) + bias → ReLU → acc_black
                                    (shared weights)
        concat(acc_white, acc_black, dense_20) → FC2(2*H1+20, H2) → ReLU → FC3(H2, 1) → Tanh
    """

    def __init__(self, hidden1: int = 128, hidden2: int = 32):
        super().__init__()
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_perspective_features = HALFPAIL_FEATURES  # 3996
        self.dense_size = HALFPAIL_DENSE  # 20

        # Shared perspective embedding (FC1 equivalent)
        # EmbeddingBag with mode='sum' acts like sparse matrix multiplication
        self.embedding = nn.EmbeddingBag(
            num_embeddings=HALFPAIL_FEATURES,
            embedding_dim=hidden1,
            mode='sum',
        )
        self.fc1_bias = nn.Parameter(torch.zeros(hidden1))

        # FC2: takes concat of both accumulators + dense features
        fc2_input = 2 * hidden1 + HALFPAIL_DENSE
        self.fc2 = nn.Linear(fc2_input, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

        # Initialize
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, white_indices, white_offsets, black_indices, black_offsets, dense):
        """
        Forward pass with packed EmbeddingBag inputs.

        Args:
            white_indices: packed 1D tensor of white perspective feature indices
            white_offsets: start offset for each sample in white_indices
            black_indices: packed 1D tensor of black perspective feature indices
            black_offsets: start offset for each sample in black_indices
            dense: [batch_size, 20] dense features
        """
        # Shared embedding for both perspectives
        acc_white = self.embedding(white_indices, white_offsets) + self.fc1_bias
        acc_white = torch.relu(acc_white)

        acc_black = self.embedding(black_indices, black_offsets) + self.fc1_bias
        acc_black = torch.relu(acc_black)

        # Concatenate: [acc_white, acc_black, dense]
        combined = torch.cat([acc_white, acc_black, dense], dim=1)

        # FC2 → ReLU → FC3 → Tanh
        h = torch.relu(self.fc2(combined))
        out = torch.tanh(self.fc3(h))
        return out

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HalfPailDataset(torch.utils.data.Dataset):
    """Dataset that decodes 164-dim dense features to HalfPail sparse indices on-the-fly.

    Wraps existing memmap or numpy array of shape (N, 164) with labels (N,).
    Supports multi-worker DataLoader on Windows (spawn) by lazily re-opening
    memmap files in each worker process instead of pickling the arrays.
    """

    def __init__(self, X, y, start: int = 0, end: int = None):
        """
        Args:
            X: features array (N, 164) - numpy array or memmap (full, unsliced)
            y: labels array (N,) or (N, 1)
            start: start index for this subset
            end: end index for this subset (None = len(X))
        """
        self._start = start
        self._end = end if end is not None else len(X)
        self.n = self._end - self._start
        # Store memmap metadata for lazy re-opening in worker processes
        if isinstance(X, np.memmap):
            self._x_path = X.filename
            self._x_shape = (len(X), X.shape[1]) if X.ndim == 2 else (len(X),)
            self._y_path = y.filename
            self._y_shape = y.shape
            self._is_memmap = True
            self.X = None
            self.y = None
        else:
            self._is_memmap = False
            self.X = X
            self.y = y

    def __getstate__(self):
        """Exclude memmap arrays from pickling (for Windows spawn workers)."""
        state = self.__dict__.copy()
        if self._is_memmap:
            state['X'] = None
            state['y'] = None
        return state

    def _ensure_open(self):
        """Lazily re-open memmap files (called in worker processes)."""
        if self._is_memmap and self.X is None:
            self.X = np.memmap(self._x_path, dtype=np.float32, mode='r', shape=self._x_shape)
            self.y = np.memmap(self._y_path, dtype=np.float32, mode='r', shape=self._y_shape)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = self._start + idx
        row = np.array(self.X[real_idx], dtype=np.float32)
        label = float(self.y[real_idx]) if self.y.ndim == 1 else float(self.y[real_idx, 0])

        if _rust_decode_halfpail is not None:
            w_idx, b_idx, dense = _rust_decode_halfpail(row.tolist())
        else:
            board = decode_board_from_dense164(row)
            w_idx, b_idx, dense = board_to_halfpail_indices(board)

        return (
            torch.tensor(w_idx, dtype=torch.long),
            torch.tensor(b_idx, dtype=torch.long),
            torch.tensor(dense, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
        )


def halfpail_collate_fn(batch):
    """Custom collate function for HalfPailDataset.

    Packs variable-length index lists into the format EmbeddingBag expects:
    a single 1D tensor of indices and a 1D tensor of offsets.
    """
    all_w_idx = []
    all_b_idx = []
    w_offsets = [0]
    b_offsets = [0]
    dense_list = []
    label_list = []

    for w_idx, b_idx, dense, label in batch:
        all_w_idx.append(w_idx)
        all_b_idx.append(b_idx)
        w_offsets.append(w_offsets[-1] + len(w_idx))
        b_offsets.append(b_offsets[-1] + len(b_idx))
        dense_list.append(dense)
        label_list.append(label)

    white_indices = torch.cat(all_w_idx) if all_w_idx else torch.tensor([], dtype=torch.long)
    black_indices = torch.cat(all_b_idx) if all_b_idx else torch.tensor([], dtype=torch.long)
    white_offsets = torch.tensor(w_offsets[:-1], dtype=torch.long)
    black_offsets = torch.tensor(b_offsets[:-1], dtype=torch.long)
    dense = torch.stack(dense_list)
    labels = torch.stack(label_list)

    return white_indices, white_offsets, black_indices, black_offsets, dense, labels


class ChunkedBatchSampler(torch.utils.data.Sampler):
    """Yields batches of contiguous indices with shuffled chunk order.

    Instead of globally shuffling indices (terrible for memmap I/O),
    divides the dataset into large contiguous chunks, shuffles the chunk order,
    and yields batch_size-sized slices from each chunk sequentially.
    """

    def __init__(self, dataset_size: int, batch_size: int, chunk_size: int, shuffle: bool = True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle

    def __iter__(self):
        n_chunks = (self.dataset_size + self.chunk_size - 1) // self.chunk_size
        chunk_order = np.arange(n_chunks)
        if self.shuffle:
            np.random.shuffle(chunk_order)

        for ci in chunk_order:
            chunk_start = ci * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, self.dataset_size)
            # Yield batch_size slices of contiguous indices within this chunk
            for bs in range(chunk_start, chunk_end, self.batch_size):
                yield list(range(bs, min(bs + self.batch_size, chunk_end)))

    def __len__(self):
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def _rust_batch_decode_chunk(X_chunk, y_chunk):
    """Decode a contiguous chunk of rows using Rust batch decoder.

    Returns tensors ready for model forward pass, same format as halfpail_collate_fn.
    """
    # Convert to flat Python lists for Rust extraction (fastest measured path)
    flat_x = np.ascontiguousarray(X_chunk, dtype=np.float32).ravel().tolist()
    flat_y = np.ascontiguousarray(y_chunk, dtype=np.float32).ravel().tolist()

    w_idx, w_off, b_idx, b_off, dense_flat, labels_flat = _rust_decode_batch(flat_x, flat_y)

    n = len(w_off)
    return (
        torch.tensor(w_idx, dtype=torch.long),
        torch.tensor(w_off, dtype=torch.long),
        torch.tensor(b_idx, dtype=torch.long),
        torch.tensor(b_off, dtype=torch.long),
        torch.tensor(dense_flat, dtype=torch.float32).view(n, HALFPAIL_DENSE),
        torch.tensor(labels_flat, dtype=torch.float32).unsqueeze(1),
    )


def train_halfpail_model(
    X,  # np.ndarray or np.memmap, shape (N, 164)
    y,  # np.ndarray or np.memmap, shape (N,), (N, 1), or (N, 2) [search_score, outcome]
    hidden1: int = 128,
    hidden2: int = 32,
    epochs: int = 200,
    batch_size: int = 4096,
    learning_rate: float = 0.003,
    validation_split: float = 0.1,
    verbose: bool = True,
    loss_fn: str = "wdl-ce",
    num_workers: int = 0,
    resume_from: Optional[str] = None,
    lambda_blend: Optional[float] = None,
) -> tuple:
    """Train a HalfPail NNUE model.

    Uses Rust batch decoding for fast on-the-fly feature computation.
    Falls back to Python DataLoader if Rust batch decoder is unavailable.

    If y has 2 columns [search_score, outcome], lambda_blend controls the mix:
      label = lambda * search_score + (1 - lambda) * outcome
    Default lambda_blend=1.0 uses pure search scores (backward compatible).

    Returns:
        (model, history)
    """
    n = len(X)
    split = int((1 - validation_split) * n)
    train_n = split

    # Handle 2-column labels: blend search_score and outcome at training time
    if y.ndim == 2 and y.shape[1] == 2:
        lb = lambda_blend if lambda_blend is not None else 1.0
        y_flat = (lb * y[:, 0] + (1 - lb) * y[:, 1]).astype(np.float32)
        if verbose:
            print(f"  Label blend: {lb:.2f} * search_score + {1-lb:.2f} * outcome")
    else:
        y_flat = y.ravel() if y.ndim > 1 else y

    use_rust_batch = _rust_decode_batch is not None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = HalfPailNNUE(hidden1, hidden2)
    if resume_from:
        state = torch.load(resume_from, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        if verbose:
            print(f"  Resumed from: {resume_from}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine annealing: LR decays from initial to ~0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    if loss_fn == "wdl-ce":
        criterion = wdl_cross_entropy
    else:
        criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None

    if verbose:
        fc2_input = 2 * hidden1 + HALFPAIL_DENSE
        print(f"  Training set: {train_n:,} positions")
        print(f"  Validation set: {n - split:,} positions")
        mode = "Rust batch decode (contiguous reads)" if use_rust_batch else "Python DataLoader"
        print(f"  Mode: HalfPail dual-perspective sparse features")
        print(f"  Decoder: {mode}")
        print(f"  Device: {device}")
        print(f"  Loss: {loss_fn}")
        print(f"  LR: {learning_rate} (cosine annealing -> {learning_rate * 0.01:.6f})")
        print(f"  Architecture: EmbeddingBag({HALFPAIL_FEATURES}, {hidden1}) shared")
        print(f"    FC2: {fc2_input} -> {hidden2}, FC3: {hidden2} -> 1")
        print(f"  Parameters: {model.num_parameters:,}")

    if use_rust_batch:
        # Fast path: contiguous chunk reads + Rust batch decode (no DataLoader needed)
        n_chunks = (train_n + batch_size - 1) // batch_size
        chunk_order = np.arange(n_chunks)

        val_n = n - split
        val_n_chunks = (val_n + batch_size - 1) // batch_size

        train_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            np.random.shuffle(chunk_order)  # shuffle chunk ORDER, not indices
            total_loss = 0.0
            num_batches = 0

            for ci in chunk_order:
                start = ci * batch_size
                end = min(start + batch_size, train_n)
                # Contiguous read from memmap + Rust batch decode
                w_idx, w_off, b_idx, b_off, dense, labels = _rust_batch_decode_chunk(
                    X[start:end], y_flat[start:end]
                )
                w_idx, w_off = w_idx.to(device), w_off.to(device)
                b_idx, b_off = b_idx.to(device), b_off.to(device)
                dense, labels = dense.to(device), labels.to(device)

                optimizer.zero_grad()
                pred = model(w_idx, w_off, b_idx, b_off, dense)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            train_loss = total_loss / max(num_batches, 1)

            # Validation (chunked to avoid OOM)
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for vi in range(val_n_chunks):
                    vs = split + vi * batch_size
                    ve = min(vs + batch_size, n)
                    w_idx, w_off, b_idx, b_off, dense, labels = _rust_batch_decode_chunk(
                        X[vs:ve], y_flat[vs:ve]
                    )
                    w_idx, w_off = w_idx.to(device), w_off.to(device)
                    b_idx, b_off = b_idx.to(device), b_off.to(device)
                    dense, labels = dense.to(device), labels.to(device)
                    pred = model(w_idx, w_off, b_idx, b_off, dense)
                    val_loss_total += criterion(pred, labels).item() * (ve - vs)
                    val_batches += (ve - vs)
            val_loss = val_loss_total / max(val_batches, 1)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " *"
            else:
                marker = ""

            scheduler.step()

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start
                elapsed = time.time() - train_start
                eta = epoch_time * (epochs - epoch - 1)
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"train={train_loss:.4f}, val={val_loss:.4f} "
                      f"lr={current_lr:.6f} "
                      f"({epoch_time:.0f}s, total {elapsed:.0f}s, ETA {eta/60:.0f}m){marker}", flush=True)
    else:
        # Fallback: Python DataLoader with per-sample decode
        train_dataset = HalfPailDataset(X, y, start=0, end=split)
        val_dataset = HalfPailDataset(X, y, start=split, end=n)

        chunk_size = batch_size * max(num_workers, 1) * 4
        train_sampler = ChunkedBatchSampler(len(train_dataset), batch_size, chunk_size, shuffle=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=halfpail_collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=halfpail_collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        train_start = time.time()
        for epoch in range(epochs):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            num_batches = 0

            for w_idx, w_off, b_idx, b_off, dense, labels in train_loader:
                w_idx, w_off = w_idx.to(device), w_off.to(device)
                b_idx, b_off = b_idx.to(device), b_off.to(device)
                dense, labels = dense.to(device), labels.to(device)
                optimizer.zero_grad()
                pred = model(w_idx, w_off, b_idx, b_off, dense)
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            train_loss = total_loss / max(num_batches, 1)

            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for w_idx, w_off, b_idx, b_off, dense, labels in val_loader:
                    w_idx, w_off = w_idx.to(device), w_off.to(device)
                    b_idx, b_off = b_idx.to(device), b_off.to(device)
                    dense, labels = dense.to(device), labels.to(device)
                    pred = model(w_idx, w_off, b_idx, b_off, dense)
                    val_loss_total += criterion(pred, labels).item()
                    val_batches += 1

            val_loss = val_loss_total / max(val_batches, 1)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                marker = " *"
            else:
                marker = ""

            scheduler.step()

            if verbose:
                current_lr = optimizer.param_groups[0]['lr']
                epoch_time = time.time() - epoch_start
                elapsed = time.time() - train_start
                eta = epoch_time * (epochs - epoch - 1)
                print(f"  Epoch {epoch+1:3d}/{epochs}: "
                      f"train={train_loss:.4f}, val={val_loss:.4f} "
                      f"lr={current_lr:.6f} "
                      f"({epoch_time:.0f}s, total {elapsed:.0f}s, ETA {eta/60:.0f}m){marker}", flush=True)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose:
        print(f"  Best validation loss: {best_val_loss:.4f}")

    return model, history


def export_halfpail_json(model: HalfPailNNUE, output_path: str):
    """Export HalfPail NNUE weights to JSON for Rust inference.

    The JSON includes a "halfpail": true marker so the Rust engine
    can detect the model format and load accordingly.
    """
    state = model.state_dict()

    weights = {
        "fc1_weight": state['embedding.weight'].tolist(),  # [3996][H1]
        "fc1_bias": state['fc1_bias'].tolist(),             # [H1]
        "fc2_weight": state['fc2.weight'].tolist(),         # [H2][2*H1+20]
        "fc2_bias": state['fc2.bias'].tolist(),             # [H2]
        "fc3_weight": state['fc3.weight'].tolist(),         # [1][H2]
        "fc3_bias": state['fc3.bias'].tolist(),             # [1]
    }

    output = {
        "halfpail": True,
        "hidden1": model.hidden1,
        "hidden2": model.hidden2,
        "num_perspective_features": model.num_perspective_features,
        "dense_size": model.dense_size,
        "weights": weights,
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    file_size = Path(output_path).stat().st_size
    print(f"Exported HalfPail model to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


# =============================================================================
# Feature Encoding (164-dim dense format for data generation)
# =============================================================================

def board_to_tensor(
    board_array: List[List[int]],
    white_scored: int = 0,
    black_scored: int = 0,
    current_player: int = 1  # 1 = white, -1 = black
) -> torch.Tensor:
    """
    Convert board array to neural network input tensor with relational features.

    Base features (144): One-hot encoding per square
    - Channel 0: White barrel
    - Channel 1: Black barrel
    - Channel 2: White pail
    - Channel 3: Black pail

    Relational features (20):
    - [0-3]  White barrel distances to goal (normalized 0-1, closest first)
    - [4-7]  Black barrel distances to goal (normalized 0-1, closest first)
    - [8]    White barrels scored (normalized 0-1)
    - [9]    Black barrels scored (normalized 0-1)
    - [10]   White pail placed (0 or 1)
    - [11]   Black pail placed (0 or 1)
    - [12]   Current player (+1 white, -1 black)
    - [13]   White immediate threats (barrels 1 step from scoring, /4)
    - [14]   Black immediate threats (barrels 1 step from scoring, /4)
    - [15]   Score differential (white_scored - black_scored) / 4
    - [16]   White barrels on board / 4
    - [17]   Black barrels on board / 4
    - [18]   White pail blocking count / 4
    - [19]   Black pail blocking count / 4
    """
    # Base features: piece positions
    base = np.zeros((BOARD_SIZE, BOARD_SIZE, NUM_PIECE_TYPES), dtype=np.float32)

    # Track barrel positions for distance and blocking calculation
    white_barrel_positions = []  # (row, col)
    black_barrel_positions = []  # (row, col)
    white_pail_pos = None
    black_pail_pos = None

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            val = board_array[row][col]
            if val == 1:    # WhiteBarrel
                base[row, col, 0] = 1.0
                white_barrel_positions.append((row, col))
            elif val == -1:  # BlackBarrel
                base[row, col, 1] = 1.0
                black_barrel_positions.append((row, col))
            elif val == 2:   # WhitePail
                base[row, col, 2] = 1.0
                white_pail_pos = (row, col)
            elif val == -2:  # BlackPail
                base[row, col, 3] = 1.0
                black_pail_pos = (row, col)

    # Relational features (20 total)
    relational = np.zeros(RELATIONAL_FEATURES, dtype=np.float32)

    # White barrel distances to goal (row 0 is goal, so distance = row)
    white_dists = sorted([r / 5.0 for r, c in white_barrel_positions])
    for i, d in enumerate(white_dists[:4]):
        relational[i] = 1.0 - d  # Closer to goal = higher value

    # Black barrel distances to goal (row 5 is goal, so distance = 5 - row)
    black_dists = sorted([(5 - r) / 5.0 for r, c in black_barrel_positions])
    for i, d in enumerate(black_dists[:4]):
        relational[4 + i] = 1.0 - d

    # Scored barrels (normalized by 4, which is max)
    relational[8] = white_scored / 4.0
    relational[9] = black_scored / 4.0

    # Pails placed
    relational[10] = 1.0 if white_pail_pos is not None else 0.0
    relational[11] = 1.0 if black_pail_pos is not None else 0.0

    # Current player
    relational[12] = current_player

    # Immediate threats (barrels 1 step from scoring)
    white_threats = sum(1 for r, c in white_barrel_positions if r == 1)
    black_threats = sum(1 for r, c in black_barrel_positions if r == 4)
    relational[13] = white_threats / 4.0
    relational[14] = black_threats / 4.0

    # Score differential
    relational[15] = (white_scored - black_scored) / 4.0

    # Barrels on board
    relational[16] = len(white_barrel_positions) / 4.0
    relational[17] = len(black_barrel_positions) / 4.0

    # Pail blocking counts
    white_pail_blocks = 0
    if white_pail_pos is not None:
        pr, pc = white_pail_pos
        for br, bc in black_barrel_positions:
            if pc == bc and pr > br:  # Pail ahead of black barrel (blocking toward row 5)
                white_pail_blocks += 1
    relational[18] = white_pail_blocks / 4.0

    black_pail_blocks = 0
    if black_pail_pos is not None:
        pr, pc = black_pail_pos
        for br, bc in white_barrel_positions:
            if pc == bc and pr < br:  # Pail ahead of white barrel (blocking toward row 0)
                black_pail_blocks += 1
    relational[19] = black_pail_blocks / 4.0

    # Combine base and relational features
    features = np.concatenate([base.flatten(), relational])
    return torch.from_numpy(features)


def flip_board_horizontal(board_array: List[List[int]]) -> List[List[int]]:
    """Flip board horizontally for data augmentation (exploits symmetry)."""
    return [[board_array[r][5 - c] for c in range(6)] for r in range(6)]


# =============================================================================
# Data Generation
# =============================================================================

@dataclass
class PositionData:
    """Training data for a single position."""
    board: List[List[int]]
    search_score: float  # From engine search (normalized to -1 to +1)
    white_scored: int    # Number of white barrels scored
    black_scored: int    # Number of black barrels scored
    current_player: int  # 1 = white, -1 = black


@dataclass
class GameResult:
    """Result of a single self-play game."""
    positions: List[PositionData]  # Rich position data with search scores
    outcome: float  # +1 white wins, -1 black wins, 0 draw
    num_moves: int


@dataclass
class TrainingStats:
    """Statistics from training data generation."""
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_positions: int = 0

    @property
    def total_games(self) -> int:
        return self.white_wins + self.black_wins + self.draws

    @property
    def balance_ratio(self) -> float:
        """Ratio of white wins to black wins (1.0 = perfectly balanced)."""
        if self.black_wins == 0:
            return float('inf')
        return self.white_wins / self.black_wins

    def __str__(self) -> str:
        total = self.total_games
        if total == 0:
            return "No games played"
        return (f"W:{self.white_wins} ({100*self.white_wins/total:.1f}%) "
                f"B:{self.black_wins} ({100*self.black_wins/total:.1f}%) "
                f"D:{self.draws} ({100*self.draws/total:.1f}%)")


class DataGenerator:
    """
    Generates training data via self-play.

    Key features:
    - Random opening moves for diversity
    - Can use NNUE or heuristic evaluation
    - Filters out near-terminal positions
    """

    def __init__(self, nnue_path: Optional[str] = None):
        """
        Initialize the data generator.

        Args:
            nnue_path: Path to NNUE weights JSON file. If provided, uses NNUE
                      for evaluation during self-play. Otherwise uses heuristics.
        """
        from tonnesjakk import Board, Engine
        self.Board = Board
        self.Engine = Engine
        # Single reusable engine - avoids memory issues with multiple engine instances
        self._engine = Engine()
        self._using_nnue = False
        self._nnue_path = nnue_path

        if nnue_path is not None:
            try:
                self._engine.load_nnue(nnue_path)
                self._using_nnue = True
                print(f"  Loaded NNUE weights from: {nnue_path}")
            except Exception as e:
                print(f"  Warning: Failed to load NNUE ({e}), using heuristics")

    def play_game(
        self,
        depth: int = 6,
        random_opening_moves: int = 4,
        max_moves: int = 100
    ) -> GameResult:
        """
        Play a single self-play game.

        Args:
            depth: Search depth for the engine
            random_opening_moves: Number of random moves at start (2-6 recommended)
            max_moves: Maximum moves before declaring draw

        Returns:
            GameResult with positions and outcome
        """
        board = self.Board()
        engine = self._engine
        engine.full_reset()  # Full reset to prevent memory issues
        positions = []

        # Random opening moves for diversity
        # This is crucial to prevent all games being identical
        actual_random = random.randint(
            max(1, random_opening_moves - 2),
            random_opening_moves + 2
        )

        for _ in range(actual_random):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Engine plays the rest
        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            result = engine.search(board, depth)
            if result.best_move is None:
                break

            # Save position with search score
            # Quiet position filtering (Stockfish-inspired):
            #   - Skip first 4 moves (too influenced by random opening)
            #   - Skip clearly decided positions (|score| > 3000)
            if move_count >= 4:
                raw_score = result.score

                # Skip noisy/decided positions
                if abs(raw_score) <= 3000:
                    is_white = "White" in repr(board.current_player)
                    current_player = 1 if is_white else -1

                    # Sigmoid normalization: tanh(score / SCALING)
                    # Unlike linear clip, this preserves information for all scores.
                    # tanh maps (-inf, +inf) to (-1, +1) smoothly.
                    normalized_score = math.tanh(raw_score / SCORE_SCALING)

                    # Flip score perspective to always be from White's viewpoint
                    if not is_white:
                        normalized_score = -normalized_score

                    positions.append(PositionData(
                        board=board.to_array(),
                        search_score=normalized_score,
                        white_scored=board.white_scored,
                        black_scored=board.black_scored,
                        current_player=current_player
                    ))

            board.make_move(result.best_move)
            move_count += 1

        # Determine outcome
        winner = board.check_winner()
        if winner is None:
            outcome = 0.0  # Draw / max moves reached
        elif "White" in repr(winner):
            outcome = 1.0
        else:
            outcome = -1.0

        return GameResult(positions, outcome, move_count)

    def generate_dataset(
        self,
        num_games: int = 10000,
        depth: int = 6,
        random_opening_moves: int = 4,
        use_search_scores: bool = True,
        augment: bool = True,
        verbose: bool = True,
        save_every: int = 0,
        save_path: Optional[str] = None,
        config: Optional[Dict] = None,
        workers: int = 1,
        lambda_blend: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, TrainingStats]:
        """
        Generate training dataset from self-play games.

        Automatically resumes from save_path if it exists, appending new data.

        Args:
            num_games: Number of games to play
            depth: Search depth (6-8 recommended for quality/speed tradeoff)
            random_opening_moves: Random moves at start (4-6 recommended)
            use_search_scores: Use engine search scores (better) vs game outcomes
            augment: Apply horizontal flip data augmentation (doubles data)
            verbose: Print progress
            save_every: Save checkpoint every N games (0 = disabled)
            save_path: Path for checkpoint saves (required if save_every > 0)
            config: Config dict to store alongside the checkpoint
            workers: Number of parallel worker processes (1 = sequential)

        Returns:
            (X, y, stats) - input tensors, labels, and statistics
        """
        chunks_X: List[torch.Tensor] = []
        chunks_y: List[torch.Tensor] = []
        stats = TrainingStats()

        # Resume from existing file if present
        streaming = save_path and save_path.endswith('.bin')
        if save_path and Path(save_path).exists():
            try:
                if streaming:
                    meta_path = save_path.replace('.bin', '_meta.json')
                    if Path(meta_path).exists():
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                        stats.white_wins = meta.get('white_wins', 0)
                        stats.black_wins = meta.get('black_wins', 0)
                        stats.draws = meta.get('draws', 0)
                        stats.total_positions = meta.get('total_positions', 0)
                        prev_games = stats.white_wins + stats.black_wins + stats.draws
                        if verbose:
                            print(f"  Resuming from {save_path}: {stats.total_positions:,} positions from {prev_games} games")
                else:
                    prev_X, prev_y, prev_stats, prev_config = self.load_dataset(save_path)
                    chunks_X.append(prev_X)
                    chunks_y.append(prev_y)
                    stats.white_wins = prev_stats.white_wins
                    stats.black_wins = prev_stats.black_wins
                    stats.draws = prev_stats.draws
                    stats.total_positions = prev_stats.total_positions
                    if verbose:
                        prev_games = prev_stats.white_wins + prev_stats.black_wins + prev_stats.draws
                        print(f"  Resuming from {save_path}: {len(prev_X):,} positions from {prev_games} games")
            except Exception as e:
                if verbose:
                    print(f"  Could not resume from {save_path}: {e}")

        nnue_path = None
        if self._using_nnue:
            nnue_path = getattr(self, '_nnue_path', None)

        start_time = time.time()

        if workers > 1:
            self._generate_parallel(
                num_games=num_games, depth=depth,
                random_opening_moves=random_opening_moves,
                use_search_scores=use_search_scores, augment=augment,
                verbose=verbose, save_every=save_every, save_path=save_path,
                config=config, workers=workers, nnue_path=nnue_path,
                chunks_X=chunks_X, chunks_y=chunks_y, stats=stats,
                start_time=start_time, lambda_blend=lambda_blend
            )
        else:
            self._generate_sequential(
                num_games=num_games, depth=depth,
                random_opening_moves=random_opening_moves,
                use_search_scores=use_search_scores, augment=augment,
                verbose=verbose, save_every=save_every, save_path=save_path,
                config=config, chunks_X=chunks_X, chunks_y=chunks_y,
                stats=stats, start_time=start_time, lambda_blend=lambda_blend
            )

        if verbose:
            elapsed = time.time() - start_time
            print(f"\nGeneration complete in {elapsed:.1f}s")
            print(f"  {stats}")
            print(f"  {stats.total_positions:,} total positions" +
                  (" (includes augmentation)" if augment else ""))
            print(f"  Balance ratio: {stats.balance_ratio:.2f} (1.0 = perfect)")
            if lambda_blend is not None:
                print(f"  Labels: lambda blend ({lambda_blend:.2f} eval + {1-lambda_blend:.2f} outcome)")
            else:
                print(f"  Labels: {'search scores' if use_search_scores else 'game outcomes'}")

        # Build final tensors
        streaming = save_path and save_path.endswith('.bin')
        if streaming:
            # Data is already on disk — return empty tensors
            # Caller should use load_streaming_dataset() for training
            X, y = torch.zeros(0, INPUT_SIZE), torch.zeros(0, 1)
            if verbose:
                print(f"  Saved dataset to: {save_path} (streaming)")
        elif not chunks_X:
            X, y = torch.zeros(0, INPUT_SIZE), torch.zeros(0, 1)
        else:
            X = torch.cat(chunks_X, dim=0)
            y = torch.cat(chunks_y, dim=0)

        # Final save (non-streaming only)
        if save_path and not streaming:
            self.save_dataset(X, y, stats, save_path, config)
            if verbose:
                print(f"  Saved dataset to: {save_path}")

        return X, y, stats

    def _generate_sequential(
        self, num_games, depth, random_opening_moves, use_search_scores,
        augment, verbose, save_every, save_path, config,
        chunks_X, chunks_y, stats, start_time, lambda_blend=None
    ):
        """Sequential game generation (single process)."""
        pending_X: List[torch.Tensor] = []
        pending_y: List[float] = []

        def _compact_pending():
            if pending_X:
                chunks_X.append(torch.stack(pending_X))
                chunks_y.append(torch.tensor(pending_y, dtype=torch.float32).unsqueeze(1))
                pending_X.clear()
                pending_y.clear()

        for game_num in range(num_games):
            result = self.play_game(depth, random_opening_moves)

            if result.outcome > 0.5:
                stats.white_wins += 1
            elif result.outcome < -0.5:
                stats.black_wins += 1
            else:
                stats.draws += 1

            for pos_data in result.positions:
                # Lambda blend: mix search scores with game outcomes
                if lambda_blend is not None:
                    label = lambda_blend * pos_data.search_score + (1 - lambda_blend) * result.outcome
                elif use_search_scores:
                    label = pos_data.search_score
                else:
                    label = result.outcome

                tensor = board_to_tensor(
                    pos_data.board,
                    white_scored=pos_data.white_scored,
                    black_scored=pos_data.black_scored,
                    current_player=pos_data.current_player
                )
                pending_X.append(tensor)
                pending_y.append(label)

                if augment:
                    flipped_board = flip_board_horizontal(pos_data.board)
                    flipped_tensor = board_to_tensor(
                        flipped_board,
                        white_scored=pos_data.white_scored,
                        black_scored=pos_data.black_scored,
                        current_player=pos_data.current_player
                    )
                    pending_X.append(flipped_tensor)
                    pending_y.append(label)

            stats.total_positions = sum(c.shape[0] for c in chunks_X) + len(pending_X)

            if verbose and (game_num + 1) % 50 == 0:
                elapsed = time.time() - start_time
                gps = (game_num + 1) / elapsed
                eta = (num_games - game_num - 1) / gps
                aug_note = " (2x augmented)" if augment else ""
                print(f"  Game {game_num + 1:5d}/{num_games} "
                      f"({gps:.1f}/s, ETA {eta/60:.1f}m) | "
                      f"{stats} | {stats.total_positions:,} positions{aug_note}", flush=True)

            if save_every > 0 and save_path and (game_num + 1) % save_every == 0:
                _compact_pending()
                save_start = time.time()
                X_all = torch.cat(chunks_X, dim=0) if chunks_X else torch.zeros(0, INPUT_SIZE)
                y_all = torch.cat(chunks_y, dim=0) if chunks_y else torch.zeros(0, 1)
                self.save_dataset(X_all, y_all, stats, save_path, config)
                save_elapsed = time.time() - save_start
                if verbose:
                    print(f"  >> Checkpoint saved ({stats.total_positions:,} positions, {save_elapsed:.1f}s)", flush=True)

        # Compact any remaining pending tensors
        _compact_pending()

    def _generate_parallel(
        self, num_games, depth, random_opening_moves, use_search_scores,
        augment, verbose, save_every, save_path, config, workers, nnue_path,
        chunks_X, chunks_y, stats, start_time, lambda_blend=None
    ):
        """Parallel game generation using multiprocessing.Pool.

        Uses streaming mode when save_path ends with .bin — writes positions
        directly to flat binary files to avoid accumulating all data in RAM.
        Falls back to in-memory mode for .npz paths (legacy behavior).
        """
        streaming = save_path and save_path.endswith('.bin')
        batch_size = save_every if save_every > 0 else num_games
        games_done = 0

        # Streaming mode: write to flat binary files
        if streaming:
            x_path = save_path
            y_path = save_path.replace('.bin', '_y.bin')
            meta_path = save_path.replace('.bin', '_meta.json')

            # Open files for appending (resume support: existing data stays)
            x_file = open(x_path, 'ab')
            y_file = open(y_path, 'ab')

        while games_done < num_games:
            batch = min(batch_size, num_games - games_done)

            # Split batch across workers
            per_worker = batch // workers
            remainder = batch % workers
            worker_args = []
            for w in range(workers):
                n = per_worker + (1 if w < remainder else 0)
                if n > 0:
                    worker_args.append((
                        n, depth, random_opening_moves,
                        use_search_scores, augment, nnue_path,
                        lambda_blend
                    ))

            # Run batch in parallel
            with multiprocessing.Pool(processes=len(worker_args)) as pool:
                results = pool.map(_generate_games_worker, worker_args)

            # Merge results from all workers
            for X_np, y_np, ww, bw, dw in results:
                if len(X_np) > 0:
                    if streaming:
                        # Write directly to disk, don't accumulate in RAM
                        X_np.astype(np.float32).tofile(x_file)
                        y_np.astype(np.float32).tofile(y_file)
                    else:
                        chunks_X.append(torch.tensor(X_np, dtype=torch.float32))
                        chunks_y.append(torch.tensor(y_np, dtype=torch.float32).unsqueeze(1))
                    stats.total_positions += len(X_np)
                stats.white_wins += ww
                stats.black_wins += bw
                stats.draws += dw

            games_done += batch
            if not streaming:
                stats.total_positions = sum(c.shape[0] for c in chunks_X)

            # Progress + checkpoint
            elapsed = time.time() - start_time
            gps = games_done / elapsed
            eta = (num_games - games_done) / gps if gps > 0 else 0
            aug_note = " (2x augmented)" if augment else ""

            if verbose:
                print(f"  Game {games_done:5d}/{num_games} "
                      f"({gps:.1f}/s, ETA {eta/60:.1f}m, {workers}w) | "
                      f"{stats} | {stats.total_positions:,} positions{aug_note}", flush=True)

            if save_every > 0 and save_path:
                if streaming:
                    save_start = time.time()
                    x_file.flush()
                    y_file.flush()
                    # Save metadata
                    meta = {
                        "total_positions": stats.total_positions,
                        "input_size": INPUT_SIZE,
                        "label_columns": 2,
                        "label_format": ["search_score", "outcome"],
                        "white_wins": stats.white_wins,
                        "black_wins": stats.black_wins,
                        "draws": stats.draws,
                        "config": config,
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f)
                    save_elapsed = time.time() - save_start
                    if verbose:
                        print(f"  >> Checkpoint flushed ({stats.total_positions:,} positions, {save_elapsed:.1f}s)", flush=True)
                else:
                    save_start = time.time()
                    X_all = torch.cat(chunks_X, dim=0) if chunks_X else torch.zeros(0, INPUT_SIZE)
                    y_all = torch.cat(chunks_y, dim=0) if chunks_y else torch.zeros(0, 1)
                    self.save_dataset(X_all, y_all, stats, save_path, config)
                    save_elapsed = time.time() - save_start
                    if verbose:
                        print(f"  >> Checkpoint saved ({stats.total_positions:,} positions, {save_elapsed:.1f}s)", flush=True)

        if streaming:
            x_file.close()
            y_file.close()
            # Final metadata save
            meta = {
                "total_positions": stats.total_positions,
                "input_size": INPUT_SIZE,
                "label_columns": 2,
                "label_format": ["search_score", "outcome"],
                "white_wins": stats.white_wins,
                "black_wins": stats.black_wins,
                "draws": stats.draws,
                "config": config,
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)

    def save_dataset(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        stats: TrainingStats,
        path: str,
        config: Optional[Dict] = None
    ):
        """Save generated dataset to file for reuse."""
        np.savez_compressed(
            path,
            X=X.numpy(),
            y=y.numpy(),
            white_wins=stats.white_wins,
            black_wins=stats.black_wins,
            draws=stats.draws,
            total_positions=stats.total_positions,
            config=json.dumps(config) if config else ""
        )

    @staticmethod
    def load_dataset(path: str) -> Tuple[torch.Tensor, torch.Tensor, TrainingStats, Optional[Dict]]:
        """Load dataset from file (.npz format, loads into RAM)."""
        data = np.load(path, allow_pickle=True)
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.float32)

        stats = TrainingStats()
        stats.white_wins = int(data['white_wins'])
        stats.black_wins = int(data['black_wins'])
        stats.draws = int(data['draws'])
        stats.total_positions = int(data['total_positions'])

        config = None
        config_str = str(data['config'])
        if config_str:
            try:
                config = json.loads(config_str)
            except:
                pass

        return X, y, stats, config

    @staticmethod
    def load_streaming_dataset(bin_path: str) -> Tuple[np.memmap, np.memmap, TrainingStats, Optional[Dict]]:
        """Load dataset from flat binary files using memory-mapping (low RAM)."""
        meta_path = bin_path.replace('.bin', '_meta.json')
        y_path = bin_path.replace('.bin', '_y.bin')

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        n = meta['total_positions']
        input_size = meta.get('input_size', INPUT_SIZE)

        X = np.memmap(bin_path, dtype=np.float32, mode='r', shape=(n, input_size))
        label_columns = meta.get('label_columns', 1)
        if label_columns == 2:
            y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n, 2))
        else:
            y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n,))

        stats = TrainingStats()
        stats.white_wins = meta.get('white_wins', 0)
        stats.black_wins = meta.get('black_wins', 0)
        stats.draws = meta.get('draws', 0)
        stats.total_positions = n

        config = meta.get('config', None)
        return X, y, stats, config


# =============================================================================
# Multiprocessing worker (top-level for pickling on Windows)
# =============================================================================

def _generate_games_worker(args):
    """Worker function for parallel game generation.

    Must be top-level (not a method) so it's picklable on Windows (spawn).
    Each worker creates its own DataGenerator with its own Engine instance.
    Returns numpy arrays to avoid torch tensor pickling issues.
    """
    num_games, depth, random_moves, use_search_scores, augment, nnue_path, lambda_blend = args

    gen = DataGenerator(nnue_path=nnue_path)

    all_X = []
    all_y = []  # Each entry is [search_score, outcome] — 2 floats per position
    white_wins = black_wins = draws = 0

    for _ in range(num_games):
        result = gen.play_game(depth, random_moves)

        if result.outcome > 0.5:
            white_wins += 1
        elif result.outcome < -0.5:
            black_wins += 1
        else:
            draws += 1

        for pos_data in result.positions:
            tensor = board_to_tensor(
                pos_data.board,
                white_scored=pos_data.white_scored,
                black_scored=pos_data.black_scored,
                current_player=pos_data.current_player
            )
            all_X.append(tensor)
            all_y.append([pos_data.search_score, result.outcome])

            if augment:
                flipped_board = flip_board_horizontal(pos_data.board)
                flipped_tensor = board_to_tensor(
                    flipped_board,
                    white_scored=pos_data.white_scored,
                    black_scored=pos_data.black_scored,
                    current_player=pos_data.current_player
                )
                all_X.append(flipped_tensor)
                all_y.append([pos_data.search_score, result.outcome])

    if all_X:
        X_np = torch.stack(all_X).numpy()
        y_np = np.array(all_y, dtype=np.float32)  # shape (N, 2)
    else:
        X_np = np.zeros((0, INPUT_SIZE), dtype=np.float32)
        y_np = np.zeros((0, 2), dtype=np.float32)

    return X_np, y_np, white_wins, black_wins, draws


# =============================================================================
# Training
# =============================================================================

def wdl_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss in Win-Draw-Loss probability space.

    Converts tanh-space values [-1, 1] to WDL probabilities [0, 1]
    and applies binary cross-entropy. This penalizes confident wrong
    predictions much harder than MSE, forcing the network to learn
    correct position ordering rather than outputting safe near-zero values.

    Inspired by Stockfish NNUE training (nnue-pytorch).
    """
    eps = 1e-7
    # Convert from tanh space [-1, 1] to WDL probability space [0, 1]
    p = torch.clamp((pred + 1) / 2, eps, 1 - eps)
    t = torch.clamp((target + 1) / 2, eps, 1 - eps)
    # Binary cross-entropy
    return -torch.mean(t * torch.log(p) + (1 - t) * torch.log(1 - p))


# =============================================================================
# Main Training Pipeline
# =============================================================================

def train_nnue(
    num_games: int = 10000,
    depth: int = 6,
    random_moves: int = 4,
    hidden1: int = 128,
    hidden2: int = 32,
    epochs: int = 200,
    output_dir: str = ".",
    use_nnue: Optional[str] = None,
    use_search_scores: bool = True,
    augment: bool = True,
    compare: bool = True,
    compare_games: int = 50,
    track_history: bool = True,
    save_data: Optional[str] = None,
    load_data: Optional[str] = None,
    save_every: int = 0,
    generate_only: bool = False,
    workers: int = 1,
    batch_size: int = 4096,
    lambda_blend: Optional[float] = None,
    loss_fn: str = "wdl-ce",
    learning_rate: float = 0.001,
    num_workers: int = 0,
    resume_from: Optional[str] = None,
) -> Optional[nn.Module]:
    """
    Complete HalfPail NNUE training pipeline.

    Recommended settings:
    - num_games: 10,000 - 20,000 for good results
    - depth: 6-8 (higher = better quality, slower generation)
    - random_moves: 4-6 (ensures diverse openings)
    - hidden1: 128 (EmbeddingBag output dimension)
    - hidden2: 32 (second hidden layer)
    - use_nnue: Path to existing NNUE weights for self-play (self-improvement loop)
    - use_search_scores: Use engine search scores (True) vs game outcomes (False)
    - augment: Apply horizontal flip augmentation (doubles training data)
    - save_data: Save generated positions to file for reuse
    - load_data: Load positions from file instead of generating
    """
    print("=" * 60)
    print("HALFPAIL NNUE TRAINING FOR TONNESJAKK")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Architecture: HalfPail EmbeddingBag({HALFPAIL_FEATURES}, {hidden1}) -> FC2({2*hidden1+HALFPAIL_DENSE}, {hidden2}) -> 1")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    print(f"  Loss: {loss_fn} ({'WDL cross-entropy' if loss_fn == 'wdl-ce' else 'mean squared error'})")
    print(f"  Features: HalfPail sparse ({HALFPAIL_FEATURES} perspective features, {HALFPAIL_DENSE} dense)")

    # Step 1: Generate or load data
    if load_data:
        print(f"\n[1/3] Loading training data from {load_data}...")
        if load_data.endswith('.bin'):
            X, y, stats, loaded_config = DataGenerator.load_streaming_dataset(load_data)
            print(f"  Loaded {len(X):,} positions (memory-mapped, features: {X.shape[1]})")
        else:
            X, y, stats, loaded_config = DataGenerator.load_dataset(load_data)
            print(f"  Loaded {len(X):,} positions (features: {X.shape[1]})")
        print(f"  {stats}")
        if loaded_config:
            print(f"  Original config: {loaded_config.get('games', '?')} games, depth {loaded_config.get('depth', '?')}")
    else:
        print(f"  Games: {num_games:,}")
        print(f"  Search depth: {depth}")
        print(f"  Random opening moves: {random_moves}")
        if use_nnue:
            print(f"  Self-play eval: NNUE ({use_nnue})")
        else:
            print(f"  Self-play eval: Heuristic")
        if lambda_blend is not None:
            print(f"  Labels: lambda blend ({lambda_blend:.2f} * search_score + {1-lambda_blend:.2f} * outcome)")
        else:
            print(f"  Labels: {'search scores' if use_search_scores else 'game outcomes'}")
        print(f"  Score normalization: tanh(score / {SCORE_SCALING})")
        print(f"  Quiet filtering: skip first 4 moves, skip |score| > 3000")
        print(f"  Augmentation: {'enabled (2x data)' if augment else 'disabled'}")
        if workers > 1:
            print(f"  Workers: {workers}")
        if save_data:
            print(f"  Save to: {save_data}" + (f" (every {save_every} games)" if save_every > 0 else ""))
        if generate_only:
            print(f"  Mode: GENERATE ONLY (no training)")

        step_label = "Generating" if generate_only else "[1/3] Generating"
        print(f"\n{step_label} training data...")
        generator = DataGenerator(nnue_path=use_nnue)
        config = {
            "games": num_games,
            "depth": depth,
            "random_moves": random_moves,
            "use_nnue": use_nnue,
            "use_search_scores": use_search_scores,
            "augment": augment,
            "input_size": INPUT_SIZE,
            "score_scaling": SCORE_SCALING,
            "lambda_blend": lambda_blend,
        }
        X, y, stats = generator.generate_dataset(
            num_games=num_games,
            depth=depth,
            random_opening_moves=random_moves,
            use_search_scores=use_search_scores,
            augment=augment,
            save_every=save_every,
            save_path=save_data,
            config=config,
            workers=workers,
            lambda_blend=lambda_blend
        )

        if generate_only:
            print(f"\nGeneration complete. Data saved to: {save_data}")
            print("Run with --load-data to train on this data.")
            return None

        # If streaming mode was used, load data via mmap for training
        if save_data and save_data.endswith('.bin') and len(X) == 0:
            print(f"\n  Loading generated data via memory-map...")
            X, y, stats, _ = DataGenerator.load_streaming_dataset(save_data)
            print(f"  Loaded {len(X):,} positions (memory-mapped)")

    # Check balance
    if stats.balance_ratio < 0.5 or stats.balance_ratio > 2.0:
        print(f"\n  WARNING: Dataset is unbalanced (ratio: {stats.balance_ratio:.2f})")
        print("  Consider adjusting search depth or random moves")

    # Step 2: Train
    print(f"\n[2/3] Training model...")
    model, history = train_halfpail_model(
        X, y,
        hidden1=hidden1,
        hidden2=hidden2,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        num_workers=num_workers,
        resume_from=resume_from,
        lambda_blend=lambda_blend,
    )

    # Step 3: Export
    print(f"\n[3/3] Exporting...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "nnue_weights.json"
    old_weights_path = None

    # Backup old weights if they exist
    if json_path.exists():
        import shutil
        # Find next available backup number
        gen = 1
        while (output_path / f"nnue_weights_gen{gen}.json").exists():
            gen += 1
        old_weights_path = output_path / f"nnue_weights_gen{gen}.json"
        shutil.copy(json_path, old_weights_path)
        print(f"  Backed up old weights to: {old_weights_path}")

    # Save PyTorch model
    torch_path = output_path / "nnue_model.pt"
    torch.save(model.state_dict(), torch_path)
    print(f"  PyTorch model: {torch_path}")

    # Save JSON for Rust
    export_halfpail_json(model, str(json_path))
    print(f"  JSON weights: {json_path}")

    # Step 4: Compare with heuristic baseline (and optionally previous version)
    comparison = None
    if compare:
        compare_depth = min(depth, 5)  # Use depth 5 max for faster comparison

        # Always compare against heuristic baseline
        print(f"\n[4/4] Comparing NNUE vs Heuristic ({compare_games} games, depth {compare_depth})...")
        comparison = compare_nnue(
            str(json_path),
            "heuristic",
            num_games=compare_games,
            depth=compare_depth,
            verbose=True
        )
        print(f"\n  NNUE vs Heuristic: NNUE={comparison['wins_a']} Heur={comparison['wins_b']} Draws={comparison['draws']}")
        print(f"  NNUE win rate: {comparison['win_rate_a']*100:.1f}%")
        print(f"  Estimated ELO diff: {comparison['elo_diff']:+d}")

        if comparison['elo_diff'] > 50:
            print("  [+] NNUE is significantly STRONGER than heuristic!")
        elif comparison['elo_diff'] > 0:
            print("  [+] NNUE is slightly stronger than heuristic")
        elif comparison['elo_diff'] > -50:
            print("  [=] NNUE is roughly equal to heuristic")
        else:
            print("  [!] NNUE is WEAKER than heuristic - needs more training")

        # Optionally also compare against previous NNUE version
        if old_weights_path:
            print(f"\n  Also comparing vs previous NNUE...")
            old_comparison = compare_nnue(
                str(json_path),
                str(old_weights_path),
                num_games=compare_games,
                depth=compare_depth,
                verbose=False
            )
            print(f"  New vs Old NNUE: New={old_comparison['wins_a']} Old={old_comparison['wins_b']} Draws={old_comparison['draws']}")
            print(f"  ELO diff vs old: {old_comparison['elo_diff']:+d}")

    # Track history
    if track_history:
        history_path = output_path / "nnue_history.json"
        history = load_training_history(str(history_path))
        generation = len(history) + 1

        config = {
            "games": num_games,
            "depth": depth,
            "random_moves": random_moves,
            "arch": [hidden1, hidden2],
            "epochs": epochs,
            "used_nnue": use_nnue is not None,
            "search_scores": use_search_scores,
            "augment": augment,
            "loss_fn": loss_fn,
        }
        add_training_result(generation, config, comparison, str(history_path))
        print(f"\n  Training history saved (generation {generation})")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Show history summary
    if track_history:
        print_training_history(str(output_path / "nnue_history.json"))

    return model


# =============================================================================
# NNUE Comparison and History
# =============================================================================

def compare_nnue(
    nnue_a: str,
    nnue_b: str,
    num_games: int = 100,
    depth: int = 6,
    time_ms: int = 0,
    verbose: bool = True
) -> Dict:
    """
    Play matches between two NNUE versions to compare strength.

    Args:
        nnue_a: Path to first NNUE weights (or "heuristic" for no NNUE)
        nnue_b: Path to second NNUE weights (or "heuristic" for no NNUE)
        num_games: Number of games to play
        depth: Search depth (used when time_ms=0)
        time_ms: Time per move in milliseconds (0 = use fixed depth instead)

    Returns:
        Dict with results: wins_a, wins_b, draws, win_rate_a, elo_diff
    """
    from tonnesjakk import Board, Engine

    engine_a = Engine()
    engine_b = Engine()

    if nnue_a != "heuristic":
        engine_a.load_nnue(nnue_a)
    if nnue_b != "heuristic":
        engine_b.load_nnue(nnue_b)

    mode_str = f"{time_ms}ms/move" if time_ms > 0 else f"depth {depth}"
    if verbose:
        print(f"  Mode: {mode_str}")

    wins_a = 0
    wins_b = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate colors
        white_is_a = (game_idx % 2 == 0)

        board = Board()
        engine_a.full_reset()
        engine_b.full_reset()

        # Random opening for variety
        for _ in range(random.randint(2, 4)):
            moves = board.generate_moves()
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Play game
        game_start = time.time()
        move_count = 0
        while board.check_winner() is None and move_count < 50:
            # Per-game time limit: 60 seconds
            if time.time() - game_start > 60.0:
                break

            # Determine which engine plays
            is_white_turn = "White" in repr(board.current_player)
            current_engine = engine_a if (is_white_turn == white_is_a) else engine_b

            if time_ms > 0:
                result = current_engine.search_timed(board, time_ms)
            else:
                result = current_engine.search(board, depth)
            if result.best_move is None:
                break
            board.make_move(result.best_move)
            move_count += 1

        game_time = time.time() - game_start

        # Score result
        winner = board.check_winner()
        if winner is None:
            draws += 1
        elif "White" in repr(winner):
            if white_is_a:
                wins_a += 1
            else:
                wins_b += 1
        else:
            if white_is_a:
                wins_b += 1
            else:
                wins_a += 1

        if verbose:
            w = "draw" if winner is None else ("A" if (("White" in repr(winner)) == white_is_a) else "B")
            print(f"  Game {game_idx + 1}/{num_games}: {w} ({move_count} moves, {game_time:.1f}s) | A={wins_a} B={wins_b} D={draws}", flush=True)

    # Calculate stats
    total = wins_a + wins_b + draws
    win_rate_a = (wins_a + 0.5 * draws) / total if total > 0 else 0.5

    # Approximate ELO difference (using logistic model)
    # ELO diff = 400 * log10(win_rate / (1 - win_rate))
    import math
    if win_rate_a > 0.01 and win_rate_a < 0.99:
        elo_diff = 400 * math.log10(win_rate_a / (1 - win_rate_a))
    else:
        elo_diff = 400 if win_rate_a > 0.5 else -400

    return {
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": win_rate_a,
        "elo_diff": round(elo_diff),
        "games": total
    }


def load_training_history(path: str = "nnue_history.json") -> List[Dict]:
    """Load training history from JSON file."""
    if Path(path).exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []


def save_training_history(history: List[Dict], path: str = "nnue_history.json"):
    """Save training history to JSON file."""
    with open(path, 'w') as f:
        json.dump(history, f, indent=2)


def add_training_result(
    generation: int,
    config: Dict,
    comparison: Optional[Dict] = None,
    history_path: str = "nnue_history.json"
):
    """Add a training result to the history."""
    history = load_training_history(history_path)

    entry = {
        "generation": generation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "comparison": comparison
    }
    history.append(entry)

    save_training_history(history, history_path)
    return entry


def print_training_history(history_path: str = "nnue_history.json"):
    """Print training history summary."""
    history = load_training_history(history_path)

    if not history:
        print("No training history found.")
        return

    print("\n" + "=" * 60)
    print("TRAINING HISTORY")
    print("=" * 60)
    print(f"{'Gen':>4} {'Date':>12} {'Games':>7} {'Depth':>5} {'vs Prev':>10} {'ELO Diff':>10}")
    print("-" * 60)

    for entry in history:
        gen = entry.get("generation", "?")
        date = entry.get("timestamp", "?")[:10]
        config = entry.get("config", {})
        games = config.get("games", "?")
        depth = config.get("depth", "?")

        comp = entry.get("comparison")
        if comp:
            win_rate = f"{comp['win_rate_a']*100:.0f}%"
            elo = f"{comp['elo_diff']:+d}"
        else:
            win_rate = "-"
            elo = "-"

        print(f"{gen:>4} {date:>12} {games:>7} {depth:>5} {win_rate:>10} {elo:>10}")

    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train HalfPail NNUE for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tonnesjakk.nnue --load-data data.bin --epochs 50   # Train on existing data
  python -m tonnesjakk.nnue --games 10000 --save-data data.bin  # Generate training data
  python -m tonnesjakk.nnue --use-nnue nnue_weights.json  # Self-improvement loop
  python -m tonnesjakk.nnue --compare a.json b.json --depth 6  # Equal-depth comparison
  python -m tonnesjakk.nnue --compare-timed 500 a.json b.json  # Equal-time comparison (500ms/move)
        """
    )

    parser.add_argument("--games", type=int, default=10000,
                        help="Number of self-play games (default: 10000)")
    parser.add_argument("--depth", type=int, default=6,
                        help="Search depth for self-play (default: 6)")
    parser.add_argument("--random-moves", type=int, default=4,
                        help="Random opening moves (default: 4)")
    parser.add_argument("--arch", type=int, nargs=2, default=[128, 32],
                        metavar=("H1", "H2"),
                        help="Hidden layer sizes (default: 128 32)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Training batch size (default: 4096)")
    parser.add_argument("--loss", type=str, default="wdl-ce", choices=["mse", "wdl-ce"],
                        help="Loss function: 'wdl-ce' (cross-entropy in WDL space) or 'mse' (default: wdl-ce)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001, try 0.003 with batch-size 4096)")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory (default: current)")
    parser.add_argument("--use-nnue", type=str, default=None,
                        help="Use existing NNUE weights for self-play (self-improvement loop)")
    parser.add_argument("--save-data", type=str, default=None,
                        help="Save generated positions to file for reuse")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load positions from file instead of generating")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable horizontal flip data augmentation")
    parser.add_argument("--no-compare", action="store_true",
                        help="Skip comparison with previous version")
    parser.add_argument("--compare-games", type=int, default=50,
                        help="Number of games for comparison (default: 50)")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't track training history")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N games during generation (default: 500)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data (no training). Use with --save-data")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--history", action="store_true",
                        help="Show training history and exit")
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_blend",
                        help="Lambda blend: mix search scores and game outcomes (0.85 = 85%% eval + 15%% outcome)")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("NNUE_A", "NNUE_B"),
                        help="Compare two NNUE versions (use 'heuristic' for no NNUE)")
    parser.add_argument("--compare-timed", type=str, nargs=3, metavar=("MS", "NNUE_A", "NNUE_B"),
                        help="Equal-time comparison: MS per move (use 'heuristic' for no NNUE)")
    parser.add_argument("--halfpail", action="store_true",
                        help="Use HalfPail architecture (default, kept for backward compatibility)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from a saved .pt model file")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers for training (0=main process, 4+ recommended)")
    parser.add_argument("--test-halfpail", action="store_true",
                        help="Run HalfPail decoding round-trip test")

    args = parser.parse_args()

    if args.test_halfpail:
        test_halfpail_decoding(1000)
    elif args.history:
        print_training_history(str(Path(args.output) / "nnue_history.json"))
    elif args.compare_timed:
        ms, nnue_a, nnue_b = int(args.compare_timed[0]), args.compare_timed[1], args.compare_timed[2]
        print(f"Comparing {nnue_a} vs {nnue_b} ({ms}ms/move, {args.compare_games} games)...")
        result = compare_nnue(
            nnue_a, nnue_b,
            num_games=args.compare_games,
            time_ms=ms,
            verbose=True
        )
        print(f"\nFinal: A={result['wins_a']} B={result['wins_b']} D={result['draws']}")
        print(f"Win rate A: {result['win_rate_a']*100:.1f}%")
        print(f"ELO difference: {result['elo_diff']:+d} (A vs B)")
    elif args.compare:
        print(f"Comparing {args.compare[0]} vs {args.compare[1]} (depth {args.depth})...")
        result = compare_nnue(
            args.compare[0],
            args.compare[1],
            num_games=args.compare_games,
            depth=args.depth,
            verbose=True
        )
        print(f"\nFinal: A={result['wins_a']} B={result['wins_b']} D={result['draws']}")
        print(f"Win rate A: {result['win_rate_a']*100:.1f}%")
        print(f"ELO difference: {result['elo_diff']:+d} (A vs B)")
    else:
        train_nnue(
            num_games=args.games,
            depth=args.depth,
            random_moves=args.random_moves,
            hidden1=args.arch[0],
            hidden2=args.arch[1],
            epochs=args.epochs,
            output_dir=args.output,
            use_nnue=args.use_nnue,
            augment=not args.no_augment,
            compare=not args.no_compare,
            compare_games=args.compare_games,
            track_history=not args.no_history,
            save_data=args.save_data,
            load_data=args.load_data,
            save_every=args.save_every,
            generate_only=args.generate_only,
            workers=args.workers,
            batch_size=args.batch_size,
            lambda_blend=args.lambda_blend,
            loss_fn=args.loss,
            learning_rate=args.lr,
            num_workers=args.num_workers,
            resume_from=args.resume_from,
        )


if __name__ == "__main__":
    main()
