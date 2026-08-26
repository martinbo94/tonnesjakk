"""Generic sparse NNUE: architecture config, PyTorch model, decoder, trainer, export.

One dataset (164-float rows: 144 one-hot piece planes + 20 relational features,
2-column labels [search_score, outcome]) serves every architecture here. The
Rust batch decoder derives each architecture's sparse indices from the shared
rows using the exact same feature code the engine evaluates with, so training
and inference can never disagree on the encoding.

Architecture knobs (see `NnueArch`):
- feature_set: "halfpail" (own-pail-square buckets, 3996 features) or
  "plain" (no buckets, 144 features)
- mirror_black: black perspective sees the board flipped vertically so shared
  weights are orientation-consistent across colors
- dense_size: 0 or 20 relational features appended before FC2
- hidden1 / hidden2: layer widths (multiples of 8)
- output_buckets: 1, or 25 = one output head per (white_scored, black_scored)

Export format "sparse_nnue_v2" is loaded by the Rust `SparseNNUE`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tonnesjakk import decode_sparse_batch as _rust_decode_sparse_batch

FEATURE_COUNTS = {"halfpail": 37 * 36 * 3, "plain": 36 * 4}
DENSE_FEATURES = 20


@dataclass(frozen=True)
class NnueArch:
    feature_set: str = "halfpail"
    mirror_black: bool = False
    dense_size: int = 20
    hidden1: int = 128
    hidden2: int = 32
    output_buckets: int = 1

    def __post_init__(self):
        if self.feature_set not in FEATURE_COUNTS:
            raise ValueError(f"unknown feature_set {self.feature_set!r}")
        if self.dense_size not in (0, DENSE_FEATURES):
            raise ValueError("dense_size must be 0 or 20")
        if self.output_buckets not in (1, 25):
            raise ValueError("output_buckets must be 1 or 25")
        if self.hidden1 % 16 != 0 or self.hidden1 <= 0:
            raise ValueError("hidden1 must be a positive multiple of 16 (i16x16 accumulator lanes)")
        if self.hidden2 % 8 != 0 or self.hidden2 <= 0:
            raise ValueError("hidden2 must be a positive multiple of 8")

    @property
    def num_features(self) -> int:
        return FEATURE_COUNTS[self.feature_set]

    @property
    def fc2_input(self) -> int:
        return 2 * self.hidden1 + self.dense_size

    def tag(self) -> str:
        """Short filesystem-safe name, e.g. halfpail_m_d20_128x32_b25."""
        return (f"{self.feature_set}{'_m' if self.mirror_black else ''}"
                f"_d{self.dense_size}_{self.hidden1}x{self.hidden2}"
                f"{'_b25' if self.output_buckets == 25 else ''}")


class SparseNNUE(nn.Module):
    """
    own-perspective sparse -> EmbeddingBag(F, H1, sum) + bias -> ReLU
    opp-perspective sparse -> (shared)                          -> ReLU
    concat(acc_w, acc_b[, dense]) -> FC2[bucket] -> ReLU -> FC3[bucket] -> tanh

    Bucketed heads are implemented as wide linears producing all buckets at
    once, then gathering the per-sample bucket — cheap for 25 small heads.
    """

    def __init__(self, arch: NnueArch):
        super().__init__()
        self.arch = arch
        B, H1, H2 = arch.output_buckets, arch.hidden1, arch.hidden2
        self.embedding = nn.EmbeddingBag(arch.num_features, H1, mode="sum")
        self.fc1_bias = nn.Parameter(torch.zeros(H1))
        self.fc2 = nn.Linear(arch.fc2_input, B * H2)
        self.fc3 = nn.Linear(H2, B) if B == 1 else None
        # Bucketed FC3: per-bucket weight vectors [B, H2] and bias [B]
        if B > 1:
            self.fc3_weight = nn.Parameter(torch.empty(B, H2))
            self.fc3_bias = nn.Parameter(torch.zeros(B))
            nn.init.xavier_uniform_(self.fc3_weight)
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        if self.fc3 is not None:
            nn.init.xavier_uniform_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)

    def forward(self, w_idx, w_off, b_idx, b_off, dense, bucket):
        acc_w = torch.relu(self.embedding(w_idx, w_off) + self.fc1_bias)
        acc_b = torch.relu(self.embedding(b_idx, b_off) + self.fc1_bias)
        parts = [acc_w, acc_b] + ([dense] if self.arch.dense_size else [])
        x = torch.cat(parts, dim=1)
        B, H2 = self.arch.output_buckets, self.arch.hidden2
        h_all = torch.relu(self.fc2(x))  # [N, B*H2]
        if B == 1:
            return torch.tanh(self.fc3(h_all))
        h = h_all.view(-1, B, H2)
        idx = bucket.view(-1, 1, 1).expand(-1, 1, H2)
        h_sel = torch.gather(h, 1, idx).squeeze(1)            # [N, H2]
        w_sel = self.fc3_weight[bucket]                         # [N, H2]
        b_sel = self.fc3_bias[bucket]                           # [N]
        out = (h_sel * w_sel).sum(dim=1) + b_sel
        return torch.tanh(out).unsqueeze(1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── decoding ────────────────────────────────────────────────────────────────

def decode_chunk(arch: NnueArch, X_chunk, y_chunk):
    """Rust batch decode of contiguous rows -> tensors for `SparseNNUE.forward`."""
    flat_x = np.ascontiguousarray(X_chunk, dtype=np.float32).ravel().tolist()
    flat_y = np.ascontiguousarray(y_chunk, dtype=np.float32).ravel().tolist()
    w_idx, w_off, b_idx, b_off, dense_flat, buckets, labels = _rust_decode_sparse_batch(
        flat_x, flat_y, arch.feature_set, arch.mirror_black, arch.dense_size, arch.output_buckets
    )
    n = len(w_off)
    dense = (torch.tensor(dense_flat, dtype=torch.float32).view(n, arch.dense_size)
             if arch.dense_size else torch.zeros(n, 0))
    return (
        torch.tensor(w_idx, dtype=torch.long),
        torch.tensor(w_off, dtype=torch.long),
        torch.tensor(b_idx, dtype=torch.long),
        torch.tensor(b_off, dtype=torch.long),
        dense,
        torch.tensor(buckets, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32).unsqueeze(1),
    )


def predecode_to_device(arch: NnueArch, X, y, start, count, batch_size, device,
                        decode_chunk_size: int = 65536, verbose: bool = True):
    """Decode rows once into device-resident batches (unified memory on Apple
    Silicon makes this essentially free; on CUDA it removes per-epoch decode)."""
    batches = []
    n_chunks = (count + decode_chunk_size - 1) // decode_chunk_size
    for di in range(n_chunks):
        ds = start + di * decode_chunk_size
        de = min(ds + decode_chunk_size, start + count)
        w_idx, w_off, b_idx, b_off, dense, buckets, labels = decode_chunk(arch, X[ds:de], y[ds:de])
        chunk_n = de - ds
        for bi in range(0, chunk_n, batch_size):
            be = min(bi + batch_size, chunk_n)
            ws, we = int(w_off[bi]), (int(w_off[be]) if be < chunk_n else len(w_idx))
            bs, be_ = int(b_off[bi]), (int(b_off[be]) if be < chunk_n else len(b_idx))
            batches.append((
                w_idx[ws:we].to(device), (w_off[bi:be] - w_off[bi]).to(device),
                b_idx[bs:be_].to(device), (b_off[bi:be] - b_off[bi]).to(device),
                dense[bi:be].to(device), buckets[bi:be].to(device), labels[bi:be].to(device),
            ))
        if verbose:
            print(f"    {min((di + 1) * decode_chunk_size, count):,}/{count:,} decoded", flush=True)
    return batches


# ─── deduplication ───────────────────────────────────────────────────────────

def dedupe_rows(X, y, verbose: bool = True):
    """Collapse identical positions, averaging their labels.

    A deterministic engine with short random openings funnels games into the
    same lines: measured 58% duplicate rows in gen-1. Averaging the labels of
    duplicates denoises the target and stops funnel lines from dominating
    training. Position identity = 144 piece planes + scored counts + pails +
    side to move (features 144+8..144+13); the remaining relational features
    are functions of those.
    """
    n = len(X)
    keys = np.empty((n, 23), dtype=np.uint8)
    chunk = 250_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        xb = np.asarray(X[s:e], dtype=np.float32)
        keys[s:e, :18] = np.packbits(xb[:, :144] > 0.5, axis=1)
        keys[s:e, 18:] = np.round(xb[:, 152:157] * 4).astype(np.int8).view(np.uint8)
    flat = np.ascontiguousarray(keys).view(np.dtype((np.void, 23))).ravel()
    _, first_idx, inverse = np.unique(flat, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    m = len(first_idx)

    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr[:, None]
    counts = np.bincount(inverse, minlength=m).astype(np.float32)
    y_unique = np.zeros((m, y_arr.shape[1]), dtype=np.float32)
    for c in range(y_arr.shape[1]):
        y_unique[:, c] = np.bincount(inverse, weights=y_arr[:, c], minlength=m) / counts

    # Emit unique positions in original (game) order for contiguous reads.
    # perm[j] = unique-id whose first occurrence is the j-th smallest row index,
    # so X_unique[j] = X[first_idx[perm[j]]] and y_unique_out[j] = y_avg[perm[j]].
    perm = np.argsort(first_idx)
    order = first_idx[perm]
    X_unique = np.empty((m, X.shape[1]), dtype=np.float32)
    for s in range(0, m, chunk):
        X_unique[s:s + chunk] = X[order[s:s + chunk]]
    y_unique = y_unique[perm]

    if verbose:
        print(f"  Dedupe: {n:,} rows -> {m:,} unique positions "
              f"({100 * (1 - m / n):.1f}% duplicates removed, labels averaged)")
    return X_unique, y_unique


# ─── training ────────────────────────────────────────────────────────────────

def wdl_cross_entropy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = torch.clamp((pred + 1) / 2, eps, 1 - eps)
    t = torch.clamp((target + 1) / 2, eps, 1 - eps)
    return -(t * torch.log(p) + (1 - t) * torch.log(1 - p)).mean()


def train_sparse_model(
    X, y, arch: NnueArch,
    epochs: int = 50,
    batch_size: int = 8192,
    learning_rate: float = 0.001,
    validation_split: float = 0.1,
    loss_fn: str = "wdl-ce",
    lambda_blend: Optional[float] = None,
    resume_from: Optional[str] = None,
    device: Optional[torch.device] = None,
    dedupe: bool = False,
    verbose: bool = True,
):
    """Train a SparseNNUE. `y` may be (N,), (N,1) or (N,2)=[search_score, outcome]
    (blended with lambda_blend: label = λ·score + (1−λ)·outcome, default λ=1).
    With dedupe=True identical positions are collapsed and their labels averaged."""
    from .utils import get_device
    device = device or get_device("auto")

    if dedupe:
        X, y = dedupe_rows(X, y, verbose=verbose)
    n = len(X)
    split = int((1 - validation_split) * n)
    if y.ndim == 2 and y.shape[1] == 2:
        lb = 1.0 if lambda_blend is None else lambda_blend
        y_flat = (lb * y[:, 0] + (1 - lb) * y[:, 1]).astype(np.float32)
        if verbose:
            print(f"  Label blend: {lb:.2f} * search_score + {1 - lb:.2f} * outcome")
    else:
        y_flat = np.asarray(y).ravel().astype(np.float32)

    model = SparseNNUE(arch)
    if resume_from:
        model.load_state_dict(torch.load(resume_from, map_location="cpu", weights_only=True))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)
    criterion = wdl_cross_entropy if loss_fn == "wdl-ce" else nn.MSELoss()

    if verbose:
        print(f"  Arch: {arch.tag()}  ({model.num_parameters:,} params)  device {device}")
        print(f"  Train {split:,} / val {n - split:,} positions, batch {batch_size}, "
              f"{epochs} epochs, lr {learning_rate} (cosine), loss {loss_fn}")
        print(f"  Pre-decoding {n:,} positions...", flush=True)

    t0 = time.time()
    train_batches = predecode_to_device(arch, X, y_flat, 0, split, batch_size, device, verbose=verbose)
    val_batches = predecode_to_device(arch, X, y_flat, split, n - split, batch_size, device, verbose=verbose)
    if verbose:
        print(f"  Decoded in {time.time() - t0:.1f}s", flush=True)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state = float("inf"), None
    t_train = time.time()
    for epoch in range(epochs):
        t_ep = time.time()
        model.train()
        total = 0.0
        for bi in np.random.permutation(len(train_batches)):
            w_idx, w_off, b_idx, b_off, dense, bucket, labels = train_batches[bi]
            optimizer.zero_grad()
            loss = criterion(model(w_idx, w_off, b_idx, b_off, dense, bucket), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        train_loss = total / max(len(train_batches), 1)

        model.eval()
        vt, vc = 0.0, 0
        with torch.no_grad():
            for w_idx, w_off, b_idx, b_off, dense, bucket, labels in val_batches:
                vt += criterion(model(w_idx, w_off, b_idx, b_off, dense, bucket), labels).item() * labels.shape[0]
                vc += labels.shape[0]
        val_loss = vt / max(vc, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        marker = ""
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        scheduler.step()
        if verbose:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: train={train_loss:.4f} val={val_loss:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6f} ({time.time() - t_ep:.0f}s, "
                  f"total {time.time() - t_train:.0f}s){marker}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(f"  Best validation loss: {best_val:.4f}")
    return model, history


# ─── export ──────────────────────────────────────────────────────────────────

def export_sparse_json(model: SparseNNUE, output_path: str) -> None:
    """Write the 'sparse_nnue_v2' JSON the Rust `SparseNNUE` loads."""
    arch = model.arch
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    B, H2 = arch.output_buckets, arch.hidden2
    fc2_w = state["fc2.weight"].view(B, H2, arch.fc2_input)
    fc2_b = state["fc2.bias"].view(B, H2)
    if B == 1:
        fc3_w = state["fc3.weight"].view(1, H2)
        fc3_b = state["fc3.bias"].view(1)
    else:
        fc3_w = state["fc3_weight"]
        fc3_b = state["fc3_bias"]
    out = {
        "format": "sparse_nnue_v2",
        **asdict(arch),
        "weights": {
            "fc1_weight": state["embedding.weight"].tolist(),
            "fc1_bias": state["fc1_bias"].tolist(),
            "fc2_weight": fc2_w.tolist(),
            "fc2_bias": fc2_b.tolist(),
            "fc3_weight": fc3_w.tolist(),
            "fc3_bias": fc3_b.tolist(),
        },
    }
    Path(output_path).write_text(json.dumps(out))


def torch_eval_positions(model: SparseNNUE, X_rows) -> np.ndarray:
    """Evaluate raw 164-rows with the torch model (centipawns, White view) —
    used to verify Rust/Python parity after export."""
    model_cpu = model.to("cpu").eval()
    y_dummy = np.zeros(len(X_rows), dtype=np.float32)
    w_idx, w_off, b_idx, b_off, dense, bucket, _ = decode_chunk(model.arch, X_rows, y_dummy)
    with torch.no_grad():
        out = model_cpu(w_idx, w_off, b_idx, b_off, dense, bucket).squeeze(1).numpy()
    # Same inverse mapping as the Rust evaluator: label = tanh(cp / 600)
    return (600.0 * np.arctanh(np.clip(out, -0.999, 0.999))).astype(np.int32)
