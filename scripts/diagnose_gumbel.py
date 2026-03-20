"""
Diagnose Gumbel AlphaZero policy target diffusion.

Hypothesis: Gumbel's improved policy targets (softmax(log_prior + sigma * Q))
are more diffuse than standard MCTS visit-count targets, leading to ~80% draws
in v13 compared to ~47% draws in v9.

This script:
  1. Analyzes saved replay buffers from v9 and v13 for aggregate entropy stats
  2. Loads both models and runs live MCTS searches (standard and Gumbel)
  3. Compares policy target entropy, top-1/top-3 mass, visit distributions

Usage:
  python scripts/diagnose_gumbel.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the project package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from tonnesjakk import Board
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine
from tonnesjakk.alphazero import (
    BOARD_PLANES, BOARD_SIZE, POLICY_SIZE, make_network,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (nats). Ignores zeros."""
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


def top_k_mass(p: np.ndarray, k: int) -> float:
    """Sum of the top-k probabilities."""
    return float(np.sort(p)[-k:].sum())


def num_nonzero(p: np.ndarray, eps: float = 1e-6) -> int:
    """Number of entries above eps."""
    return int(np.sum(p > eps))


def policy_stats(policies: np.ndarray) -> dict:
    """Compute aggregate statistics over a batch of policy vectors."""
    entropies = np.array([entropy(p) for p in policies])
    top1s = np.array([top_k_mass(p, 1) for p in policies])
    top3s = np.array([top_k_mass(p, 3) for p in policies])
    nnz = np.array([num_nonzero(p) for p in policies])
    return {
        "n_positions": len(policies),
        "entropy_mean": float(entropies.mean()),
        "entropy_std": float(entropies.std()),
        "entropy_median": float(np.median(entropies)),
        "top1_mean": float(top1s.mean()),
        "top1_std": float(top1s.std()),
        "top1_median": float(np.median(top1s)),
        "top3_mean": float(top3s.mean()),
        "top3_std": float(top3s.std()),
        "top3_median": float(np.median(top3s)),
        "nonzero_mean": float(nnz.mean()),
        "nonzero_std": float(nnz.std()),
    }


def print_stats(label: str, stats: dict):
    """Pretty-print policy statistics."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Positions:         {stats['n_positions']:,}")
    print(f"  Entropy:           {stats['entropy_mean']:.4f} +/- {stats['entropy_std']:.4f}  (median {stats['entropy_median']:.4f})")
    print(f"  Top-1 probability: {stats['top1_mean']:.4f} +/- {stats['top1_std']:.4f}  (median {stats['top1_median']:.4f})")
    print(f"  Top-3 probability: {stats['top3_mean']:.4f} +/- {stats['top3_std']:.4f}  (median {stats['top3_median']:.4f})")
    print(f"  Non-zero actions:  {stats['nonzero_mean']:.1f} +/- {stats['nonzero_std']:.1f}")


def print_comparison(label_a: str, stats_a: dict, label_b: str, stats_b: dict):
    """Print side-by-side comparison of two policy stat dicts."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*60}")
    print(f"  {'Metric':<25s}  {'':>12s}  {'':>12s}  {'delta':>10s}")
    print(f"  {'':<25s}  {label_a:>12s}  {label_b:>12s}  {'':>10s}")
    print(f"  {'-'*25}  {'-'*12}  {'-'*12}  {'-'*10}")
    for key, name in [
        ("entropy_mean", "Entropy (mean)"),
        ("entropy_median", "Entropy (median)"),
        ("top1_mean", "Top-1 prob (mean)"),
        ("top1_median", "Top-1 prob (median)"),
        ("top3_mean", "Top-3 prob (mean)"),
        ("top3_median", "Top-3 prob (median)"),
        ("nonzero_mean", "Non-zero actions"),
    ]:
        va, vb = stats_a[key], stats_b[key]
        delta = vb - va
        sign = "+" if delta > 0 else ""
        print(f"  {name:<25s}  {va:>12.4f}  {vb:>12.4f}  {sign}{delta:>9.4f}")


# ---------------------------------------------------------------------------
# Part 1: Analyze saved replay buffers
# ---------------------------------------------------------------------------

def analyze_replay_buffer(buf_path: str, label: str) -> dict | None:
    """Load and analyze policy targets from a saved replay buffer."""
    path = Path(buf_path)
    if not path.exists():
        print(f"  [SKIP] {path} not found")
        return None

    data = np.load(path)
    policies = data["policies"]
    values = data["values"]

    stats = policy_stats(policies)

    # Also compute value distribution stats
    n_white = int(np.sum(values > 0.5))
    n_black = int(np.sum(values < -0.5))
    n_draw = int(np.sum(np.abs(values) <= 0.5))
    stats["value_white_pct"] = n_white / len(values) * 100
    stats["value_black_pct"] = n_black / len(values) * 100
    stats["value_draw_pct"] = n_draw / len(values) * 100

    print_stats(f"{label} (replay buffer: {path.name})", stats)
    print(f"  Value targets:     {stats['value_white_pct']:.1f}% white / "
          f"{stats['value_draw_pct']:.1f}% draw / "
          f"{stats['value_black_pct']:.1f}% black")

    return stats


# ---------------------------------------------------------------------------
# Part 2: Load model and run live searches
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, network_type: str = "resnet",
               hidden: int = 64, num_blocks: int = 5) -> torch.nn.Module:
    """Load a model from checkpoint."""
    net = make_network(network_type, hidden=hidden, num_blocks=num_blocks)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    return net


def make_batch_eval_fn(net, batch_size: int = 8):
    """Create a batch evaluation function for MCTSEngine."""
    plane_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE
    input_buffer = torch.zeros(batch_size, plane_size, dtype=torch.float32)

    def batch_eval_fn(batch_planes: list) -> tuple:
        n = len(batch_planes)
        np_batch = np.array(batch_planes, dtype=np.float32)
        cpu_tensor = torch.as_tensor(np_batch)
        buf = input_buffer[:n]
        buf.copy_(cpu_tensor)
        with torch.no_grad():
            policy_logits, values = net(buf)
        return policy_logits.numpy().tolist(), values.numpy().tolist()

    return batch_eval_fn


def run_live_searches(net, simulations: int = 200, c_puct: float = 1.0,
                      batch_size: int = 8, use_gumbel: bool = False,
                      n_games: int = 10, label: str = "") -> dict | None:
    """Run self-play games and collect policy target statistics."""
    engine = _RustMCTSEngine(simulations, c_puct, use_gumbel=use_gumbel)
    eval_fn = make_batch_eval_fn(net, batch_size=batch_size)

    all_policies = []
    results = {"white": 0, "black": 0, "draw": 0}

    for i in range(n_games):
        nr = engine.play_network_game(
            eval_fn, batch_size=batch_size,
            random_opening=4, max_moves=80,
            temp_moves=3, temperature=0.8,
        )
        for ex in nr.examples:
            all_policies.append(np.array(ex.policy_target, dtype=np.float32))
        results[nr.winner] += 1

    if not all_policies:
        print(f"  [SKIP] No positions collected for {label}")
        return None

    policies = np.array(all_policies)
    stats = policy_stats(policies)
    stats["game_white"] = results["white"]
    stats["game_black"] = results["black"]
    stats["game_draw"] = results["draw"]

    print_stats(f"{label} (live: {n_games} games, {simulations} sims)", stats)
    print(f"  Game results:      {results['white']}W / {results['draw']}D / {results['black']}B "
          f"({results['draw']/n_games*100:.0f}% draws)")

    return stats


def run_single_position_comparison(net, simulations: int = 200, c_puct: float = 1.0,
                                   batch_size: int = 8, label: str = ""):
    """Run standard vs Gumbel search from starting position and compare."""
    eval_fn = make_batch_eval_fn(net, batch_size=batch_size)
    board = Board()

    # Standard MCTS
    engine_std = _RustMCTSEngine(simulations, c_puct, use_gumbel=False)
    result_std = engine_std.search_network_batched(board, eval_fn, batch_size)
    p_std = np.array(result_std.policy_target, dtype=np.float32)

    # For Gumbel, we need to go through play_network_game since search_network_batched
    # doesn't support Gumbel. Instead, let's run a single game and look at move 1 targets.
    # But we can't isolate a single position that way.
    # So we just report standard MCTS results per position.

    print(f"\n  --- {label}: Starting position search ({simulations} sims) ---")
    print(f"  Standard MCTS:")
    print(f"    Visits:    {result_std.visits}")
    print(f"    Root val:  {result_std.root_value:.4f}")
    print(f"    Entropy:   {entropy(p_std):.4f}")
    print(f"    Top-1:     {top_k_mass(p_std, 1):.4f}")
    print(f"    Top-3:     {top_k_mass(p_std, 3):.4f}")
    print(f"    Non-zero:  {num_nonzero(p_std)}")

    # Show top moves
    top_indices = np.argsort(p_std)[::-1][:5]
    print(f"    Top moves: ", end="")
    for idx in top_indices:
        if p_std[idx] < 1e-6:
            break
        from_sq = idx // 36
        to_sq = idx % 36
        print(f"({from_sq}->{to_sq}: {p_std[idx]:.3f}) ", end="")
    print()


# ---------------------------------------------------------------------------
# Part 3: Entropy histogram (text-based)
# ---------------------------------------------------------------------------

def print_entropy_histogram(policies: np.ndarray, label: str, bins: int = 20):
    """Print a text histogram of per-position entropy."""
    entropies = np.array([entropy(p) for p in policies])
    counts, edges = np.histogram(entropies, bins=bins)
    max_count = max(counts)
    bar_width = 40

    print(f"\n  Entropy histogram: {label}")
    print(f"  {'Range':>15s}  {'Count':>6s}  {'':>{bar_width}s}")
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i+1]
        bar_len = int(c / max_count * bar_width) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  [{lo:6.3f}-{hi:6.3f})  {c:>6d}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("GUMBEL vs STANDARD MCTS POLICY TARGET DIAGNOSTIC")
    print("=" * 60)

    # --- Part 1: Replay buffer analysis ---
    print("\n\n" + "=" * 60)
    print("PART 1: REPLAY BUFFER ANALYSIS")
    print("=" * 60)

    v9_buf = base / "alphazero_v9" / "latest_model.pt.buffer.npz"
    v13_buf = base / "alphazero_v13" / "latest_model.pt.buffer.npz"

    stats_v9_buf = analyze_replay_buffer(str(v9_buf), "v9 (standard MCTS)")
    stats_v13_buf = analyze_replay_buffer(str(v13_buf), "v13 (Gumbel)")

    if stats_v9_buf and stats_v13_buf:
        print_comparison("v9 (std)", stats_v9_buf, "v13 (gumbel)", stats_v13_buf)

    # Entropy histograms from buffers
    if v9_buf.exists():
        data_v9 = np.load(v9_buf)
        print_entropy_histogram(data_v9["policies"], "v9 (standard MCTS)")
    if v13_buf.exists():
        data_v13 = np.load(v13_buf)
        print_entropy_histogram(data_v13["policies"], "v13 (Gumbel)")

    # --- Part 2: Live search comparison ---
    print("\n\n" + "=" * 60)
    print("PART 2: LIVE SEARCH COMPARISON")
    print("=" * 60)

    # Load v9 model (hidden=64, num_blocks=5)
    v9_model_path = base / "alphazero_v9" / "latest_model.pt"
    v13_model_path = base / "alphazero_v13" / "latest_model.pt"

    n_games = 20  # games per config

    if v9_model_path.exists():
        print(f"\nLoading v9 model from {v9_model_path}...")
        net_v9 = load_model(str(v9_model_path), hidden=64, num_blocks=5)

        # v9 with standard MCTS (as trained)
        stats_v9_std = run_live_searches(
            net_v9, simulations=200, c_puct=1.0, batch_size=16,
            use_gumbel=False, n_games=n_games,
            label="v9 model + Standard MCTS",
        )

        # v9 with Gumbel (counterfactual)
        stats_v9_gumbel = run_live_searches(
            net_v9, simulations=200, c_puct=1.0, batch_size=16,
            use_gumbel=True, n_games=n_games,
            label="v9 model + Gumbel MCTS",
        )

        if stats_v9_std and stats_v9_gumbel:
            print_comparison("v9+std", stats_v9_std, "v9+gumbel", stats_v9_gumbel)

        # Starting position analysis
        run_single_position_comparison(net_v9, simulations=200, c_puct=1.0,
                                       batch_size=16, label="v9 model")
    else:
        print(f"  [SKIP] v9 model not found at {v9_model_path}")

    if v13_model_path.exists():
        print(f"\nLoading v13 model from {v13_model_path}...")
        net_v13 = load_model(str(v13_model_path), hidden=128, num_blocks=5)

        # v13 with Gumbel (as trained)
        stats_v13_gumbel = run_live_searches(
            net_v13, simulations=400, c_puct=1.0, batch_size=16,
            use_gumbel=True, n_games=n_games,
            label="v13 model + Gumbel MCTS (as trained)",
        )

        # v13 with standard MCTS (counterfactual)
        stats_v13_std = run_live_searches(
            net_v13, simulations=400, c_puct=1.0, batch_size=16,
            use_gumbel=False, n_games=n_games,
            label="v13 model + Standard MCTS",
        )

        if stats_v13_std and stats_v13_gumbel:
            print_comparison("v13+std", stats_v13_std, "v13+gumbel", stats_v13_gumbel)

        # Starting position analysis
        run_single_position_comparison(net_v13, simulations=400, c_puct=1.0,
                                       batch_size=16, label="v13 model")
    else:
        print(f"  [SKIP] v13 model not found at {v13_model_path}")

    # --- Part 3: Cross-comparison summary ---
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
  Key question: Are Gumbel policy targets more diffuse (higher entropy)?

  Initial hypothesis was that Gumbel targets are TOO DIFFUSE. But the data
  may show the OPPOSITE: Gumbel's softmax(log_prior + sigma * Q) targets can
  become near-one-hot when sigma is large (sigma = (c_visit + max_visit) * c_scale,
  with c_visit=50, c_scale=1.0). After enough simulations, one action dominates
  the normalized Q, and sigma amplifies this into a near-deterministic target.

  If Gumbel targets are TOO SHARP:
  - The policy network learns to always pick one move with ~100% confidence
  - This kills exploration during self-play (even in temp_moves regime)
  - Without exploring alternatives, games converge to repetitive draws
  - The value head gets noisy signal because all games look similar

  Possible fixes if targets are too sharp:
  - Label smoothing: blend Gumbel targets with uniform over legal moves
  - Temperature: apply T>1 to Gumbel logits before softmax
  - Lower sigma: reduce GUMBEL_C_SCALE (e.g. 0.5) or GUMBEL_C_VISIT (e.g. 10)
  - Use visit-count targets for training, Gumbel only for move selection
  - Add entropy bonus to policy loss

  Possible fixes if targets are too diffuse:
  - Increase GUMBEL_C_SCALE to make Q-values dominate
  - Decrease GUMBEL_C_VISIT to reduce initial diffusion
  - Add temperature sharpening to Gumbel policy targets before training
    """)


if __name__ == "__main__":
    main()
