"""Inspect an AlphaZero model checkpoint.

Usage:
  python scripts/inspect_model.py alphazero_run3/latest_model.pt
  python scripts/inspect_model.py alphazero_run3/latest_model.pt --positions 10
"""

import argparse

import numpy as np
import torch

from tonnesjakk._core import MCTSEngine as _RustMCTSEngine
from tonnesjakk.alphazero import AlphaZeroTrainer, POLICY_SIZE


def _get_starting_planes():
    """Get planes for the starting position via a minimal Rust game."""
    engine = _RustMCTSEngine(1, 1.4)
    results = engine.play_alphabeta_games(1, depth=1, random_opening=0, max_moves=1)
    return np.array(results[0].examples[0].planes, dtype=np.float32).reshape(6, 6, 6)


def _eval_position(net, planes_np):
    """Run network on a single position, return (probs, value, entropy)."""
    planes = torch.tensor(planes_np, dtype=torch.float32).reshape(1, 6, 6, 6)
    with torch.no_grad():
        logits, value = net(planes)
    probs = logits.softmax(1)[0]
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum().item()
    return probs, value.item(), entropy


def inspect(path: str, n_positions: int = 5):
    trainer = AlphaZeroTrainer(device="cpu")
    trainer.load(path)
    net = trainer.network
    net.eval()

    max_entropy = np.log(POLICY_SIZE)

    print(f"Model: {path}")
    print(f"  Network: {trainer.network_type}, {net.num_parameters:,} params")
    print(f"  Replay buffer: {len(trainer.replay_buffer):,} examples")
    print()

    # --- Starting position ---
    start_planes = _get_starting_planes()
    probs, value, entropy = _eval_position(net, start_planes)
    top_k = probs.topk(5)

    print("Starting position:")
    print(f"  Value:   {value:+.4f}  (should be near 0 = balanced)")
    print(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} max  ({entropy/max_entropy:.0%})")
    print(f"  Top 5 probs: {[f'{p:.3f}' for p in top_k.values.tolist()]}")
    print()

    # --- Evaluate random buffer positions ---
    if len(trainer.replay_buffer) >= n_positions:
        indices = np.random.choice(len(trainer.replay_buffer), n_positions, replace=False)
        print(f"Sample buffer positions ({n_positions} random):")
        print(f"  {'Pos':>3}  {'True Val':>9}  {'Net Val':>8}  {'Entropy':>8}  {'Top Prob':>9}")
        for i, idx in enumerate(indices):
            planes_np, policy_np, true_val = trainer.replay_buffer[idx]
            probs, pred_val, entropy = _eval_position(net, planes_np)
            print(f"  {i+1:3d}  {true_val:+9.4f}  {pred_val:+8.4f}  {entropy:8.2f}  {probs.max().item():9.3f}")
        print()

    # --- Buffer stats ---
    if len(trainer.replay_buffer) > 0:
        values = np.array([ex[2] for ex in trainer.replay_buffer])
        print("Replay buffer value distribution:")
        print(f"  Mean:   {values.mean():+.4f}")
        print(f"  Std:    {values.std():.4f}")
        print(f"  Positive (White wins): {(values > 0).sum():,} ({(values > 0).mean():.0%})")
        print(f"  Zero (draws):          {(values == 0).sum():,} ({(values == 0).mean():.0%})")
        print(f"  Negative (Black wins): {(values < 0).sum():,} ({(values < 0).mean():.0%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect AlphaZero checkpoint")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--positions", type=int, default=5,
                        help="Moves to play out (default: 5)")
    args = parser.parse_args()
    inspect(args.checkpoint, args.positions)
