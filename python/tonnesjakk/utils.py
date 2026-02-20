"""
Shared utility functions for Tonnesjakk.

Moved from mcts.py and alphazero.py to avoid circular imports
and make them available across the package.
"""

import math
from typing import Tuple

from tonnesjakk import Board


def safe_str(obj) -> str:
    """Convert to string, replacing Unicode chars that Windows cp1252 can't handle."""
    return str(obj).encode("ascii", errors="replace").decode("ascii")


def is_white(board: Board) -> bool:
    """Check if it's White's turn."""
    return "White" in repr(board.current_player)


def is_white_winner(winner) -> bool:
    """Check if the winner is White."""
    return "White" in repr(winner)


def elo_with_ci(wins: int, losses: int, draws: int) -> Tuple[float, float, float]:
    """Return (elo_diff, elo_lo, elo_hi) using Wilson 95% CI on win rate."""
    n = wins + losses + draws
    if n == 0:
        return 0.0, -400.0, 400.0

    successes = wins + 0.5 * draws
    p_hat = successes / n

    z = 1.96
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom

    lo = max(0.001, centre - spread)
    hi = min(0.999, centre + spread)

    def wr_to_elo(wr: float) -> float:
        if wr <= 0.001:
            return -400.0
        if wr >= 0.999:
            return 400.0
        return 400.0 * math.log10(wr / (1.0 - wr))

    return wr_to_elo(p_hat), wr_to_elo(lo), wr_to_elo(hi)


def get_device(requested: str = "auto"):
    """Detect best available device: CUDA > MPS > CPU."""
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
