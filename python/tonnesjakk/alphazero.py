"""
AlphaZero-style self-play training for Tonnesjakk.

Combines MCTS with a dual-headed neural network (policy + value) to bootstrap
game-playing knowledge from self-play alone -- no heuristic needed.

The network has two outputs:
  - Policy head: probability distribution over moves (guides MCTS exploration)
  - Value head: position evaluation in [-1, +1] (replaces heuristic at leaves)

Training loop:
  1. Self-play N games using MCTS + current network
  2. Train network on (board, mcts_policy, game_outcome) triples
  3. Repeat

Usage:
  python -m tonnesjakk.alphazero --iterations 20 --games-per-iter 50 --simulations 200
  python -m tonnesjakk.alphazero --evaluate model.pt --games 50 --opponent-depth 5
"""

import argparse
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tonnesjakk import Board, Engine
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine, POLICY_SIZE as _RUST_POLICY_SIZE
from tonnesjakk.utils import is_white as _is_white, is_white_winner as _is_white_winner, safe_str as _safe_str, elo_with_ci

SCORE_SCALING = 600.0  # tanh(score/600) normalization, matches NNUE training


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_SIZE = 37 * 36  # 1332: from_idx (0-36) x to_idx (0-35)
BOARD_PLANES = 6       # 4 piece planes + current player + bias
BOARD_SIZE = 6


# Device detection — imported from utils
from tonnesjakk.utils import get_device  # noqa: E402


# ---------------------------------------------------------------------------
# Board symmetry augmentation (#8)
# ---------------------------------------------------------------------------

MIRROR_COL = [5, 4, 3, 2, 1, 0]


def _mirror_sq(sq: int) -> int:
    """Mirror a square index (0-35) or off-board (36) left-right."""
    if sq == 36:
        return 36
    row, col = sq // 6, sq % 6
    return row * 6 + MIRROR_COL[col]


# Build policy mirror mapping once at module load
_POLICY_MIRROR = np.zeros(POLICY_SIZE, dtype=np.int64)
for _from_idx in range(37):
    for _to_idx in range(36):
        _old_idx = _from_idx * 36 + _to_idx
        _new_idx = _mirror_sq(_from_idx) * 36 + _mirror_sq(_to_idx)
        _POLICY_MIRROR[_old_idx] = _new_idx


def mirror_planes(planes: np.ndarray) -> np.ndarray:
    """Mirror board planes left-right. Input shape: (6, 6, 6) or (N, 6, 6, 6)."""
    return np.ascontiguousarray(planes[..., ::-1])


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a policy vector using precomputed mapping. Shape: (1332,) or (N, 1332)."""
    return policy[..., _POLICY_MIRROR]


# ---------------------------------------------------------------------------
# Move encoding
# ---------------------------------------------------------------------------

def move_to_index(move) -> int:
    """Encode a Move as a policy index in [0, 1331].

    Encoding: from_idx * 36 + to_idx
      - from_idx: row*6+col for board squares (0-35), 36 for off-board placement
      - to_idx: row*6+col for destination (0-35)

    Compound moves (with pail placement) map to the same index as the
    barrel-only action. This is a minor simplification -- pail placement
    only matters in the first 1-2 moves.
    """
    to_idx = move.barrel_to.row * 6 + move.barrel_to.col
    if move.is_barrel_placement:
        from_idx = 36  # off-board
    else:
        from_idx = move.barrel_from.row * 6 + move.barrel_from.col
    return from_idx * 36 + to_idx


def board_to_planes(board) -> np.ndarray:
    """Convert board to 6x6x6 float planes for network input.

    Planes:
      0: White barrels (1.0 where present)
      1: Black barrels
      2: White pail
      3: Black pail
      4: Current player (1.0 if White to move, 0.0 if Black)
      5: Ones (bias plane)
    """
    arr = board.to_array()  # 6x6, values in {0, +1, -1, +2, -2}
    planes = np.zeros((BOARD_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = arr[r][c]
            if v == 1:
                planes[0, r, c] = 1.0
            elif v == -1:
                planes[1, r, c] = 1.0
            elif v == 2:
                planes[2, r, c] = 1.0
            elif v == -2:
                planes[3, r, c] = 1.0
    if _is_white(board):
        planes[4, :, :] = 1.0
    planes[5, :, :] = 1.0
    return planes


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class AlphaZeroNet(nn.Module):
    """Dual-headed network for AlphaZero.

    Architecture: MLP with shared trunk, separate policy and value heads.
    Input: 6x6x6 = 216 features (flattened board planes).
    Policy: 1332 logits (from x to square pairs).
    Value: scalar in [-1, +1] (White's perspective).
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        input_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE  # 216

        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden, POLICY_SIZE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 6, 6, 6) or (batch, 216) board planes.

        Returns:
            (policy_logits, value): policy is (batch, 1332), value is (batch,).
        """
        x = x.view(x.size(0), -1)
        shared = self.shared(x)
        value = self.value_head(shared).squeeze(-1)
        policy_logits = self.policy_head(shared)
        return policy_logits, value

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ResidualBlock(nn.Module):
    """Single residual block: two 3x3 convs with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroResNet(nn.Module):
    """ResNet dual-headed network for AlphaZero.

    Architecture matching alpha-zero-general / AlphaZero.jl for small boards:
    - Initial 3x3 conv expanding to `channels` filters
    - `num_blocks` residual blocks (each 2x conv + skip)
    - Separate policy head (1x1 conv -> FC -> 1332) and value head (1x1 conv -> FC -> tanh)

    Input: 6x6x6 = (6 planes, 6 rows, 6 cols) board representation.
    Policy: 1332 logits (from x to square pairs).
    Value: scalar in [-1, +1] (White's perspective).

    Default (5 blocks x 128 filters) has ~1.6M parameters — 7x the MLP but
    learns spatial patterns (jumps, pushes, blocking) via weight sharing.
    """

    def __init__(self, num_blocks: int = 5, channels: int = 128):
        super().__init__()
        self.channels = channels

        # Initial convolution: 6 input planes -> channels
        self.conv_init = nn.Conv2d(BOARD_PLANES, channels, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # Policy head: 1x1 conv -> batch norm -> ReLU -> flatten -> FC
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, POLICY_SIZE)

        # Value head: 1x1 conv -> batch norm -> ReLU -> flatten -> FC -> ReLU -> FC -> tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, channels)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 6, 6, 6) or (batch, 216) board planes.

        Returns:
            (policy_logits, value): policy is (batch, 1332), value is (batch,).
        """
        # Reshape flat input to spatial: (batch, 6_planes, 6_rows, 6_cols)
        if x.dim() == 2:
            x = x.view(-1, BOARD_PLANES, BOARD_SIZE, BOARD_SIZE)

        # Initial conv
        x = F.relu(self.bn_init(self.conv_init(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_network(network_type: str = "resnet", hidden: int = 128, num_blocks: int = 5) -> nn.Module:
    """Create a network by type name.

    Args:
        network_type: "resnet" (default) or "mlp".
        hidden: Hidden size / channel count.
        num_blocks: Number of residual blocks (resnet only).
    """
    if network_type == "resnet":
        return AlphaZeroResNet(num_blocks=num_blocks, channels=hidden)
    elif network_type == "mlp":
        return AlphaZeroNet(hidden=hidden)
    else:
        raise ValueError(f"Unknown network type: {network_type!r}. Use 'resnet' or 'mlp'.")


# ---------------------------------------------------------------------------
# Network-aware MCTS
# ---------------------------------------------------------------------------

class NetworkMCTS:
    """MCTS using a neural network for move priors and leaf evaluation.

    Uses the Rust MCTSEngine with batched evaluation: multiple MCTS leaves are
    selected using virtual loss, then evaluated in a single network forward pass.
    This reduces Python round-trips from N_sims to N_sims/batch_size.
    """

    def __init__(
        self,
        network,
        simulations: int = 200,
        c_puct: float = 1.4,
        batch_size: int = 8,
        dirichlet_alpha: float = 0.5,
        dirichlet_epsilon: float = 0.25,
        device: torch.device = None,
    ):
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or torch.device("cpu")
        self._engine = _RustMCTSEngine(simulations, c_puct)
        self._batch_eval_fn = self._make_batch_eval_fn()

    def _make_batch_eval_fn(self):
        """Create the batched Python callback for Rust MCTS leaf evaluation.

        Accepts a list of plane vectors (one per leaf), returns batched results.
        """
        net = self.network
        device = self.device

        def batch_eval_fn(batch_planes: list) -> tuple:
            # batch_planes: list of N lists of 216 floats
            tensor = torch.tensor(batch_planes, dtype=torch.float32, device=device)
            with torch.no_grad():
                policy_logits, values = net(tensor)
            return policy_logits.cpu().tolist(), values.cpu().tolist()

        return batch_eval_fn

    def search(self, board: Board, add_noise: bool = True) -> Tuple[Optional[object], dict]:
        """Run MCTS from the given position using batched evaluation.

        Args:
            board: Current board state.
            add_noise: Unused (kept for interface compat).

        Returns:
            (best_move, info_dict) with policy target and visit stats.
        """
        result = self._engine.search_network_batched(
            board, self._batch_eval_fn, self.batch_size
        )

        policy_target = np.array(result.policy_target, dtype=np.float32)

        info = {
            "visits": result.visits,
            "policy_target": policy_target,
            "root_value": result.root_value,
        }

        return result.best_move, info


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def self_play_game(
    network_mcts: NetworkMCTS,
    temperature: float = 1.0,
    temp_threshold: int = 15,
    max_moves: int = 80,
    random_opening: int = 4,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str]:
    """Play one self-play game, collecting training data.

    Args:
        network_mcts: MCTS with network evaluation.
        temperature: Exploration temperature for move selection.
        temp_threshold: After this many moves, switch to deterministic.
        max_moves: Maximum game length.
        random_opening: Number of random opening moves.

    Returns:
        (examples, winner_str) where examples is list of
        (board_planes, policy_target, outcome) tuples.
    """
    board = Board()
    examples = []

    # Random opening for variety
    for _ in range(random_opening):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        board.make_move(random.choice(moves))

    move_count = 0
    while board.check_winner() is None and move_count < max_moves:
        best_move, info = network_mcts.search(board, add_noise=True)
        if best_move is None:
            break

        # Collect training example (outcome filled in after game)
        planes = board_to_planes(board)
        policy_target = info["policy_target"]
        examples.append((planes, policy_target))

        # Temperature-based move selection
        if move_count < temp_threshold and temperature > 0:
            # Build visit distribution and sample
            move = _sample_move_by_temperature(
                network_mcts, board, info, temperature
            )
            if move is None:
                move = best_move
        else:
            move = best_move

        board.make_move(move)
        move_count += 1

    # Determine outcome
    # For draws, use heuristic eval of final position as value target.
    # This bootstraps the value head -- pure game outcome (0.0) gives no
    # signal when most games draw due to weak early play.
    winner = board.check_winner()
    if winner is None:
        engine = Engine()
        heuristic_score = engine.evaluate_position(board)
        outcome = math.tanh(heuristic_score / SCORE_SCALING)
        winner_str = "draw"
    elif _is_white_winner(winner):
        outcome = 1.0
        winner_str = "white"
    else:
        outcome = -1.0
        winner_str = "black"

    # Fill in outcomes
    training_examples = [
        (planes, policy, outcome) for planes, policy in examples
    ]

    return training_examples, winner_str


def _sample_move_by_temperature(network_mcts, board, info, temperature):
    """Sample a move from visit count distribution with temperature."""
    children_info = info.get("children", [])
    if not children_info:
        return None

    # We need actual move objects, not just strings. Re-generate legal moves
    # and match by visit count ordering.
    # Simpler approach: regenerate moves and use policy_target as distribution
    moves = board.generate_moves()
    if not moves:
        return None

    policy_target = info["policy_target"]
    move_probs = []
    valid_moves = []
    for m in moves:
        idx = move_to_index(m)
        prob = policy_target[idx]
        if prob > 0:
            move_probs.append(prob)
            valid_moves.append(m)

    if not valid_moves:
        return None

    # Apply temperature
    probs = np.array(move_probs, dtype=np.float64)
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / probs.sum()

    idx = np.random.choice(len(valid_moves), p=probs)
    return valid_moves[idx]


def heuristic_self_play_game(
    simulations: int = 400,
    c_puct: float = 1.4,
    max_moves: int = 80,
    random_opening: int = 6,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], str]:
    """Play one self-play game using pure heuristic MCTS (Rust, very fast).

    These games produce decisive outcomes (wins/losses) to bootstrap the value
    head. The policy targets come from heuristic MCTS visit counts -- not ideal,
    but they teach basic move ordering.

    Returns:
        (examples, winner_str) same format as self_play_game().
    """
    board = Board()
    engine = _RustMCTSEngine(simulations, c_puct)
    examples = []

    # More random opening moves for variety (heuristic games are fast)
    for _ in range(random_opening):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        board.make_move(random.choice(moves))

    move_count = 0
    while board.check_winner() is None and move_count < max_moves:
        result = engine.search_heuristic(board)
        if result.best_move is None:
            break

        planes = board_to_planes(board)
        policy_target = np.array(result.policy_target, dtype=np.float32)
        examples.append((planes, policy_target))

        # Temperature-based selection for first moves, then deterministic
        if move_count < 10:
            moves = board.generate_moves()
            if moves:
                probs = np.array([policy_target[move_to_index(m)] for m in moves], dtype=np.float64)
                total = probs.sum()
                if total > 0:
                    probs /= total
                    idx = np.random.choice(len(moves), p=probs)
                    board.make_move(moves[idx])
                else:
                    board.make_move(result.best_move)
            else:
                break
        else:
            board.make_move(result.best_move)
        move_count += 1

    # Determine outcome
    winner = board.check_winner()
    if winner is None:
        eng = Engine()
        heuristic_score = eng.evaluate_position(board)
        outcome = math.tanh(heuristic_score / SCORE_SCALING)
        winner_str = "draw"
    elif _is_white_winner(winner):
        outcome = 1.0
        winner_str = "white"
    else:
        outcome = -1.0
        winner_str = "black"

    training_examples = [
        (planes, policy, outcome) for planes, policy in examples
    ]
    return training_examples, winner_str


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AlphaZeroTrainer:
    """AlphaZero training loop: self-play -> train -> repeat.

    Args:
        hidden: Network hidden layer size / channel count.
        simulations: MCTS simulations per move.
        c_puct: PUCT exploration constant.
        lr: Learning rate.
        games_per_iter: Self-play games per training iteration.
        training_epochs: Epochs per training iteration.
        temperature: Exploration temperature for self-play.
        buffer_max: Maximum training examples to keep in replay buffer.
        train_window: Max examples to train on per iteration. When the buffer
            exceeds this, a random subset biased toward recent data is sampled.
            Keeps training time constant as buffer grows. Default: 20000.
        network_type: "resnet" (default, 5 res blocks) or "mlp" (3-layer MLP).
        num_blocks: Residual blocks (resnet only, default: 5).
    """

    def __init__(
        self,
        hidden: int = 128,
        simulations: int = 200,
        c_puct: float = 1.4,
        lr: float = 0.001,
        games_per_iter: int = 50,
        training_epochs: int = 5,
        batch_size: int = 256,
        temperature: float = 1.0,
        buffer_max: int = 100000,
        train_window: int = 20000,
        network_type: str = "resnet",
        num_blocks: int = 5,
        policy_weight: float = 1.0,
        device: str = "auto",
    ):
        self.network_type = network_type
        self.device = get_device(device)
        self.network = make_network(network_type, hidden=hidden, num_blocks=num_blocks)
        self.network.to(self.device)
        self.simulations = simulations
        self.c_puct = c_puct
        self.lr = lr
        self.games_per_iter = games_per_iter
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.buffer_max = buffer_max
        self.train_window = train_window
        self.policy_weight = policy_weight

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []

        print(f"AlphaZero network: {self.network.num_parameters:,} parameters ({self.device})")

    def set_lr_schedule(self, total_steps: int):
        """Update LR schedule T_max for known total iteration count."""
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=self.lr * 0.01
        )

    def run(
        self,
        iterations: int = 50,
        eval_every: int = 5,
        eval_games: int = 20,
        eval_depth: int = 5,
        save_dir: str = "alphazero_checkpoints",
        heuristic_ratio: float = 0.0,
        verbose: bool = True,
    ):
        """Run the full AlphaZero training loop.

        Args:
            iterations: Number of train iterations.
            eval_every: Evaluate against heuristic every N iterations.
            eval_games: Games per evaluation.
            eval_depth: Alpha-beta depth for evaluation opponent.
            save_dir: Directory for saving checkpoints.
            heuristic_ratio: Fraction of self-play games to play with heuristic
                MCTS instead of network (0.0-1.0). Heuristic games are very fast
                and produce decisive outcomes -- critical for bootstrapping the
                value head early in training.
            verbose: Print progress.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_elo = -400.0
        total_games = 0

        for iteration in range(1, iterations + 1):
            iter_start = time.time()

            # --- Self-play ---
            self.network.eval()
            mcts = NetworkMCTS(
                self.network,
                simulations=self.simulations,
                c_puct=self.c_puct,
                device=self.device,
            )

            new_examples = []
            results = {"white": 0, "black": 0, "draw": 0}
            n_heuristic = int(self.games_per_iter * heuristic_ratio)
            n_network = self.games_per_iter - n_heuristic

            # Heuristic self-play games (fully in Rust, ~4ms each)
            if n_heuristic > 0:
                rust_engine = _RustMCTSEngine(max(200, self.simulations), self.c_puct)
                h_results = rust_engine.play_heuristic_games(
                    n_heuristic, random_opening=6, max_moves=80, temp_moves=10,
                )
                for hr in h_results:
                    for ex in hr.examples:
                        new_examples.append((
                            np.array(ex.planes, dtype=np.float32).reshape(6, 6, 6),
                            np.array(ex.policy_target, dtype=np.float32),
                            ex.value_target,
                        ))
                    results[hr.winner] += 1
                    total_games += 1

            heuristic_time = time.time() - iter_start

            # Network self-play games (game loop in Rust, NN callback in Python)
            rust_net_engine = _RustMCTSEngine(self.simulations, self.c_puct)
            batch_eval_fn = mcts._batch_eval_fn
            for g in range(n_network):
                nr = rust_net_engine.play_network_game(
                    batch_eval_fn, batch_size=mcts.batch_size,
                    random_opening=4, max_moves=80,
                    temp_moves=15, temperature=self.temperature,
                )
                for ex in nr.examples:
                    new_examples.append((
                        np.array(ex.planes, dtype=np.float32).reshape(6, 6, 6),
                        np.array(ex.policy_target, dtype=np.float32),
                        ex.value_target,
                    ))
                results[nr.winner] += 1
                total_games += 1

            selfplay_time = time.time() - iter_start

            # Add to replay buffer
            self.replay_buffer.extend(new_examples)
            if len(self.replay_buffer) > self.buffer_max:
                self.replay_buffer = self.replay_buffer[-self.buffer_max:]

            # --- Training ---
            train_start = time.time()
            self.network.train()
            policy_loss, value_loss = self._train_epoch()
            self.scheduler.step()
            train_time = time.time() - train_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            if verbose:
                h_str = f" ({n_heuristic}h+{n_network}n)" if n_heuristic > 0 else ""
                pw_str = f" pw={self.policy_weight:.1f}" if self.policy_weight != 1.0 else ""
                print(
                    f"Iter {iteration:3d}/{iterations} | "
                    f"games: {self.games_per_iter}{h_str} "
                    f"(W:{results['white']} B:{results['black']} D:{results['draw']}) | "
                    f"buf: {len(self.replay_buffer):,} (train {min(len(self.replay_buffer), self.train_window):,}) | "
                    f"loss: p={policy_loss:.4f} v={value_loss:.4f}{pw_str} | "
                    f"lr={current_lr:.6f} | "
                    f"time: {selfplay_time:.0f}s play + {train_time:.0f}s train",
                    flush=True,
                )

            # --- Evaluation ---
            if eval_every > 0 and iteration % eval_every == 0:
                elo, elo_lo, elo_hi, w, d, l = self._evaluate(
                    eval_games, eval_depth
                )
                if verbose:
                    print(
                        f"  >> Eval vs heuristic (depth {eval_depth}): "
                        f"{w}W-{d}D-{l}L | "
                        f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]",
                        flush=True,
                    )
                if elo > best_elo:
                    best_elo = elo
                    self._save(save_path / "best_model.pt")
                    if verbose:
                        print(f"  >> New best ELO: {elo:+.0f}, saved.", flush=True)

            # Save periodic checkpoint
            if iteration % 10 == 0:
                self._save(save_path / f"model_iter{iteration}.pt")

        # Save final model
        self._save(save_path / "final_model.pt")
        if verbose:
            print(f"\nTraining complete. {total_games} total self-play games.")
            print(f"Best ELO vs heuristic: {best_elo:+.0f}")
            print(f"Models saved to {save_path}/")

    def _train_epoch(self) -> Tuple[float, float]:
        """Train on replay buffer for configured number of epochs.

        When the buffer exceeds train_window, samples a subset biased toward
        recent data (75% from newest half, 25% from oldest half). This keeps
        training time constant regardless of buffer size.

        Returns (avg_policy_loss, avg_value_loss) from last epoch.
        """
        if not self.replay_buffer:
            return 0.0, 0.0

        buf_size = len(self.replay_buffer)

        if buf_size <= self.train_window:
            # Buffer fits in window -- use everything
            indices = list(range(buf_size))
        else:
            # Sample train_window examples, biased toward recent data:
            # 75% from the newest half, 25% from the oldest half
            mid = buf_size // 2
            recent_pool = buf_size - mid  # size of newest half
            old_pool = mid                # size of oldest half
            n_recent = min(int(self.train_window * 0.75), recent_pool)
            n_old = min(self.train_window - n_recent, old_pool)
            recent_idx = np.random.choice(
                range(mid, buf_size), size=n_recent, replace=False
            ).tolist()
            old_idx = np.random.choice(
                range(0, mid), size=n_old, replace=False
            ).tolist()
            indices = recent_idx + old_idx

        # Build tensors from selected indices
        boards = np.array([self.replay_buffer[i][0] for i in indices])
        policies = np.array([self.replay_buffer[i][1] for i in indices])
        values = np.array([self.replay_buffer[i][2] for i in indices], dtype=np.float32)

        boards_t = torch.tensor(boards, device=self.device)
        policies_t = torch.tensor(policies, device=self.device)
        values_t = torch.tensor(values, device=self.device)

        dataset_size = len(indices)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        for epoch in range(self.training_epochs):
            # Shuffle
            perm = torch.randperm(dataset_size, device=self.device)
            boards_t = boards_t[perm]
            policies_t = policies_t[perm]
            values_t = values_t[perm]

            for i in range(0, dataset_size, self.batch_size):
                batch_boards = boards_t[i:i+self.batch_size]
                batch_policies = policies_t[i:i+self.batch_size]
                batch_values = values_t[i:i+self.batch_size]

                # Symmetry augmentation: mirror 50% of examples (#8)
                mirror_mask = torch.rand(batch_boards.size(0), device=self.device) < 0.5
                if mirror_mask.any():
                    # Mirror planes: flip last dimension (cols)
                    batch_boards[mirror_mask] = batch_boards[mirror_mask].flip(-1)
                    # Mirror policy using precomputed mapping
                    mirror_idx = torch.tensor(_POLICY_MIRROR, dtype=torch.long, device=self.device)
                    batch_policies[mirror_mask] = batch_policies[mirror_mask][:, mirror_idx]

                # Forward
                policy_logits, value_pred = self.network(batch_boards)

                # Policy loss: cross-entropy with MCTS visit distribution
                # -sum(target * log_softmax(logits))
                log_probs = F.log_softmax(policy_logits, dim=-1)
                policy_loss = -torch.sum(batch_policies * log_probs, dim=-1).mean()

                # Value loss: MSE
                value_loss = F.mse_loss(value_pred, batch_values)

                # Combined loss with policy weight (#6)
                loss = self.policy_weight * policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        avg_p = total_policy_loss / max(1, num_batches)
        avg_v = total_value_loss / max(1, num_batches)
        return avg_p, avg_v

    def _evaluate(
        self, num_games: int, opponent_depth: int
    ) -> Tuple[float, float, float, int, int, int]:
        """Play network MCTS vs alpha-beta engine (game loop in Rust).

        Returns (elo, elo_lo, elo_hi, wins, draws, losses).
        """
        self.network.eval()
        mcts = NetworkMCTS(
            self.network,
            simulations=self.simulations,
            c_puct=self.c_puct,
            device=self.device,
        )
        rust_engine = _RustMCTSEngine(self.simulations, self.c_puct)
        result = rust_engine.play_eval_match(
            mcts._batch_eval_fn,
            num_games=num_games,
            opponent_depth=opponent_depth,
            batch_size=mcts.batch_size,
            random_opening=2,
            max_moves=80,
        )
        wins, draws, losses = result.wins, result.draws, result.losses
        elo, elo_lo, elo_hi = elo_with_ci(wins, losses, draws)
        return elo, elo_lo, elo_hi, wins, draws, losses

    def _save(self, path: Path):
        """Save model checkpoint + replay buffer."""
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "buffer_size": len(self.replay_buffer),
            "network_type": self.network_type,
        }, path)
        # Save replay buffer alongside model (as compressed numpy)
        if self.replay_buffer:
            buf_path = Path(str(path) + ".buffer.npz")
            boards = np.array([ex[0] for ex in self.replay_buffer])
            policies = np.array([ex[1] for ex in self.replay_buffer])
            values = np.array([ex[2] for ex in self.replay_buffer], dtype=np.float32)
            np.savez_compressed(buf_path, boards=boards, policies=policies, values=values)

    def load(self, path: str, load_buffer: bool = True):
        """Load model checkpoint + optionally replay buffer."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        # Check network type matches
        saved_type = checkpoint.get("network_type", "mlp")
        if saved_type != self.network_type:
            print(f"  WARNING: checkpoint is {saved_type}, trainer is {self.network_type} -- loading buffer only")
            # Skip loading model weights and optimizer (architecture mismatch)
        else:
            self.network.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        # Load replay buffer if available
        buf_path = Path(str(path) + ".buffer.npz")
        if load_buffer and buf_path.exists():
            data = np.load(buf_path)
            boards = data["boards"]
            policies = data["policies"]
            values = data["values"]
            self.replay_buffer = [
                (boards[i], policies[i], float(values[i]))
                for i in range(len(values))
            ]
            print(f"Loaded model from {path} (buffer: {len(self.replay_buffer):,} examples)")
        else:
            print(f"Loaded model from {path} (no buffer)")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_game(network_mcts: NetworkMCTS, max_moves: int = 80):
    """Play and display a game using the network MCTS."""
    board = Board()

    # Random opening
    for _ in range(2):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        board.make_move(random.choice(moves))

    print("After 2 random opening moves:")
    print(board.display())
    print()

    move_count = 0
    while board.check_winner() is None and move_count < max_moves:
        move, info = network_mcts.search(board, add_noise=False)
        if move is None:
            break

        player = "White" if _is_white(board) else "Black"
        q_str = ""
        if info.get("children"):
            q_str = f" Q={info['children'][0]['q']:+.3f}"
        print(f"Move {move_count + 1} ({player}): {_safe_str(move)}  "
              f"[{info['visits']} visits{q_str}]")

        board.make_move(move)
        move_count += 1

        print(board.display())
        print()

    winner = board.check_winner()
    if winner is None:
        print(f"Draw after {move_count} moves")
    else:
        w = "White" if _is_white_winner(winner) else "Black"
        print(f"{w} wins after {move_count} moves")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero-style training for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m tonnesjakk.alphazero --iterations 20 --games-per-iter 50 --simulations 200
  python -m tonnesjakk.alphazero --evaluate alphazero_checkpoints/best_model.pt --games 50
  python -m tonnesjakk.alphazero --demo alphazero_checkpoints/best_model.pt --simulations 200
        """,
    )

    # Training
    parser.add_argument("--iterations", type=int, default=50,
                        help="Training iterations (default: 50)")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="Self-play games per iteration (default: 50)")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--training-epochs", type=int, default=5,
                        help="Training epochs per iteration (default: 5)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size (default: 256)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Network hidden size (default: 128)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                        help="PUCT exploration constant (default: 1.4)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Self-play temperature (default: 1.0)")
    parser.add_argument("--buffer-max", type=int, default=100000,
                        help="Max replay buffer size (default: 100000)")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "mlp"],
                        help="Network architecture (default: resnet)")
    parser.add_argument("--num-blocks", type=int, default=5,
                        help="Residual blocks for resnet (default: 5)")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Policy loss weight (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps (default: auto)")

    # Evaluation
    parser.add_argument("--eval-every", type=int, default=5,
                        help="Evaluate every N iterations (0=never, default: 5)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per evaluation (default: 20)")
    parser.add_argument("--eval-depth", type=int, default=5,
                        help="Opponent alpha-beta depth (default: 5)")

    # Modes
    parser.add_argument("--evaluate", type=str, default=None, metavar="MODEL",
                        help="Evaluate a saved model against heuristic")
    parser.add_argument("--demo", type=str, default=None, metavar="MODEL",
                        help="Play a demo game with a saved model")
    parser.add_argument("--resume", type=str, default=None, metavar="MODEL",
                        help="Resume training from a checkpoint")
    parser.add_argument("--games", type=int, default=50,
                        help="Games for --evaluate mode (default: 50)")
    parser.add_argument("--opponent-depth", type=int, default=5,
                        help="Opponent depth for --evaluate (default: 5)")
    parser.add_argument("--save-dir", type=str, default="alphazero_checkpoints",
                        help="Checkpoint directory (default: alphazero_checkpoints)")

    args = parser.parse_args()

    if args.evaluate:
        # Evaluate a saved model
        trainer = AlphaZeroTrainer(
            hidden=args.hidden,
            simulations=args.simulations,
            c_puct=args.c_puct,
            network_type=args.network,
            num_blocks=args.num_blocks,
            device=args.device,
        )
        trainer.load(args.evaluate)
        elo, elo_lo, elo_hi, w, d, l = trainer._evaluate(
            args.games, args.opponent_depth
        )
        print(f"Result: {w}W-{d}D-{l}L")
        print(f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]")

    elif args.demo:
        # Play a demo game
        net = make_network(args.network, hidden=args.hidden, num_blocks=args.num_blocks)
        checkpoint = torch.load(args.demo, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["model_state_dict"])
        net.eval()
        mcts = NetworkMCTS(net, simulations=args.simulations, c_puct=args.c_puct)
        demo_game(mcts)

    else:
        # Train
        print("=" * 60)
        print("ALPHAZERO TRAINING FOR TONNESJAKK")
        print("=" * 60)
        print(f"  Network: {args.network} (blocks={args.num_blocks}, channels={args.hidden})")
        print(f"  Iterations: {args.iterations}")
        print(f"  Games/iter: {args.games_per_iter}")
        print(f"  Simulations: {args.simulations}")
        print(f"  LR: {args.lr} (cosine annealing)")
        print(f"  Policy weight: {args.policy_weight}")
        print(f"  Device: {trainer.device}")
        print(f"  Eval every: {args.eval_every} iters (depth {args.eval_depth})")
        print(f"  Save dir: {args.save_dir}")
        print()

        trainer = AlphaZeroTrainer(
            hidden=args.hidden,
            simulations=args.simulations,
            c_puct=args.c_puct,
            lr=args.lr,
            games_per_iter=args.games_per_iter,
            training_epochs=args.training_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            buffer_max=args.buffer_max,
            network_type=args.network,
            num_blocks=args.num_blocks,
            policy_weight=args.policy_weight,
            device=args.device,
        )

        if args.resume:
            trainer.load(args.resume)

        trainer.run(
            iterations=args.iterations,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            eval_depth=args.eval_depth,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
