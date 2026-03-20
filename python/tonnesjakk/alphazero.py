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

Game loops (self-play, heuristic self-play, evaluation matches) run in Rust
via MCTSEngine methods (play_network_game, play_heuristic_games, play_eval_match).
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tonnesjakk import Board
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine
from tonnesjakk.utils import elo_with_ci


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_SIZE = 37 * 36  # 1332: from_idx (0-36) x to_idx (0-35)
BOARD_PLANES = 5       # 4 piece planes (my/opp relative encoding) + bias
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
    """Mirror board planes left-right. Input shape: (5, 6, 6) or (N, 5, 6, 6)."""
    return np.ascontiguousarray(planes[..., ::-1])


def mirror_policy(policy: np.ndarray) -> np.ndarray:
    """Mirror a policy vector using precomputed mapping. Shape: (1332,) or (N, 1332)."""
    return policy[..., _POLICY_MIRROR]


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class AlphaZeroNet(nn.Module):
    """Dual-headed network for AlphaZero.

    Architecture: MLP with shared trunk, separate policy and value heads.
    Input: 5x6x6 = 180 features (flattened board planes).
    Policy: 1332 logits (from x to square pairs).
    Value: scalar in [-1, +1] (White's perspective).
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        input_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE  # 180

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
            nn.LeakyReLU(0.01),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self.policy_head = nn.Linear(hidden, POLICY_SIZE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 5, 6, 6) or (batch, 180) board planes.

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

    Input: 5x6x6 = (5 planes, 6 rows, 6 cols) board representation.
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

        # Value head: 1x1 conv(4) -> BN -> ReLU -> flatten -> FC -> LeakyReLU -> Dropout -> FC -> tanh
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, channels)
        self.value_dropout = nn.Dropout(0.25)
        self.value_fc2 = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 5, 6, 6) or (batch, 180) board planes.

        Returns:
            (policy_logits, value): policy is (batch, 1332), value is (batch,).
        """
        # Reshape flat input to spatial: (batch, 5_planes, 6_rows, 6_cols)
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
        v = F.leaky_relu(self.value_fc1(v), 0.01)
        v = self.value_dropout(v)
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
        use_amp: bool = False,
    ):
        self.network = network
        self.simulations = simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device or torch.device("cpu")
        self.use_amp = use_amp
        self._engine = _RustMCTSEngine(simulations, c_puct)
        self._batch_eval_fn = self._make_batch_eval_fn()

    def _make_batch_eval_fn(self):
        """Create the batched Python callback for Rust MCTS leaf evaluation.

        Accepts a list of plane vectors (one per leaf), returns batched results.

        Optimizations:
        - Pre-allocated input buffer on device (eliminates per-call allocations)
        - Zero-copy numpy→torch via as_tensor
        - AMP (FP16) inference when enabled
        - Bulk C-level .numpy().tolist() for output conversion
        """
        net = self.network
        device = self.device
        use_amp = self.use_amp
        amp_dtype = torch.float16
        # Pre-allocate input buffer sized to max batch
        plane_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE
        input_buffer = torch.zeros(self.batch_size, plane_size, dtype=torch.float32, device=device)

        def batch_eval_fn(batch_planes: list) -> tuple:
            # batch_planes: list of N lists of plane_size floats
            n = len(batch_planes)
            # Zero-copy: list → numpy → torch, then copy to pre-allocated device buffer
            np_batch = np.array(batch_planes, dtype=np.float32)
            cpu_tensor = torch.as_tensor(np_batch)
            buf = input_buffer[:n]
            buf.copy_(cpu_tensor)

            with torch.no_grad():
                if use_amp:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        policy_logits, values = net(buf)
                    # Ensure FP32 output to Rust
                    policy_logits = policy_logits.float()
                    values = values.float()
                else:
                    policy_logits, values = net(buf)

            return policy_logits.cpu().numpy().tolist(), values.cpu().numpy().tolist()

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
# Training logger (JSONL)
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Append-only JSONL logger for training metrics.

    Writes one JSON object per line to ``{save_dir}/training.log``.
    Three record types:
      - ``config``    — training parameters, written once at start
      - ``iteration`` — per-iteration metrics (losses, game results, search_score stats)
      - ``eval``      — evaluation match results

    This log captures aggregate search_score statistics per iteration. For
    per-position search scores (e.g. to trace how the engine evaluated each
    move in a game), load the replay buffer directly::

        data = np.load("checkpoint.pt.buffer.npz")
        search_scores = data["search_scores"]  # one score per position
        values = data["values"]                 # game outcome per position
    """

    def __init__(self, save_dir: str):
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._path = path / "training.log"
        self._file = open(self._path, "a")

    def _write(self, record: dict):
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_config(self, **kwargs):
        self._write({"type": "config", "timestamp": time.time(), **kwargs})

    def log_iteration(self, iteration: int, *, game_results: dict,
                      buffer_size: int, policy_loss: float, value_loss: float,
                      lr: float, selfplay_time: float, train_time: float,
                      search_score_mean: float = 0.0, search_score_std: float = 0.0,
                      search_score_mae: float = 0.0, n_heuristic: int = 0,
                      n_network: int = 0, draws_dropped: int = 0):
        self._write({
            "type": "iteration",
            "timestamp": time.time(),
            "iteration": iteration,
            "game_results": game_results,
            "white_wins": game_results.get("white", 0),
            "black_wins": game_results.get("black", 0),
            "draws": game_results.get("draw", 0),
            "draws_dropped": draws_dropped,
            "buffer_size": buffer_size,
            "policy_loss": round(policy_loss, 6),
            "value_loss": round(value_loss, 6),
            "lr": lr,
            "selfplay_time": round(selfplay_time, 1),
            "train_time": round(train_time, 1),
            "search_score_mean": round(search_score_mean, 4),
            "search_score_std": round(search_score_std, 4),
            "search_score_mae": round(search_score_mae, 4),
            "n_heuristic": n_heuristic,
            "n_network": n_network,
        })

    def log_eval(self, iteration: int, *, wins: int, draws: int, losses: int,
                 elo: float, elo_lo: float, elo_hi: float, depth: int):
        self._write({
            "type": "eval",
            "timestamp": time.time(),
            "iteration": iteration,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "elo": round(elo, 1),
            "elo_lo": round(elo_lo, 1),
            "elo_hi": round(elo_hi, 1),
            "depth": depth,
        })

    def close(self):
        self._file.close()


# ---------------------------------------------------------------------------
# Game result helpers
# ---------------------------------------------------------------------------

# Per-game result: (outcome, examples) where outcome is "white"/"black"/"draw"
# and examples is a list of (planes, policy, value_target, search_score) tuples.
GameResult = Tuple[str, List[Tuple[np.ndarray, np.ndarray, float, float]]]


def _filter_draws(games: List[GameResult], max_draw_fraction: float,
                  rng: Optional[np.random.Generator] = None,
                  ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, float]],
                             Dict[str, int], int]:
    """Filter per-game results to cap the fraction of draw games.

    Returns (examples, results_dict, draws_dropped).
    """
    if max_draw_fraction >= 1.0:
        # No filtering — flatten and return
        examples = []
        results: Dict[str, int] = {"white": 0, "black": 0, "draw": 0}
        for outcome, exs in games:
            examples.extend(exs)
            results[outcome] += 1
        return examples, results, 0

    decisive = [(o, exs) for o, exs in games if o != "draw"]
    draws = [(o, exs) for o, exs in games if o == "draw"]

    n_decisive = len(decisive)
    if n_decisive == 0:
        # All draws — keep max_draw_fraction of them
        keep = max(1, int(len(draws) * max_draw_fraction))
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(draws)
        kept_draws = draws[:keep]
        dropped = len(draws) - keep
    else:
        # Allow up to max_draw_fraction of total
        # n_draws / (n_decisive + n_draws) <= max_draw_fraction
        # n_draws <= max_draw_fraction * n_decisive / (1 - max_draw_fraction)
        max_draws = int(max_draw_fraction * n_decisive / (1.0 - max_draw_fraction))
        max_draws = max(max_draws, 0)
        if len(draws) <= max_draws:
            kept_draws = draws
            dropped = 0
        else:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(draws)
            kept_draws = draws[:max_draws]
            dropped = len(draws) - max_draws

    examples = []
    results = {"white": 0, "black": 0, "draw": 0}
    for outcome, exs in decisive + kept_draws:
        examples.extend(exs)
        results[outcome] += 1
    return examples, results, dropped


# ---------------------------------------------------------------------------
# Multiprocessing workers (top-level for pickling)
# ---------------------------------------------------------------------------

def _play_alphabeta_games_worker(args):
    """Worker for parallel alpha-beta game generation. Pure Rust, no model needed."""
    num_games, depth, random_opening, max_moves, simulations, c_puct, value_blend_lambda, adjudication_threshold, adjudication_min_moves = args

    engine = _RustMCTSEngine(simulations, c_puct,
                             value_blend_lambda=value_blend_lambda,
                             adjudication_threshold=adjudication_threshold,
                             adjudication_min_moves=adjudication_min_moves)
    results = engine.play_alphabeta_games(num_games, depth=depth,
                                          random_opening=random_opening,
                                          max_moves=max_moves)

    games = []  # list of (outcome, examples) per game
    for r in results:
        examples = []
        for ex in r.examples:
            examples.append((
                np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                np.array(ex.policy_target, dtype=np.float32),
                ex.value_target,
                ex.search_score,
            ))
        games.append((r.winner, examples))

    return games


def _play_network_games_worker(args):
    """Worker function for parallel network self-play.

    Must be top-level (not a method) so it's picklable with spawn start method.
    Each worker creates its own model (CPU) + MCTSEngine, plays games sequentially,
    and returns per-game grouped results for draw filtering.
    """
    (
        num_games, model_state_dict, network_type, hidden, num_blocks,
        simulations, c_puct, mcts_batch_size, temperature,
        random_opening, max_moves, temp_moves,
        full_search_fraction, cheap_sims,
        value_blend_lambda, adjudication_threshold, adjudication_min_moves,
        use_gumbel, forward_only, repetition_penalty,
    ) = args

    # Create fresh model on CPU
    net = make_network(network_type, hidden=hidden, num_blocks=num_blocks)
    net.load_state_dict(model_state_dict)
    net.eval()
    device = torch.device("cpu")
    net.to(device)

    # Create batch eval function (same as NetworkMCTS._make_batch_eval_fn but inline)
    plane_size = BOARD_PLANES * BOARD_SIZE * BOARD_SIZE
    input_buffer = torch.zeros(mcts_batch_size, plane_size, dtype=torch.float32, device=device)

    def batch_eval_fn(batch_planes: list) -> tuple:
        n = len(batch_planes)
        np_batch = np.array(batch_planes, dtype=np.float32)
        cpu_tensor = torch.as_tensor(np_batch)
        buf = input_buffer[:n]
        buf.copy_(cpu_tensor)
        with torch.no_grad():
            policy_logits, values = net(buf)
        return policy_logits.numpy().tolist(), values.numpy().tolist()

    engine = _RustMCTSEngine(simulations, c_puct,
                             value_blend_lambda=value_blend_lambda,
                             adjudication_threshold=adjudication_threshold,
                             adjudication_min_moves=adjudication_min_moves,
                             use_gumbel=use_gumbel,
                             forward_only=forward_only,
                             repetition_penalty=repetition_penalty)

    games = []  # list of (outcome, examples) per game
    for _ in range(num_games):
        nr = engine.play_network_game(
            batch_eval_fn, batch_size=mcts_batch_size,
            random_opening=random_opening, max_moves=max_moves,
            temp_moves=temp_moves, temperature=temperature,
            full_search_fraction=full_search_fraction,
            cheap_sims=cheap_sims,
        )
        examples = []
        for ex in nr.examples:
            examples.append((
                np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                np.array(ex.policy_target, dtype=np.float32),
                ex.value_target,
                ex.search_score,
            ))
        games.append((nr.winner, examples))

    return games


def _play_onnx_games_worker(args):
    """Worker function for parallel ONNX self-play.

    Runs entirely in Rust — no Python model, no GIL contention.
    Each worker loads the ONNX model and plays games via MCTSEngine.
    """
    from tonnesjakk._core import OnnxSession as _OnnxSession

    (
        num_games, onnx_path, use_coreml,
        simulations, c_puct, mcts_batch_size, temperature,
        random_opening, max_moves, temp_moves,
        full_search_fraction, cheap_sims,
        value_blend_lambda, adjudication_threshold, adjudication_min_moves,
        use_gumbel, forward_only, repetition_penalty,
    ) = args

    onnx_session = _OnnxSession(onnx_path, use_coreml=use_coreml)
    engine = _RustMCTSEngine(simulations, c_puct,
                             value_blend_lambda=value_blend_lambda,
                             adjudication_threshold=adjudication_threshold,
                             adjudication_min_moves=adjudication_min_moves,
                             use_gumbel=use_gumbel,
                             forward_only=forward_only,
                             repetition_penalty=repetition_penalty)

    games = []  # list of (outcome, examples) per game
    for _ in range(num_games):
        nr = engine.play_network_game_onnx(
            onnx_session, batch_size=mcts_batch_size,
            random_opening=random_opening, max_moves=max_moves,
            temp_moves=temp_moves, temperature=temperature,
            full_search_fraction=full_search_fraction,
            cheap_sims=cheap_sims,
        )
        examples = []
        for ex in nr.examples:
            examples.append((
                np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                np.array(ex.policy_target, dtype=np.float32),
                ex.value_target,
                ex.search_score,
            ))
        games.append((nr.winner, examples))

    return games


def _play_mixed_games_worker(args):
    """Worker for parallel network-vs-heuristic training games.

    Each game alternates which side the network plays. Collects training
    examples only from the network's turns.
    """
    from tonnesjakk._core import OnnxSession as _OnnxSession

    (
        num_games, onnx_path, use_coreml,
        simulations, c_puct, mcts_batch_size, temperature,
        random_opening, max_moves, temp_moves,
        opponent_depth,
        value_blend_lambda, adjudication_threshold, adjudication_min_moves,
        use_gumbel, forward_only, repetition_penalty,
    ) = args

    onnx_session = _OnnxSession(onnx_path, use_coreml=use_coreml)
    engine = _RustMCTSEngine(simulations, c_puct,
                             value_blend_lambda=value_blend_lambda,
                             adjudication_threshold=adjudication_threshold,
                             adjudication_min_moves=adjudication_min_moves,
                             use_gumbel=use_gumbel,
                             forward_only=forward_only,
                             repetition_penalty=repetition_penalty)

    games = []
    for game_idx in range(num_games):
        mcts_is_white = (game_idx % 2 == 0)
        nr = engine.play_mixed_game_onnx(
            onnx_session,
            opponent_depth=opponent_depth,
            batch_size=mcts_batch_size,
            random_opening=random_opening,
            max_moves=max_moves,
            temp_moves=temp_moves,
            temperature=temperature,
            mcts_is_white=mcts_is_white,
        )
        examples = []
        for ex in nr.examples:
            examples.append((
                np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                np.array(ex.policy_target, dtype=np.float32),
                ex.value_target,
                ex.search_score,
            ))
        games.append((nr.winner, examples))

    return games


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
        hidden: int = 64,
        simulations: int = 200,       # 200 sufficient for branching factor ~16 (Wang et al. 2020)
        c_puct: float = 1.0,
        lr: float = 0.001,
        games_per_iter: int = 100,    # maximize outer self-play loop (Wang et al. 2020)
        training_epochs: int = 5,     # 3-5 to avoid overfitting (Wang et al. 2020)
        batch_size: int = 256,
        temperature: float = 1.0,
        buffer_max: int = 200000,
        buffer_min: int = 20000,
        train_window: int = 20000,
        network_type: str = "resnet",
        num_blocks: int = 5,
        policy_weight: float = 0.5,   # value-heavy loss for small games (Wang & Emmerich 2019)
        device: str = "auto",
        mcts_batch_size: int = 8,
        use_amp: bool = False,
        num_workers: int = 1,
        full_search_fraction: float = 1.0,
        cheap_sims: int = 50,
        gate_threshold: float = 0.0,
        temp_moves: int = 3,
        value_blend_lambda: float = 0.5,
        adjudication_threshold: float = 0.6,
        adjudication_min_moves: int = 30,
        max_moves: int = 80,
        use_gumbel: bool = False,
        forward_only: bool = False,
        repetition_penalty: float = 0.0,
        max_draw_fraction: float = 1.0,
        eval_simulations: Optional[int] = None,
        mixed_opponent_depth: int = 0,
        mixed_fraction: float = 0.0,
    ):
        self.network_type = network_type
        self.hidden = hidden
        self.num_blocks = num_blocks
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
        self.buffer_min = buffer_min
        self.train_window = train_window
        self.policy_weight = policy_weight
        self.mcts_batch_size = mcts_batch_size
        self.use_amp = use_amp
        self.num_workers = num_workers
        self.full_search_fraction = full_search_fraction
        self.cheap_sims = cheap_sims
        self.gate_threshold = gate_threshold
        self.temp_moves = temp_moves
        self.value_blend_lambda = value_blend_lambda
        self.adjudication_threshold = adjudication_threshold
        self.adjudication_min_moves = adjudication_min_moves
        self.max_moves = max_moves
        self.use_gumbel = use_gumbel
        self.forward_only = forward_only
        self.repetition_penalty = repetition_penalty
        self.max_draw_fraction = max_draw_fraction
        self.eval_simulations = eval_simulations  # None = use self.simulations
        self.mixed_opponent_depth = mixed_opponent_depth  # 0 = disabled
        self.mixed_fraction = mixed_fraction  # fraction of games that are network-vs-heuristic

        self._current_iteration = 0
        self._total_iterations = 0

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=lr * 0.01
        )
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []

        amp_str = "FP16" if self.use_amp else "FP32"
        workers_str = f", workers={self.num_workers}" if self.num_workers > 1 else ""
        print(f"AlphaZero network: {self.network.num_parameters:,} parameters ({self.device}, {amp_str}, mcts_batch={self.mcts_batch_size}{workers_str})")

    def _make_rust_engine(self) -> '_RustMCTSEngine':
        """Create a Rust MCTSEngine with current training parameters."""
        return _RustMCTSEngine(
            self.simulations, self.c_puct,
            value_blend_lambda=self.value_blend_lambda,
            adjudication_threshold=self.adjudication_threshold,
            adjudication_min_moves=self.adjudication_min_moves,
            use_gumbel=self.use_gumbel,
            forward_only=self.forward_only,
            repetition_penalty=self.repetition_penalty,
        )

    def set_lr_schedule(self, total_steps: int):
        """Update LR schedule T_max for known total iteration count."""
        self._total_iterations = total_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=self.lr * 0.01
        )

    def _effective_buffer_max(self) -> int:
        """Growing replay buffer: starts at buffer_min, grows linearly to buffer_max."""
        if self._total_iterations <= 0 or self.buffer_min >= self.buffer_max:
            return self.buffer_max
        progress = min(1.0, self._current_iteration / self._total_iterations)
        return int(self.buffer_min + (self.buffer_max - self.buffer_min) * progress)

    def _play_network_games_parallel(self, n_network: int):
        """Play network self-play games in parallel using multiprocessing.

        Splits games across workers, each with its own model copy on CPU.
        Returns (new_examples, results) matching the sequential path.
        """
        # Distribute games across workers
        per_worker = n_network // self.num_workers
        remainder = n_network % self.num_workers
        game_counts = [per_worker + (1 if i < remainder else 0) for i in range(self.num_workers)]
        # Filter out workers with 0 games
        game_counts = [c for c in game_counts if c > 0]

        # Copy model weights to CPU once
        model_state_dict = {k: v.cpu() for k, v in self.network.state_dict().items()}

        worker_args = [
            (
                count, model_state_dict, self.network_type, self.hidden, self.num_blocks,
                self.simulations, self.c_puct, self.mcts_batch_size, self.temperature,
                4, self.max_moves, self.temp_moves,  # random_opening, max_moves, temp_moves
                self.full_search_fraction, self.cheap_sims,
                self.value_blend_lambda, self.adjudication_threshold, self.adjudication_min_moves,
                self.use_gumbel, self.forward_only, self.repetition_penalty,
            )
            for count in game_counts
        ]

        with multiprocessing.Pool(processes=len(game_counts)) as pool:
            results_list = pool.map(_play_network_games_worker, worker_args)

        # Merge per-game results from all workers
        all_games = []
        for worker_games in results_list:
            all_games.extend(worker_games)

        return all_games

    def _play_onnx_games_parallel(self, n_network: int, onnx_path: str):
        """Play network self-play games in parallel using ONNX (pure Rust).

        Each worker loads the ONNX model independently — no Python model,
        no GIL contention, no PyTorch overhead.
        """
        per_worker = n_network // self.num_workers
        remainder = n_network % self.num_workers
        game_counts = [per_worker + (1 if i < remainder else 0) for i in range(self.num_workers)]
        game_counts = [c for c in game_counts if c > 0]

        worker_args = [
            (
                count, onnx_path, False,  # use_coreml=False (CPU for workers)
                self.simulations, self.c_puct, self.mcts_batch_size, self.temperature,
                4, self.max_moves, self.temp_moves,  # random_opening, max_moves, temp_moves
                self.full_search_fraction, self.cheap_sims,
                self.value_blend_lambda, self.adjudication_threshold, self.adjudication_min_moves,
                self.use_gumbel, self.forward_only, self.repetition_penalty,
            )
            for count in game_counts
        ]

        with multiprocessing.Pool(processes=len(game_counts)) as pool:
            results_list = pool.map(_play_onnx_games_worker, worker_args)

        # Merge per-game results from all workers
        all_games = []
        for worker_games in results_list:
            all_games.extend(worker_games)

        return all_games

    def _play_mixed_games_parallel(self, n_games: int, onnx_path: str):
        """Play network-vs-heuristic training games in parallel using ONNX."""
        per_worker = n_games // self.num_workers
        remainder = n_games % self.num_workers
        game_counts = [per_worker + (1 if i < remainder else 0) for i in range(self.num_workers)]
        game_counts = [c for c in game_counts if c > 0]

        worker_args = [
            (
                count, onnx_path, False,
                self.simulations, self.c_puct, self.mcts_batch_size, self.temperature,
                4, self.max_moves, self.temp_moves,
                self.mixed_opponent_depth,
                self.value_blend_lambda, self.adjudication_threshold, self.adjudication_min_moves,
                self.use_gumbel, self.forward_only, self.repetition_penalty,
            )
            for count in game_counts
        ]

        with multiprocessing.Pool(processes=len(game_counts)) as pool:
            results_list = pool.map(_play_mixed_games_worker, worker_args)

        all_games = []
        for worker_games in results_list:
            all_games.extend(worker_games)
        return all_games

    def _play_mixed_games_sequential(self, n_games: int, onnx_path: str):
        """Play network-vs-heuristic training games sequentially using ONNX."""
        from tonnesjakk._core import OnnxSession as _OnnxSession

        onnx_session = _OnnxSession(onnx_path, use_coreml=False)
        engine = self._make_rust_engine()

        games = []
        for game_idx in range(n_games):
            mcts_is_white = (game_idx % 2 == 0)
            nr = engine.play_mixed_game_onnx(
                onnx_session,
                opponent_depth=self.mixed_opponent_depth,
                batch_size=self.mcts_batch_size,
                random_opening=4,
                max_moves=self.max_moves,
                temp_moves=self.temp_moves,
                temperature=self.temperature,
                mcts_is_white=mcts_is_white,
            )
            examples = []
            for ex in nr.examples:
                examples.append((
                    np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                    np.array(ex.policy_target, dtype=np.float32),
                    ex.value_target,
                    ex.search_score,
                ))
            games.append((nr.winner, examples))

        return games

    def _play_onnx_games_sequential(self, n_network: int, onnx_path: str):
        """Play network self-play games sequentially using ONNX (pure Rust)."""
        from tonnesjakk._core import OnnxSession as _OnnxSession

        onnx_session = _OnnxSession(onnx_path, use_coreml=False)
        engine = self._make_rust_engine()

        games = []
        for _ in range(n_network):
            nr = engine.play_network_game_onnx(
                onnx_session, batch_size=self.mcts_batch_size,
                random_opening=4, max_moves=self.max_moves,
                temp_moves=self.temp_moves, temperature=self.temperature,
                full_search_fraction=self.full_search_fraction,
                cheap_sims=self.cheap_sims,
            )
            examples = []
            for ex in nr.examples:
                examples.append((
                    np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                    np.array(ex.policy_target, dtype=np.float32),
                    ex.value_target,
                    ex.search_score,
                ))
            games.append((nr.winner, examples))

        return games

    def generate_bootstrap_games(self, num_games: int, depth: int = 7,
                                  random_opening: int = 4, max_moves: int = 80):
        """Generate alpha-beta self-play games for bootstrapping.

        Uses multiprocessing to generate games in parallel. Each game uses
        alpha-beta search (not MCTS) with soft policy targets from heuristic
        evaluation of all moves. Results go directly into the replay buffer.
        """
        workers = max(1, self.num_workers)
        # Generate in batches so we can print progress
        batch_size = workers * 50  # ~50 games per worker per batch
        games_done = 0
        total_examples = 0
        draws_dropped = 0
        game_results = {"white": 0, "black": 0, "draw": 0}

        t0 = time.time()
        print(f"Generating {num_games} alpha-beta bootstrap games (depth {depth}, {workers} workers)...")

        while games_done < num_games:
            batch = min(batch_size, num_games - games_done)
            per_worker = batch // workers
            remainder = batch % workers
            game_counts = [per_worker + (1 if i < remainder else 0) for i in range(workers)]
            game_counts = [c for c in game_counts if c > 0]

            worker_args = [
                (count, depth, random_opening, max_moves, self.simulations, self.c_puct,
                 self.value_blend_lambda, self.adjudication_threshold, self.adjudication_min_moves)
                for count in game_counts
            ]

            if len(game_counts) == 1:
                results_list = [_play_alphabeta_games_worker(worker_args[0])]
            else:
                with multiprocessing.Pool(processes=len(game_counts)) as pool:
                    results_list = pool.map(_play_alphabeta_games_worker, worker_args)

            # Merge per-game results from all workers
            batch_games = []
            for worker_games in results_list:
                batch_games.extend(worker_games)

            # Filter draws
            examples, batch_results, dropped = _filter_draws(
                batch_games, self.max_draw_fraction)
            self.replay_buffer.extend(examples)
            total_examples += len(examples)
            draws_dropped += dropped
            for k in game_results:
                game_results[k] += batch_results[k]

            games_done += batch
            elapsed = time.time() - t0
            rate = games_done / elapsed
            remaining = (num_games - games_done) / rate if rate > 0 else 0
            print(f"  {games_done:,}/{num_games:,} games ({rate:.1f}/s) | "
                  f"W:{game_results['white']} B:{game_results['black']} D:{game_results['draw']} | "
                  f"{total_examples:,} examples | ~{remaining:.0f}s remaining",
                  flush=True)

        effective_max = self._effective_buffer_max()
        if len(self.replay_buffer) > effective_max:
            self.replay_buffer = self.replay_buffer[-effective_max:]

        elapsed = time.time() - t0
        drop_str = f" (dropped {draws_dropped} draw games)" if draws_dropped > 0 else ""
        print(f"Bootstrap complete: {num_games} games in {elapsed/60:.1f}m | "
              f"buffer: {len(self.replay_buffer):,} examples{drop_str}")

    def run(
        self,
        iterations: int = 50,
        eval_every: int = 5,
        eval_games: int = 20,
        eval_depth: int = 5,
        save_dir: str = "alphazero_checkpoints",
        heuristic_ratio: float = 0.0,
        verbose: bool = True,
        logger: Optional["TrainingLogger"] = None,
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

        if not hasattr(self, '_best_elo'):
            self._best_elo = -400.0
        total_games = 0

        for iteration in range(1, iterations + 1):
            self._current_iteration += 1
            iter_start = time.time()

            # --- Self-play ---
            self.network.eval()

            # Export ONNX model for pure-Rust inference (no Python/GIL in self-play)
            onnx_path = str(save_path / "_current.onnx")
            self.export_onnx(onnx_path)

            all_games: List[GameResult] = []
            n_heuristic = int(self.games_per_iter * heuristic_ratio)
            n_remaining = self.games_per_iter - n_heuristic
            # Split remaining games between self-play and mixed (network vs heuristic)
            n_mixed = int(n_remaining * self.mixed_fraction) if self.mixed_opponent_depth > 0 else 0
            n_network = n_remaining - n_mixed

            # Alpha-beta self-play games (fully in Rust, strong + balanced)
            if n_heuristic > 0:
                rust_engine = self._make_rust_engine()
                ab_results = rust_engine.play_alphabeta_games(
                    n_heuristic, depth=7, random_opening=4, max_moves=self.max_moves,
                )
                for ar in ab_results:
                    examples = []
                    for ex in ar.examples:
                        examples.append((
                            np.array(ex.planes, dtype=np.float32).reshape(BOARD_PLANES, BOARD_SIZE, BOARD_SIZE),
                            np.array(ex.policy_target, dtype=np.float32),
                            ex.value_target,
                            ex.search_score,
                        ))
                    all_games.append((ar.winner, examples))

            # Network-vs-heuristic training games (network learns from playing a stronger opponent)
            if n_mixed > 0:
                if self.num_workers > 1:
                    mixed_games = self._play_mixed_games_parallel(n_mixed, onnx_path)
                else:
                    mixed_games = self._play_mixed_games_sequential(n_mixed, onnx_path)
                all_games.extend(mixed_games)

            # Network self-play games via ONNX (pure Rust, no Python/GIL)
            if n_network > 0:
                if self.num_workers > 1:
                    net_games = self._play_onnx_games_parallel(n_network, onnx_path)
                else:
                    net_games = self._play_onnx_games_sequential(n_network, onnx_path)
                all_games.extend(net_games)

            selfplay_time = time.time() - iter_start

            # Filter draws and flatten to examples
            total_games += len(all_games)
            new_examples, results, draws_dropped = _filter_draws(
                all_games, self.max_draw_fraction)

            # Add to replay buffer (growing buffer: starts small, grows to buffer_max)
            self.replay_buffer.extend(new_examples)
            effective_max = self._effective_buffer_max()
            if len(self.replay_buffer) > effective_max:
                self.replay_buffer = self.replay_buffer[-effective_max:]

            # --- Training ---
            train_start = time.time()
            self.network.train()
            policy_loss, value_loss = self._train_epoch()
            self.scheduler.step()
            train_time = time.time() - train_start

            current_lr = self.optimizer.param_groups[0]["lr"]
            if verbose:
                parts = []
                if n_heuristic > 0:
                    parts.append(f"{n_heuristic}h")
                if n_mixed > 0:
                    parts.append(f"{n_mixed}mx")
                parts.append(f"{n_network}n")
                h_str = f" ({'+'.join(parts)})" if (n_heuristic > 0 or n_mixed > 0) else ""
                pw_str = f" pw={self.policy_weight:.1f}" if self.policy_weight != 1.0 else ""
                drop_str = f" -{draws_dropped}D" if draws_dropped > 0 else ""
                print(
                    f"Iter {iteration:3d}/{iterations} | "
                    f"games: {self.games_per_iter}{h_str} "
                    f"(W:{results['white']} B:{results['black']} D:{results['draw']}{drop_str}) | "
                    f"buf: {len(self.replay_buffer):,} (train {min(len(self.replay_buffer), self.train_window):,}) | "
                    f"loss: p={policy_loss:.4f} v={value_loss:.4f}{pw_str} | "
                    f"lr={current_lr:.6f} | "
                    f"time: {selfplay_time:.0f}s play + {train_time:.0f}s train",
                    flush=True,
                )

            # Compute search_score stats for logging
            ss_mean = ss_std = ss_mae = 0.0
            if new_examples:
                ss = np.array([ex[3] for ex in new_examples], dtype=np.float32)
                vt = np.array([ex[2] for ex in new_examples], dtype=np.float32)
                ss_mean = float(np.mean(ss))
                ss_std = float(np.std(ss))
                ss_mae = float(np.mean(np.abs(ss - vt)))

            if logger:
                logger.log_iteration(
                    iteration, game_results=results,
                    buffer_size=len(self.replay_buffer),
                    policy_loss=policy_loss, value_loss=value_loss,
                    lr=current_lr, selfplay_time=selfplay_time,
                    train_time=train_time,
                    search_score_mean=ss_mean, search_score_std=ss_std,
                    search_score_mae=ss_mae,
                    n_heuristic=n_heuristic, n_network=n_network,
                    draws_dropped=draws_dropped,
                )

            # --- Evaluation ---
            if eval_every > 0 and iteration % eval_every == 0:
                elo, elo_lo, elo_hi, w, d, l = self._evaluate(
                    eval_games, eval_depth, onnx_path=onnx_path,
                )
                if verbose:
                    eval_sims = self.eval_simulations or self.simulations
                    print(
                        f"  >> Eval vs heuristic (depth {eval_depth}, {eval_sims} sims): "
                        f"{w}W-{d}D-{l}L | "
                        f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]",
                        flush=True,
                    )
                if logger:
                    logger.log_eval(
                        iteration, wins=w, draws=d, losses=l,
                        elo=elo, elo_lo=elo_lo, elo_hi=elo_hi, depth=eval_depth,
                    )
                if elo > self._best_elo:
                    self._best_elo = elo
                    self._save(save_path / "best_model.pt")
                    if verbose:
                        print(f"  >> New best ELO: {elo:+.0f}, saved.", flush=True)

                # Model gating: revert to best checkpoint if win rate drops
                if self.gate_threshold > 0:
                    total = w + d + l
                    win_rate = (w + 0.5 * d) / total if total > 0 else 0.5
                    if win_rate < self.gate_threshold:
                        best_path = save_path / "best_model.pt"
                        if best_path.exists():
                            self.load(str(best_path), load_buffer=False)
                            if verbose:
                                print(f"  >> Gating: win rate {win_rate:.1%} < {self.gate_threshold:.1%}, "
                                      f"reverted to best model.", flush=True)

            # Save periodic checkpoint
            if iteration % 10 == 0:
                self._save(save_path / f"model_iter{iteration}.pt")

        # Save final model
        self._save(save_path / "final_model.pt")
        if verbose:
            print(f"\nTraining complete. {total_games} total self-play games.")
            print(f"Best ELO vs heuristic: {self._best_elo:+.0f}")
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

                # Forward (with optional AMP)
                if self.use_amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        policy_logits, value_pred = self.network(batch_boards)
                        log_probs = F.log_softmax(policy_logits, dim=-1)
                        policy_loss = -torch.sum(batch_policies * log_probs, dim=-1).mean()
                        value_loss = F.mse_loss(value_pred, batch_values)
                        loss = self.policy_weight * policy_loss + value_loss
                else:
                    policy_logits, value_pred = self.network(batch_boards)
                    log_probs = F.log_softmax(policy_logits, dim=-1)
                    policy_loss = -torch.sum(batch_policies * log_probs, dim=-1).mean()
                    value_loss = F.mse_loss(value_pred, batch_values)
                    loss = self.policy_weight * policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

        avg_p = total_policy_loss / max(1, num_batches)
        avg_v = total_value_loss / max(1, num_batches)
        return avg_p, avg_v

    def _evaluate(
        self, num_games: int, opponent_depth: int,
        onnx_path: Optional[str] = None,
    ) -> Tuple[float, float, float, int, int, int]:
        """Play network MCTS vs alpha-beta engine (game loop in Rust).

        If onnx_path is provided, uses ONNX inference (pure Rust, no Python).
        Returns (elo, elo_lo, elo_hi, wins, draws, losses).
        """
        # Eval can use a different sim budget than training
        eval_sims = self.eval_simulations or self.simulations
        # Eval always uses full move set (forward_only=False)
        rust_engine = _RustMCTSEngine(
            eval_sims, self.c_puct,
            value_blend_lambda=self.value_blend_lambda,
            adjudication_threshold=self.adjudication_threshold,
            adjudication_min_moves=self.adjudication_min_moves,
            use_gumbel=self.use_gumbel,
            forward_only=False,
            repetition_penalty=self.repetition_penalty,
        )
        if onnx_path:
            from tonnesjakk._core import OnnxSession as _OnnxSession
            onnx_session = _OnnxSession(onnx_path, use_coreml=False)
            result = rust_engine.play_eval_match_onnx(
                onnx_session,
                num_games=num_games,
                opponent_depth=opponent_depth,
                batch_size=self.mcts_batch_size,
                random_opening=2,
                max_moves=80,
            )
        else:
            self.network.eval()
            mcts = NetworkMCTS(
                self.network,
                simulations=eval_sims,
                c_puct=self.c_puct,
                batch_size=self.mcts_batch_size,
                device=self.device,
                use_amp=self.use_amp,
            )
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
        """Save model checkpoint + replay buffer.

        The buffer sidecar (``<path>.buffer.npz``) contains arrays:
        ``boards``, ``policies``, ``values``, and ``search_scores``.
        ``search_scores`` holds the per-position search evaluation (current
        player perspective) — useful for analysing how the engine assessed
        each position during self-play.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
            search_scores = np.array([ex[3] if len(ex) > 3 else 0.0 for ex in self.replay_buffer], dtype=np.float32)
            np.savez_compressed(buf_path, boards=boards, policies=policies, values=values, search_scores=search_scores)

    def export_onnx(self, path: str):
        """Export current network to ONNX format for Rust-side inference.

        The exported model takes input shape [batch, 5, 6, 6] and produces:
          - output 0: policy logits [batch, 1332]
          - output 1: value [batch]
        """
        import io
        import os
        import sys
        import warnings
        import logging
        self.network.eval()
        dummy = torch.randn(1, BOARD_PLANES, BOARD_SIZE, BOARD_SIZE, device=self.device)
        # Suppress all ONNX export output (stdout, stderr, warnings, logging)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logging.disable(logging.CRITICAL)
            try:
                torch.onnx.export(
                    self.network,
                    dummy,
                    path,
                    input_names=["planes"],
                    output_names=["policy", "value"],
                    dynamic_axes={
                        "planes": {0: "batch"},
                        "policy": {0: "batch"},
                        "value": {0: "batch"},
                    },
                    opset_version=18,
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                logging.disable(logging.NOTSET)

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
            search_scores = data["search_scores"] if "search_scores" in data else np.zeros_like(values)
            self.replay_buffer = [
                (boards[i], policies[i], float(values[i]), float(search_scores[i]))
                for i in range(len(values))
            ]
            print(f"Loaded model from {path} (buffer: {len(self.replay_buffer):,} examples)")
        else:
            print(f"Loaded model from {path} (no buffer)")


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
    parser.add_argument("--mcts-batch-size", type=int, default=8,
                        help="MCTS evaluation batch size (default: 8)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (FP16) inference and training")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps (default: auto)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel self-play workers (default: 1)")

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
            mcts_batch_size=args.mcts_batch_size,
            use_amp=args.amp,
            num_workers=args.workers,
        )
        trainer.load(args.evaluate)
        elo, elo_lo, elo_hi, w, d, l = trainer._evaluate(
            args.games, args.opponent_depth
        )
        print(f"Result: {w}W-{d}D-{l}L")
        print(f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]")

    else:
        # Train
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
            mcts_batch_size=args.mcts_batch_size,
            use_amp=args.amp,
            num_workers=args.workers,
        )

        print("=" * 60)
        print("ALPHAZERO TRAINING FOR TONNESJAKK")
        print("=" * 60)
        print(f"  Network: {args.network} (blocks={args.num_blocks}, channels={args.hidden})")
        print(f"  Iterations: {args.iterations}")
        print(f"  Games/iter: {args.games_per_iter}")
        print(f"  Simulations: {args.simulations}")
        print(f"  Workers: {args.workers}")
        print(f"  LR: {args.lr} (cosine annealing)")
        print(f"  Policy weight: {args.policy_weight}")
        print(f"  Device: {trainer.device}")
        print(f"  Eval every: {args.eval_every} iters (depth {args.eval_depth})")
        print(f"  Save dir: {args.save_dir}")
        print()

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
