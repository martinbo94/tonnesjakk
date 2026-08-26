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
    python -m tonnesjakk.nnue --generate-only --games 150000 --depth 8 --workers 4 --save-data data.bin
    python -m tonnesjakk.nnue --load-data data.bin --feature-set plain --mirror --epochs 50 --output runs/x
    (strength testing: scripts/match.py --nnue-a runs/x/nnue_weights.json --time-a 100 --time-b 100)
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

# Model / trainer / export live in nnue_arch.py (generic over architectures);
# this module owns data generation, the CLI, and comparison utilities.
from .nnue_arch import NnueArch, train_sparse_model, export_sparse_json

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

    def __init__(self, nnue_path: Optional[str] = None, tb_dir: Optional[str] = None):
        """
        Initialize the data generator.

        Args:
            nnue_path: Path to NNUE weights JSON file. If provided, uses NNUE
                      for evaluation during self-play. Otherwise uses heuristics.
            tb_dir: Directory of solved tablebase phases. If provided, the
                    engine probes them in search and solved positions get
                    exact win/loss/draw labels.
        """
        from tonnesjakk import Board, Engine
        self.Board = Board
        self.Engine = Engine
        # Single reusable engine - avoids memory issues with multiple engine instances
        self._engine = Engine()
        self._using_nnue = False
        self._nnue_path = nnue_path
        self._tb_dir = tb_dir
        self._using_tb = False

        if nnue_path is not None:
            try:
                self._engine.load_nnue(nnue_path)
                self._using_nnue = True
                print(f"  Loaded NNUE weights from: {nnue_path}")
            except Exception as e:
                print(f"  Warning: Failed to load NNUE ({e}), using heuristics")
        if tb_dir is not None:
            try:
                phases = self._engine.load_tablebases(tb_dir)
                self._using_tb = len(phases) > 0
                print(f"  Loaded tablebases from {tb_dir}: {phases}")
            except Exception as e:
                print(f"  Warning: Failed to load tablebases ({e})")

    def play_game(
        self,
        depth: int = 6,
        random_opening_moves: int = 4,
        max_moves: int = 100,
        noise_prob: float = 0.0,
    ) -> GameResult:
        """
        Play a single self-play game.

        Args:
            depth: Search depth for the engine
            random_opening_moves: Number of random moves at start (2-6 recommended)
            max_moves: Maximum moves before declaring draw
            noise_prob: Probability of playing a random barrel move instead of
                the engine's choice during the first 20 plies. Positions are
                still labeled with the engine's search score; only the game
                trajectory is diversified (a deterministic engine funnels
                games into the same lines: gen-1 was 74% duplicate positions).

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
            # Barrel moves only: the pail is a once-per-game strategic decision
            # left to the engine, not burned on a random opening square.
            moves = [m for m in board.generate_moves() if not m.is_pail_only]
            if not moves or board.check_winner():
                break
            board.make_move(random.choice(moves))

        # Draw-rule tracking: `recent` = hashes since last irreversible event
        # (passed to the engine so search sees repetitions), `counts` for 3-fold.
        recent = [board.get_hash()]
        counts = {board.get_hash(): 1}
        NO_PROGRESS_LIMIT = 60

        # Engine plays the rest
        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            # Real draw rules (threefold repetition / no-progress clock)
            if board.halfmove_clock >= NO_PROGRESS_LIMIT or counts.get(board.get_hash(), 0) >= 3:
                break

            engine.set_game_history(recent)
            result = engine.search(board, depth)
            if result.best_move is None:
                break

            # Save position with search score
            # Quiet position filtering (Stockfish-inspired):
            #   - Skip first 4 moves (too influenced by random opening)
            #   - Skip clearly decided positions (|score| > 3000)
            if move_count >= 4:
                raw_score = result.score

                # Tablebase-solved position: exact label (+1 white wins, -1 black
                # wins, 0 draw). These must NOT be dropped by the "decided" filter
                # below — they are exactly the endgame knowledge the net should learn.
                tb = self._engine.tablebase_probe(board) if self._using_tb else None
                if tb is not None:
                    is_white = "White" in repr(board.current_player)
                    positions.append(PositionData(
                        board=board.to_array(),
                        search_score={"white": 1.0, "black": -1.0, "draw": 0.0}[tb[0]],
                        white_scored=board.white_scored,
                        black_scored=board.black_scored,
                        current_player=1 if is_white else -1
                    ))
                # Skip noisy/decided positions
                elif abs(raw_score) <= 3000:
                    is_white = "White" in repr(board.current_player)
                    current_player = 1 if is_white else -1

                    # Sigmoid normalization: tanh(score / SCALING)
                    # Unlike linear clip, this preserves information for all scores.
                    # tanh maps (-inf, +inf) to (-1, +1) smoothly.
                    normalized_score = math.tanh(raw_score / SCORE_SCALING)

                    # NOTE: the engine's search score is ALREADY from White's
                    # perspective (white-maximizing minimax), regardless of side
                    # to move. An earlier version negated it for black-to-move
                    # positions here, which inverted half of all labels and made
                    # them uncorrelated with game outcome (r = -0.002); fixed
                    # 2026-08-25 (r = +0.75 after repair). Do not "flip" it.

                    positions.append(PositionData(
                        board=board.to_array(),
                        search_score=normalized_score,
                        white_scored=board.white_scored,
                        black_scored=board.black_scored,
                        current_player=current_player
                    ))

            chosen = result.best_move
            if noise_prob > 0 and move_count < 20 and random.random() < noise_prob:
                alternatives = [m for m in board.generate_moves() if not m.is_pail_only]
                if alternatives:
                    chosen = random.choice(alternatives)
            board.make_move(chosen)
            move_count += 1

            h = board.get_hash()
            if board.halfmove_clock == 0:
                recent = [h]
            else:
                recent.append(h)
            counts[h] = counts.get(h, 0) + 1

        # Determine outcome
        winner = board.check_winner()
        if winner is None:
            outcome = 0.0  # Draw (repetition/no-progress) or max moves reached
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
        lambda_blend: Optional[float] = None,
        noise_prob: float = 0.0,
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
        tb_dir = self._tb_dir if self._using_tb else None

        start_time = time.time()

        if workers > 1:
            self._generate_parallel(
                num_games=num_games, depth=depth,
                random_opening_moves=random_opening_moves,
                use_search_scores=use_search_scores, augment=augment,
                verbose=verbose, save_every=save_every, save_path=save_path,
                config=config, workers=workers, nnue_path=nnue_path,
                chunks_X=chunks_X, chunks_y=chunks_y, stats=stats,
                start_time=start_time, lambda_blend=lambda_blend,
                noise_prob=noise_prob, tb_dir=tb_dir,
            )
        else:
            self._generate_sequential(
                num_games=num_games, depth=depth,
                random_opening_moves=random_opening_moves,
                use_search_scores=use_search_scores, augment=augment,
                verbose=verbose, save_every=save_every, save_path=save_path,
                config=config, chunks_X=chunks_X, chunks_y=chunks_y,
                stats=stats, start_time=start_time, lambda_blend=lambda_blend,
                noise_prob=noise_prob,
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
        chunks_X, chunks_y, stats, start_time, lambda_blend=None, noise_prob=0.0
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
            result = self.play_game(depth, random_opening_moves, noise_prob=noise_prob)

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
        chunks_X, chunks_y, stats, start_time, lambda_blend=None, noise_prob=0.0,
        tb_dir=None
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
                        lambda_blend, noise_prob, tb_dir
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
    num_games, depth, random_moves, use_search_scores, augment, nnue_path, lambda_blend, noise_prob, tb_dir = args

    gen = DataGenerator(nnue_path=nnue_path, tb_dir=tb_dir)

    all_X = []
    all_y = []  # Each entry is [search_score, outcome] — 2 floats per position
    white_wins = black_wins = draws = 0

    for _ in range(num_games):
        result = gen.play_game(depth, random_moves, noise_prob=noise_prob)

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
    save_data: Optional[str] = None,
    load_data: Optional[str] = None,
    save_every: int = 0,
    generate_only: bool = False,
    workers: int = 1,
    batch_size: int = 4096,
    lambda_blend: Optional[float] = None,
    loss_fn: str = "wdl-ce",
    learning_rate: float = 0.001,
    resume_from: Optional[str] = None,
    feature_set: str = "halfpail",
    mirror_black: bool = False,
    dense_size: int = 20,
    output_buckets: int = 1,
    dedupe: bool = False,
    noise_prob: float = 0.0,
    data_fraction: float = 1.0,
    tb_dir: Optional[str] = None,
) -> Optional[nn.Module]:
    """
    Complete NNUE training pipeline (data generation, training, export).

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
    arch = NnueArch(feature_set=feature_set, mirror_black=mirror_black, dense_size=dense_size,
                    hidden1=hidden1, hidden2=hidden2, output_buckets=output_buckets)
    print("=" * 60)
    print("NNUE TRAINING FOR TONNESJAKK")
    print("=" * 60)
    print(f"\nSettings:")
    print(f"  Architecture: {arch.tag()}  (EmbeddingBag({arch.num_features}, {hidden1}) -> "
          f"FC2({arch.fc2_input}, {hidden2}) x{output_buckets} -> 1)")
    if resume_from:
        print(f"  Resuming from: {resume_from}")
    print(f"  Loss: {loss_fn} ({'WDL cross-entropy' if loss_fn == 'wdl-ce' else 'mean squared error'})")

    # Step 1: Generate or load data
    if load_data:
        print(f"\n[1/3] Loading training data from {load_data}...")
        if "," in load_data:
            # Several streaming datasets: a lazy row-concatenation over the memmaps.
            # (An eager np.concatenate of gen-1..3 is 25 GB and swaps the machine.)
            from .nnue_arch import ConcatRows
            parts = [DataGenerator.load_streaming_dataset(p.strip()) for p in load_data.split(",")]
            X = ConcatRows([p[0] for p in parts])
            y = np.concatenate([np.asarray(p[1]) for p in parts])  # labels are small
            stats = TrainingStats()
            for p in parts:
                stats.white_wins += p[2].white_wins
                stats.black_wins += p[2].black_wins
                stats.draws += p[2].draws
            stats.total_positions = len(X)
            loaded_config = parts[0][3]
            print(f"  Loaded {len(X):,} positions from {len(parts)} files")
        elif load_data.endswith('.bin'):
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
        generator = DataGenerator(nnue_path=use_nnue, tb_dir=tb_dir)
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
            "noise_prob": noise_prob,
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
            lambda_blend=lambda_blend,
            noise_prob=noise_prob,
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

    # Data-size curve support: train on a random subset of the rows
    if data_fraction < 1.0:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(X), int(len(X) * data_fraction), replace=False))
        from .nnue_arch import RowView
        X, y = RowView(X, keep), np.asarray(y)[keep]
        print(f"  Data fraction {data_fraction:.2f}: using {len(X):,} rows")

    # Check balance
    if stats.balance_ratio < 0.5 or stats.balance_ratio > 2.0:
        print(f"\n  WARNING: Dataset is unbalanced (ratio: {stats.balance_ratio:.2f})")
        print("  Consider adjusting search depth or random moves")

    # Step 2: Train
    print(f"\n[2/3] Training model...")
    model, history = train_sparse_model(
        X, y, arch,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        resume_from=resume_from,
        lambda_blend=lambda_blend,
        dedupe=dedupe,
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
    export_sparse_json(model, str(json_path))
    print(f"  JSON weights: {json_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Measure strength with the match harness (equal time is the honest test):")
    print(f"  python scripts/match.py --time-a 100 --time-b 100 --nnue-a {json_path} --games 400 --sprt 0 10")

    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data and train NNUE evaluators for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training data (checkpointed, resumable)
  python -m tonnesjakk.nnue --generate-only --games 150000 --depth 8 --workers 4 --save-data data.bin
  # Train an architecture on existing data
  python -m tonnesjakk.nnue --load-data data.bin --feature-set plain --mirror --output-buckets 25 \\
      --arch 256 32 --epochs 50 --lambda 0.8 --output runs/plain_m_256
  # Self-improvement loop: generate with an NNUE engine
  python -m tonnesjakk.nnue --generate-only --use-nnue runs/best/nnue_weights.json --save-data gen2.bin
  # Measure strength (equal time): scripts/match.py --nnue-a runs/x/nnue_weights.json --time-a 100 --time-b 100
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
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N games during generation (default: 500)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data (no training). Use with --save-data")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    parser.add_argument("--lambda", type=float, default=None, dest="lambda_blend",
                        help="Lambda blend: mix search scores and game outcomes (0.85 = 85%% eval + 15%% outcome)")
    parser.add_argument("--feature-set", type=str, default="halfpail", choices=["halfpail", "plain"],
                        help="Sparse feature set: halfpail (pail-square buckets) or plain (default: halfpail)")
    parser.add_argument("--mirror", action="store_true",
                        help="Mirror the board for the black perspective (orientation-consistent shared weights)")
    parser.add_argument("--no-dense", action="store_true",
                        help="Drop the 20 dense relational features")
    parser.add_argument("--output-buckets", type=int, default=1, choices=[1, 25],
                        help="Output heads: 1, or 25 keyed on (white_scored, black_scored)")
    parser.add_argument("--dedupe", action="store_true",
                        help="Collapse duplicate positions before training, averaging their labels")
    parser.add_argument("--noise-prob", type=float, default=0.0,
                        help="Data generation: probability of a random move (first 20 plies) for trajectory diversity")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Train on a random fraction of the loaded rows (data-size curves)")
    parser.add_argument("--tb", type=str, default=None,
                        help="Data generation: tablebase directory (engine probes it; solved positions get exact labels)")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from a saved .pt model file")

    args = parser.parse_args()

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
        save_data=args.save_data,
        load_data=args.load_data,
        save_every=args.save_every,
        generate_only=args.generate_only,
        workers=args.workers,
        batch_size=args.batch_size,
        lambda_blend=args.lambda_blend,
        loss_fn=args.loss,
        learning_rate=args.lr,
        resume_from=args.resume_from,
        feature_set=args.feature_set,
        mirror_black=args.mirror,
        dense_size=0 if args.no_dense else 20,
        output_buckets=args.output_buckets,
        dedupe=args.dedupe,
        noise_prob=args.noise_prob,
        data_fraction=args.data_fraction,
        tb_dir=args.tb,
    )


if __name__ == "__main__":
    main()
