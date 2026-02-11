#!/usr/bin/env python3
"""
Standardized NNUE model testing framework for Tonnesjakk.

Runs time-controlled matches, captures per-move search stats (depth, nodes, NPS),
computes ELO differences with 95% confidence intervals, and saves results to a
history file for tracking model progress over time.

Usage:
    python scripts/test_model.py models_norel/nnue_weights.json
    python scripts/test_model.py model.json --reference old_model.json
    python scripts/test_model.py model.json --games 100 --time-ms 100
    python scripts/test_model.py --history
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tonnesjakk import Board, Engine

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MoveStats:
    depth: int
    nodes: int
    qnodes: int


@dataclass
class GameResult:
    winner: str  # "model", "opponent", "draw"
    moves: int
    model_move_stats: List[MoveStats] = field(default_factory=list)
    opponent_move_stats: List[MoveStats] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def model_avg_depth(self) -> float:
        if not self.model_move_stats:
            return 0.0
        return sum(s.depth for s in self.model_move_stats) / len(self.model_move_stats)

    @property
    def opponent_avg_depth(self) -> float:
        if not self.opponent_move_stats:
            return 0.0
        return sum(s.depth for s in self.opponent_move_stats) / len(self.opponent_move_stats)

    @property
    def model_total_nodes(self) -> int:
        return sum(s.nodes for s in self.model_move_stats)

    @property
    def opponent_total_nodes(self) -> int:
        return sum(s.nodes for s in self.opponent_move_stats)


@dataclass
class MatchupResult:
    model_name: str
    opponent_name: str
    settings: Dict
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: List[GameResult] = field(default_factory=list)
    wall_time: float = 0.0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        if self.total == 0:
            return 0.5
        return (self.wins + 0.5 * self.draws) / self.total

    @property
    def elo_diff(self) -> float:
        wr = self.win_rate
        if wr <= 0.01:
            return -400.0
        if wr >= 0.99:
            return 400.0
        return 400.0 * math.log10(wr / (1.0 - wr))


# ---------------------------------------------------------------------------
# ELO confidence interval (Wilson score interval)
# ---------------------------------------------------------------------------

def elo_with_ci(wins: int, losses: int, draws: int) -> Tuple[float, float, float]:
    """Return (elo_diff, elo_lo, elo_hi) using Wilson 95% CI on win rate."""
    n = wins + losses + draws
    if n == 0:
        return 0.0, -400.0, 400.0

    # Treat draws as half-wins for a binomial model
    successes = wins + 0.5 * draws
    p_hat = successes / n

    # Wilson score interval (z = 1.96 for 95%)
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


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def make_engine(weights_path: Optional[str]) -> Engine:
    """Create an engine, loading NNUE weights if a path is provided."""
    engine = Engine()
    if weights_path is not None:
        engine.load_nnue(weights_path)
    return engine


def is_white_turn(board: Board) -> bool:
    return "White" in repr(board.current_player)


# ---------------------------------------------------------------------------
# Core match runner
# ---------------------------------------------------------------------------

def run_match(
    model_path: Optional[str],
    opponent_path: Optional[str],
    num_games: int,
    time_ms: int,
    max_moves: int = 80,
    game_timeout: float = 120.0,
    verbose: bool = True,
) -> MatchupResult:
    """
    Play a match between two engines with full per-move stat tracking.

    Args:
        model_path: Path to NNUE weights for the model under test (None = heuristic).
        opponent_path: Path to NNUE weights for the opponent (None = heuristic).
        num_games: Number of games to play (colors alternate).
        time_ms: Milliseconds per move for search_timed.
        max_moves: Maximum moves per game before declaring a draw.
        game_timeout: Maximum wall-clock seconds per game.
        verbose: Print per-game progress.

    Returns:
        MatchupResult with per-game data and aggregates.
    """
    model_name = model_path if model_path else "heuristic"
    opponent_name = opponent_path if opponent_path else "heuristic"

    result = MatchupResult(
        model_name=model_name,
        opponent_name=opponent_name,
        settings={"games": num_games, "time_ms": time_ms},
    )

    engine_model = make_engine(model_path)
    engine_opponent = make_engine(opponent_path)

    match_start = time.time()

    for game_idx in range(num_games):
        # Alternate colors: model is white on even games
        model_is_white = (game_idx % 2 == 0)

        board = Board()
        engine_model.full_reset()
        engine_opponent.full_reset()

        # Random opening (2–4 random moves)
        opening_moves = random.randint(2, 4)
        for _ in range(opening_moves):
            moves = board.generate_moves()
            if not moves or board.check_winner() is not None:
                break
            board.make_move(random.choice(moves))

        # Play game, collecting per-move stats
        game_result = GameResult(winner="draw", moves=0)
        game_start = time.time()
        move_count = 0

        while board.check_winner() is None and move_count < max_moves:
            if time.time() - game_start > game_timeout:
                break

            white_turn = is_white_turn(board)
            is_model_turn = (white_turn == model_is_white)
            current_engine = engine_model if is_model_turn else engine_opponent

            sr = current_engine.search_timed(board, time_ms)
            if sr.best_move is None:
                break

            stats = MoveStats(
                depth=sr.depth,
                nodes=sr.nodes_searched,
                qnodes=sr.quiesce_nodes,
            )
            if is_model_turn:
                game_result.model_move_stats.append(stats)
            else:
                game_result.opponent_move_stats.append(stats)

            board.make_move(sr.best_move)
            move_count += 1

        game_result.moves = move_count
        game_result.wall_time = time.time() - game_start

        # Determine winner
        winner = board.check_winner()
        if winner is None:
            game_result.winner = "draw"
            result.draws += 1
        else:
            white_won = "White" in repr(winner)
            if white_won == model_is_white:
                game_result.winner = "model"
                result.wins += 1
            else:
                game_result.winner = "opponent"
                result.losses += 1

        result.games.append(game_result)

        if verbose:
            tag = {"model": "W", "opponent": "L", "draw": "D"}[game_result.winner]
            md = game_result.model_avg_depth
            od = game_result.opponent_avg_depth
            print(
                f"  Game {game_idx + 1:>3}/{num_games}: {tag}  "
                f"({move_count:>2} moves, {game_result.wall_time:.1f}s)  "
                f"depth: {md:.1f} vs {od:.1f}  "
                f"| W={result.wins} L={result.losses} D={result.draws}",
                flush=True,
            )

    result.wall_time = time.time() - match_start
    return result


# ---------------------------------------------------------------------------
# Aggregate engine stats from a MatchupResult
# ---------------------------------------------------------------------------

def engine_stats(matchup: MatchupResult) -> Dict:
    """Compute aggregate engine statistics for both sides."""
    model_depths: List[float] = []
    model_nodes: List[int] = []
    model_time: float = 0.0
    opp_depths: List[float] = []
    opp_nodes: List[int] = []

    for g in matchup.games:
        if g.model_move_stats:
            model_depths.append(g.model_avg_depth)
            model_nodes.append(g.model_total_nodes)
        if g.opponent_move_stats:
            opp_depths.append(g.opponent_avg_depth)
            opp_nodes.append(g.opponent_total_nodes)
        model_time += g.wall_time

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    total_model_nodes = sum(model_nodes)
    total_opp_nodes = sum(opp_nodes)
    total_model_moves = sum(len(g.model_move_stats) for g in matchup.games)
    total_opp_moves = sum(len(g.opponent_move_stats) for g in matchup.games)

    return {
        "model": {
            "avg_depth": round(avg(model_depths), 2),
            "avg_nodes_per_move": round(total_model_nodes / max(total_model_moves, 1)),
            "total_nodes": total_model_nodes,
            "avg_nps": round(total_model_nodes / max(matchup.wall_time / 2, 0.001)),
        },
        "opponent": {
            "avg_depth": round(avg(opp_depths), 2),
            "avg_nodes_per_move": round(total_opp_nodes / max(total_opp_moves, 1)),
            "total_nodes": total_opp_nodes,
            "avg_nps": round(total_opp_nodes / max(matchup.wall_time / 2, 0.001)),
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(matchup: MatchupResult) -> None:
    """Print a formatted summary of a matchup."""
    elo, elo_lo, elo_hi = elo_with_ci(matchup.wins, matchup.losses, matchup.draws)
    stats = engine_stats(matchup)

    m = stats["model"]
    o = stats["opponent"]

    print()
    print("=" * 64)
    print(f"  Model:    {matchup.model_name}")
    print(f"  Opponent: {matchup.opponent_name}")
    print(f"  Settings: {matchup.settings['games']} games, {matchup.settings['time_ms']}ms/move")
    print("-" * 64)
    print(f"  Result:   +{matchup.wins} ={matchup.draws} -{matchup.losses}")
    print(f"  Win rate: {matchup.win_rate:.1%}")
    print(f"  ELO diff: {elo:+.0f}  (95% CI: {elo_lo:+.0f} to {elo_hi:+.0f})")
    print("-" * 64)
    print(f"  {'':18s} {'Model':>12s} {'Opponent':>12s}")
    print(f"  {'Avg depth':18s} {m['avg_depth']:12.1f} {o['avg_depth']:12.1f}")
    print(f"  {'Avg nodes/move':18s} {m['avg_nodes_per_move']:12,d} {o['avg_nodes_per_move']:12,d}")
    print(f"  {'Avg NPS':18s} {m['avg_nps']:12,d} {o['avg_nps']:12,d}")
    print("-" * 64)
    print(f"  Wall time: {matchup.wall_time:.1f}s ({matchup.wall_time / max(matchup.total, 1):.2f}s/game)")
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# History file management
# ---------------------------------------------------------------------------

HISTORY_PATH = Path(__file__).resolve().parent / "model_test_history.json"


def load_history(path: Path = HISTORY_PATH) -> List[Dict]:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_history(matchup: MatchupResult, path: Path = HISTORY_PATH) -> None:
    """Append a matchup result to the history file."""
    history = load_history(path)
    elo, elo_lo, elo_hi = elo_with_ci(matchup.wins, matchup.losses, matchup.draws)
    stats = engine_stats(matchup)

    # Build compact per-game records (no per-move arrays — too large)
    game_records = []
    for g in matchup.games:
        game_records.append({
            "winner": g.winner,
            "moves": g.moves,
            "model_avg_depth": round(g.model_avg_depth, 1),
            "opponent_avg_depth": round(g.opponent_avg_depth, 1),
            "model_nodes": g.model_total_nodes,
            "opponent_nodes": g.opponent_total_nodes,
            "wall_time": round(g.wall_time, 2),
        })

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": matchup.model_name,
        "opponent": matchup.opponent_name,
        "settings": matchup.settings,
        "results": {
            "wins": matchup.wins,
            "losses": matchup.losses,
            "draws": matchup.draws,
            "win_rate": round(matchup.win_rate, 4),
            "elo_diff": round(elo),
            "elo_ci_lo": round(elo_lo),
            "elo_ci_hi": round(elo_hi),
        },
        "engine_stats": stats,
        "games": game_records,
    }

    history.append(entry)

    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"  Saved to {path}")


def print_history(path: Path = HISTORY_PATH) -> None:
    """Print a tabular summary of all past matchups."""
    history = load_history(path)
    if not history:
        print("No test history found.")
        return

    print()
    print(f"{'Date':19s}  {'Model':30s}  {'Opponent':15s}  {'Games':>5s}  "
          f"{'W/L/D':>11s}  {'WR%':>5s}  {'ELO':>7s}  {'95% CI':>15s}  "
          f"{'M.Depth':>7s}  {'O.Depth':>7s}")
    print("-" * 145)

    for h in history:
        r = h["results"]
        es = h.get("engine_stats", {})
        m_depth = es.get("model", {}).get("avg_depth", 0)
        o_depth = es.get("opponent", {}).get("avg_depth", 0)
        wld = f"+{r['wins']} ={r['draws']} -{r['losses']}"
        ci = f"{r.get('elo_ci_lo', '?'):+.0f} to {r.get('elo_ci_hi', '?'):+.0f}"
        print(
            f"{h['timestamp']:19s}  {h['model']:30s}  {h['opponent']:15s}  "
            f"{h['settings']['games']:5d}  {wld:>11s}  {r['win_rate']:5.1%}  "
            f"{r['elo_diff']:+7.0f}  {ci:>15s}  {m_depth:7.1f}  {o_depth:7.1f}"
        )

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Standardized NNUE model testing for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/test_model.py nnue_weights.json                  # vs heuristic, 500 games
  python scripts/test_model.py model.json --reference old.json    # also vs reference model
  python scripts/test_model.py model.json --games 100 --time-ms 100
  python scripts/test_model.py --history                          # show all past results
""",
    )
    parser.add_argument("model", nargs="?", help="Path to NNUE weights JSON to test")
    parser.add_argument("--reference", help="Path to reference NNUE weights for head-to-head")
    parser.add_argument("--games", type=int, default=500, help="Number of games per matchup (default: 500)")
    parser.add_argument("--time-ms", type=int, default=50, help="Milliseconds per move (default: 50)")
    parser.add_argument("--max-moves", type=int, default=80, help="Max moves per game (default: 80)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to history")
    parser.add_argument("--history", action="store_true", help="Show past test results and exit")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-game output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    # -- History mode --
    if args.history:
        print_history()
        return

    if args.model is None:
        parser.error("model path is required (or use --history)")

    # Validate model file
    if not Path(args.model).exists():
        print(f"Error: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    if args.reference and not Path(args.reference).exists():
        print(f"Error: reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    verbose = not args.quiet

    # -- Matchup 1: model vs heuristic --
    print(f"\n{'='*64}")
    print(f"  MATCHUP 1: {args.model} vs heuristic")
    print(f"  {args.games} games, {args.time_ms}ms/move")
    print(f"{'='*64}\n")

    matchup_heur = run_match(
        model_path=args.model,
        opponent_path=None,
        num_games=args.games,
        time_ms=args.time_ms,
        max_moves=args.max_moves,
        verbose=verbose,
    )
    print_results(matchup_heur)
    if not args.no_save:
        save_history(matchup_heur)

    # -- Matchup 2: model vs reference (optional) --
    if args.reference:
        print(f"\n{'='*64}")
        print(f"  MATCHUP 2: {args.model} vs {args.reference}")
        print(f"  {args.games} games, {args.time_ms}ms/move")
        print(f"{'='*64}\n")

        matchup_ref = run_match(
            model_path=args.model,
            opponent_path=args.reference,
            num_games=args.games,
            time_ms=args.time_ms,
            max_moves=args.max_moves,
            verbose=verbose,
        )
        print_results(matchup_ref)
        if not args.no_save:
            save_history(matchup_ref)


if __name__ == "__main__":
    main()
