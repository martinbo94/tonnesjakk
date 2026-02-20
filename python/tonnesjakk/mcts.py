"""
Monte Carlo Tree Search (MCTS) for Tonnesjakk.

Implements PUCT-based MCTS with pluggable leaf evaluators:
  - heuristic: Uses the engine's hand-crafted evaluation
  - rollout:   Random playout to game end
  - shallow:   Fixed-depth alpha-beta search

This is the first step toward AlphaZero-style training where the search
itself generates training signal (policy from visit counts, value from
game outcomes).

Usage:
  python -m tonnesjakk.mcts --demo --simulations 200
  python -m tonnesjakk.mcts --match --games 20 --simulations 400 --opponent-depth 5
"""

import argparse
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from tonnesjakk import Board, Engine, Player


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the MCTS search tree.

    All values are stored from White's perspective. The PUCT formula in
    ucb_score() flips the sign when the parent is Black (Black wants to
    minimize White's score).
    """

    __slots__ = (
        "parent", "move", "children", "visit_count",
        "total_value", "prior", "is_terminal", "terminal_value", "player",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        move=None,
        prior: float = 1.0,
        player_is_white: bool = True,
    ):
        self.parent = parent
        self.move = move              # Move that led to this node
        self.children: List["MCTSNode"] = []
        self.visit_count: int = 0
        self.total_value: float = 0.0  # Sum of values (White's perspective)
        self.prior: float = prior      # Policy prior (uniform for now)
        self.is_terminal: bool = False
        self.terminal_value: float = 0.0
        self.player: bool = player_is_white  # True = White to move at this node

    @property
    def q_value(self) -> float:
        """Mean value from White's perspective."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def ucb_score(self, c_puct: float = 1.4) -> float:
        """PUCT score for child selection.

        Q is from White's perspective. When the parent is Black (wanting to
        minimize White's score), we negate Q so that Black prefers children
        with lower White value.
        """
        if self.parent is None:
            return 0.0

        # Exploration bonus
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        # Q from the perspective of the parent's player
        q = self.q_value
        if not self.parent.player:
            # Parent is Black: Black wants to minimize White's score
            q = -q

        # But wait -- parent is *selecting* among children. Parent wants
        # the child with the highest score *for parent's side*.
        # If parent is White, pick highest Q (White's perspective).
        # If parent is Black, pick highest -Q (= lowest White Q).
        # The negation above already handles this.

        return q + exploration

    def best_child(self, c_puct: float = 1.4) -> "MCTSNode":
        """Select child with highest PUCT score."""
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def most_visited_child(self) -> "MCTSNode":
        """Select child with most visits (for final move selection)."""
        return max(self.children, key=lambda c: c.visit_count)


# ---------------------------------------------------------------------------
# MCTS Engine
# ---------------------------------------------------------------------------

SCORE_SCALING = 600.0  # Same as NNUE training: tanh(score/600)


class MCTS:
    """Monte Carlo Tree Search with pluggable evaluation.

    Args:
        simulations: Number of simulations per move.
        evaluator: Leaf evaluation mode ("heuristic", "rollout", "shallow").
        shallow_depth: Search depth for "shallow" evaluator.
        c_puct: Exploration constant for PUCT formula.
    """

    def __init__(
        self,
        simulations: int = 800,
        evaluator: str = "heuristic",
        shallow_depth: int = 3,
        c_puct: float = 1.4,
    ):
        self.simulations = simulations
        self.evaluator = evaluator
        self.shallow_depth = shallow_depth
        self.c_puct = c_puct
        self.engine = Engine()  # For heuristic/shallow evaluation

    def search(self, board: Board) -> Tuple[Optional[object], dict]:
        """Run MCTS from the given position.

        Returns:
            (best_move, info_dict) where info_dict contains visit counts,
            Q values, and the policy distribution.
        """
        is_white = _is_white(board)
        root = MCTSNode(player_is_white=is_white)

        # Expand root
        self._expand(root, board)

        if not root.children:
            return None, {"visits": 0}

        # If only one legal move, no need to search
        if len(root.children) == 1:
            root.children[0].visit_count = 1
            return root.children[0].move, {"visits": 1}

        # Run simulations
        for _ in range(self.simulations):
            node = root
            sim_board = board.copy()

            # 1. SELECT: walk down tree using PUCT
            while node.children and not node.is_terminal:
                node = node.best_child(self.c_puct)
                sim_board.make_move(node.move)

            # 2. EXPAND: if not terminal and not yet expanded
            if not node.is_terminal and node.visit_count > 0:
                self._expand(node, sim_board)
                if node.children:
                    node = node.children[0]  # Pick first child
                    sim_board.make_move(node.move)

            # 3. EVALUATE: get value of leaf (White's perspective)
            if node.is_terminal:
                value = node.terminal_value
            else:
                value = self._evaluate(sim_board)

            # 4. BACKPROPAGATE: update values up to root
            while node is not None:
                node.visit_count += 1
                node.total_value += value
                node = node.parent

        # Build info dict
        total_visits = sum(c.visit_count for c in root.children)
        info = {
            "visits": total_visits,
            "children": [
                {
                    "move": str(c.move),
                    "visits": c.visit_count,
                    "q": round(c.q_value, 4),
                    "prior": round(c.prior, 4),
                    "policy": c.visit_count / max(1, total_visits),
                }
                for c in sorted(root.children, key=lambda c: -c.visit_count)[:10]
            ],
        }

        best = root.most_visited_child()
        return best.move, info

    def _expand(self, node: MCTSNode, board: Board):
        """Create child nodes for all legal moves."""
        winner = board.check_winner()
        if winner is not None:
            node.is_terminal = True
            node.terminal_value = 1.0 if _is_white_winner(winner) else -1.0
            return

        moves = board.generate_moves()
        if not moves:
            node.is_terminal = True
            node.terminal_value = 0.0  # Draw / stalemate
            return

        # Uniform priors (future: use policy network here)
        prior = 1.0 / len(moves)
        child_is_white = not node.player  # Next player's turn

        for move in moves:
            child = MCTSNode(
                parent=node,
                move=move,
                prior=prior,
                player_is_white=child_is_white,
            )
            node.children.append(child)

    def _evaluate(self, board: Board) -> float:
        """Evaluate a leaf position. Returns value in [-1, +1] from White's perspective."""
        if self.evaluator == "heuristic":
            raw = self.engine.evaluate_position(board)
            return math.tanh(raw / SCORE_SCALING)

        elif self.evaluator == "rollout":
            return self._rollout(board)

        elif self.evaluator == "shallow":
            self.engine.full_reset()
            result = self.engine.search(board, self.shallow_depth)
            raw = result.score
            # search() returns score from current player's perspective,
            # but evaluate_position() returns White's perspective.
            # Adjust: if Black to move, negate.
            if not _is_white(board):
                raw = -raw
            return math.tanh(raw / SCORE_SCALING)

        else:
            raise ValueError(f"Unknown evaluator: {self.evaluator}")

    def _rollout(self, board: Board, max_moves: int = 60) -> float:
        """Random playout to game end. Returns value from White's perspective."""
        sim = board.copy()
        for _ in range(max_moves):
            winner = sim.check_winner()
            if winner is not None:
                return 1.0 if _is_white_winner(winner) else -1.0
            moves = sim.generate_moves()
            if not moves:
                return 0.0
            sim.make_move(random.choice(moves))

        # Max moves reached: use heuristic as tiebreaker
        raw = self.engine.evaluate_position(sim)
        return math.tanh(raw / SCORE_SCALING)


# ---------------------------------------------------------------------------
# Player wrapper
# ---------------------------------------------------------------------------

class MCTSPlayer:
    """Wraps MCTS for use in game-playing loops."""

    def __init__(self, mcts: MCTS, temperature: float = 1.0):
        self.mcts = mcts
        self.temperature = temperature

    def select_move(self, board: Board) -> Tuple[Optional[object], dict]:
        """Select a move using MCTS with temperature-based sampling.

        temperature=0: Always pick most visited (deterministic).
        temperature=1: Sample proportional to visit counts.
        temperature>1: More uniform / exploratory.
        """
        move, info = self.mcts.search(board)

        if move is None or self.temperature == 0.0:
            return move, info

        root_children = info.get("children", [])
        if not root_children:
            return move, info

        # Temperature-based sampling from visit counts
        # We need the actual children from the tree, but info only has summaries.
        # For temperature > 0, re-search would be wasteful. Instead, use the
        # visit counts from info to build a distribution.
        # But we need the actual move objects, not strings. So we access the
        # root's children directly. However, the root is local to search().
        # Workaround: for temperature != 0, we sample from the MCTS root.
        # Let's refactor slightly: always return the root for sampling.

        # For now, with temperature=0 or very close, use deterministic.
        # Temperature sampling will be added when we have the root reference.
        return move, info


# ---------------------------------------------------------------------------
# Game playing
# ---------------------------------------------------------------------------

@dataclass
class TrainingExample:
    """One training example from MCTS self-play."""
    board_array: list          # Board state (from board.to_array())
    policy: dict               # Move -> visit fraction
    outcome: float             # Game outcome from White's perspective


@dataclass
class GameRecord:
    """Record of a single game."""
    moves: List[str] = field(default_factory=list)
    winner: Optional[str] = None
    num_moves: int = 0
    training_examples: List[TrainingExample] = field(default_factory=list)


def play_mcts_game(
    mcts: MCTS,
    collect_training_data: bool = False,
    max_moves: int = 80,
    random_opening_moves: int = 2,
    verbose: bool = False,
) -> GameRecord:
    """Play a single game with MCTS on both sides.

    Args:
        mcts: MCTS instance for move selection.
        collect_training_data: If True, save (board, policy, outcome) triples.
        max_moves: Maximum moves before declaring draw.
        random_opening_moves: Number of random opening moves.
        verbose: Print board and move info.

    Returns:
        GameRecord with moves and optionally training data.
    """
    board = Board()
    record = GameRecord()
    examples = []

    # Random opening
    for i in range(random_opening_moves):
        moves = board.generate_moves()
        if not moves or board.check_winner() is not None:
            break
        move = random.choice(moves)
        board.make_move(move)
        record.moves.append(str(move))

    if verbose:
        print(f"After {random_opening_moves} random opening moves:")
        print(board.display())
        print()

    # Main game loop
    move_count = 0
    while board.check_winner() is None and move_count < max_moves:
        move, info = mcts.search(board)

        if move is None:
            break

        # Collect training data
        if collect_training_data:
            policy = {}
            for child_info in info.get("children", []):
                policy[child_info["move"]] = child_info["policy"]
            examples.append(TrainingExample(
                board_array=board.to_array(),
                policy=policy,
                outcome=0.0,  # Filled in after game ends
            ))

        if verbose:
            player = "White" if _is_white(board) else "Black"
            q_str = ""
            if info.get("children"):
                q_str = f" Q={info['children'][0]['q']:+.3f}"
            print(f"Move {move_count + 1} ({player}): {_safe_str(move)}  "
                  f"[{info['visits']} visits{q_str}]")

        board.make_move(move)
        record.moves.append(str(move))
        move_count += 1

        if verbose:
            print(board.display())
            print()

    # Determine outcome
    winner = board.check_winner()
    if winner is None:
        outcome = 0.0
        record.winner = "draw"
    elif _is_white_winner(winner):
        outcome = 1.0
        record.winner = "white"
    else:
        outcome = -1.0
        record.winner = "black"

    record.num_moves = len(record.moves)

    # Fill in outcomes for training examples
    if collect_training_data:
        for ex in examples:
            ex.outcome = outcome
        record.training_examples = examples

    if verbose:
        print(f"Game over: {record.winner} wins after {record.num_moves} moves")

    return record


def run_mcts_match(
    simulations: int = 800,
    evaluator: str = "heuristic",
    shallow_depth: int = 3,
    opponent_depth: int = 6,
    num_games: int = 50,
    c_puct: float = 1.4,
    max_moves: int = 80,
    verbose: bool = True,
) -> dict:
    """Run a match between MCTS and the alpha-beta engine.

    MCTS and alpha-beta alternate colors each game for fairness.

    Returns:
        Dict with wins, losses, draws, and ELO estimate.
    """
    mcts = MCTS(
        simulations=simulations,
        evaluator=evaluator,
        shallow_depth=shallow_depth,
        c_puct=c_puct,
    )
    engine = Engine()

    wins = 0
    losses = 0
    draws = 0

    t0 = time.time()

    for game_idx in range(num_games):
        board = Board()
        engine.full_reset()
        mcts_is_white = (game_idx % 2 == 0)

        # Random opening (2 moves for variety)
        for _ in range(2):
            moves = board.generate_moves()
            if not moves or board.check_winner() is not None:
                break
            board.make_move(random.choice(moves))

        move_count = 0
        while board.check_winner() is None and move_count < max_moves:
            is_white_turn = _is_white(board)
            is_mcts_turn = (is_white_turn == mcts_is_white)

            if is_mcts_turn:
                move, _ = mcts.search(board)
            else:
                engine.full_reset()
                sr = engine.search(board, opponent_depth)
                move = sr.best_move

            if move is None:
                break

            board.make_move(move)
            move_count += 1

        # Determine result
        winner = board.check_winner()
        if winner is None:
            draws += 1
            result_str = "draw"
        else:
            white_won = _is_white_winner(winner)
            if white_won == mcts_is_white:
                wins += 1
                result_str = "MCTS wins"
            else:
                losses += 1
                result_str = "AB wins"

        if verbose:
            color = "W" if mcts_is_white else "B"
            elapsed = time.time() - t0
            total = wins + losses + draws
            elo, elo_lo, elo_hi = elo_with_ci(wins, losses, draws)
            print(
                f"Game {game_idx + 1}/{num_games} "
                f"(MCTS={color}): {result_str:10s} | "
                f"MCTS {wins}W-{draws}D-{losses}L | "
                f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}] | "
                f"{elapsed:.1f}s"
            )

    total = wins + losses + draws
    elo, elo_lo, elo_hi = elo_with_ci(wins, losses, draws)

    result = {
        "mcts_wins": wins,
        "ab_wins": losses,
        "draws": draws,
        "total": total,
        "elo": round(elo),
        "elo_lo": round(elo_lo),
        "elo_hi": round(elo_hi),
        "simulations": simulations,
        "evaluator": evaluator,
        "opponent_depth": opponent_depth,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"MCTS ({simulations} sims, {evaluator}) vs AB (depth {opponent_depth})")
        print(f"Result: {wins}W-{draws}D-{losses}L")
        print(f"ELO: {elo:+.0f} [{elo_lo:+.0f}, {elo_hi:+.0f}]")
        print(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _safe_str(obj) -> str:
    """Convert to string, replacing Unicode chars that Windows cp1252 can't handle."""
    return str(obj).encode("ascii", errors="replace").decode("ascii")


def _is_white(board: Board) -> bool:
    """Check if it's White's turn."""
    return "White" in repr(board.current_player)


def _is_white_winner(winner) -> bool:
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCTS for Tonnesjakk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python -m tonnesjakk.mcts --demo --simulations 200
  python -m tonnesjakk.mcts --match --games 20 --simulations 400 --opponent-depth 5
  python -m tonnesjakk.mcts --match --simulations 800 --evaluator shallow --shallow-depth 3
        """,
    )
    parser.add_argument("--match", action="store_true",
                        help="Run MCTS vs alpha-beta match")
    parser.add_argument("--demo", action="store_true",
                        help="Play one demo game with ASCII board output")
    parser.add_argument("--self-play", action="store_true",
                        help="Run MCTS self-play and collect training data")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS simulations per move (default: 800)")
    parser.add_argument("--evaluator", type=str, default="heuristic",
                        choices=["heuristic", "rollout", "shallow"],
                        help="Leaf evaluation method (default: heuristic)")
    parser.add_argument("--shallow-depth", type=int, default=3,
                        help="Search depth for shallow evaluator (default: 3)")
    parser.add_argument("--games", type=int, default=50,
                        help="Number of games (default: 50)")
    parser.add_argument("--opponent-depth", type=int, default=6,
                        help="Alpha-beta opponent search depth (default: 6)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Move selection temperature (default: 0 = deterministic)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                        help="PUCT exploration constant (default: 1.4)")
    parser.add_argument("--max-moves", type=int, default=80,
                        help="Max moves per game (default: 80)")

    args = parser.parse_args()

    if args.demo:
        print(f"MCTS Demo: {args.simulations} simulations, {args.evaluator} evaluator")
        print("=" * 60)
        mcts = MCTS(
            simulations=args.simulations,
            evaluator=args.evaluator,
            shallow_depth=args.shallow_depth,
            c_puct=args.c_puct,
        )
        record = play_mcts_game(
            mcts,
            verbose=True,
            max_moves=args.max_moves,
            random_opening_moves=2,
        )
        print(f"\nGame length: {record.num_moves} moves")

    elif args.match:
        run_mcts_match(
            simulations=args.simulations,
            evaluator=args.evaluator,
            shallow_depth=args.shallow_depth,
            opponent_depth=args.opponent_depth,
            num_games=args.games,
            c_puct=args.c_puct,
            max_moves=args.max_moves,
        )

    elif args.self_play:
        print(f"Self-play: {args.games} games, {args.simulations} sims, "
              f"{args.evaluator} evaluator")
        print("=" * 60)
        mcts = MCTS(
            simulations=args.simulations,
            evaluator=args.evaluator,
            shallow_depth=args.shallow_depth,
            c_puct=args.c_puct,
        )
        all_examples = []
        results = {"white": 0, "black": 0, "draw": 0}

        for i in range(args.games):
            record = play_mcts_game(
                mcts,
                collect_training_data=True,
                max_moves=args.max_moves,
            )
            results[record.winner] += 1
            all_examples.extend(record.training_examples)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"Game {i + 1}/{args.games}: "
                    f"W={results['white']} B={results['black']} D={results['draw']} | "
                    f"{len(all_examples)} training examples"
                )

        print(f"\nSelf-play complete: {len(all_examples)} training examples "
              f"from {args.games} games")
        print(f"Results: W={results['white']} B={results['black']} D={results['draw']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
