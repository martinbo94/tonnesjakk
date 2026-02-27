"""Watch the AlphaZero network play a game against itself or heuristic.

Usage:
  python scripts/watch_game.py alphazero_run3/latest_model.pt
  python scripts/watch_game.py alphazero_run3/latest_model.pt --simulations 400
  python scripts/watch_game.py alphazero_run3/latest_model.pt --vs-heuristic 5
"""

import argparse

import numpy as np
import torch

from tonnesjakk import Board
from tonnesjakk._core import MCTSEngine as _RustMCTSEngine
from tonnesjakk.alphazero import (
    AlphaZeroTrainer,
    NetworkMCTS,
    POLICY_SIZE,
)


def _fmt_move(m):
    """Format a move for display."""
    if m.is_barrel_placement:
        to = m.barrel_to
        return f"place->({to.row},{to.col})"
    frm = m.barrel_from
    to = m.barrel_to
    return f"({frm.row},{frm.col})->({to.row},{to.col})"


def _net_eval(net, board, engine, device):
    """Get raw network policy and value for a position (no MCTS)."""
    planes = engine.board_planes(board)
    planes_t = torch.tensor(planes, dtype=torch.float32).reshape(1, 6, 6, 6).to(device)
    with torch.no_grad():
        logits, value = net(planes_t)
    probs = logits.softmax(1)[0].cpu()
    return probs, value.item()


def _mcts_search(engine, board, eval_fn, batch_size):
    """Run MCTS search, return (policy, root_value, best_move)."""
    result = engine.search_network_batched(board, eval_fn, batch_size=batch_size)
    return np.array(result.policy_target), result.root_value, result.best_move


def play_self_game(trainer, simulations, max_moves=80):
    """Play a self-play game using MCTS + network, printing each move."""
    trainer.network.eval()
    mcts = NetworkMCTS(
        trainer.network,
        simulations=simulations,
        c_puct=1.4,
        batch_size=trainer.mcts_batch_size,
        device=trainer.device,
        use_amp=trainer.use_amp,
    )
    engine = _RustMCTSEngine(simulations, 1.4)

    board = Board()
    print(board.display())
    print()

    for move_num in range(max_moves):
        winner = board.check_winner()
        if winner is not None:
            break

        moves = board.generate_moves()
        if not moves:
            break

        # Raw network eval
        net_probs, raw_value = _net_eval(trainer.network, board, engine, trainer.device)

        # MCTS search
        mcts_policy, mcts_value, best_move = _mcts_search(
            engine, board, mcts._batch_eval_fn, mcts.batch_size
        )

        # Collect move probabilities
        move_probs = []
        for m in moves:
            idx = m.policy_index()
            mp = mcts_policy[idx] if idx < len(mcts_policy) else 0
            np_ = net_probs[idx].item() if idx < len(net_probs) else 0
            move_probs.append((m, mp, np_))
        move_probs.sort(key=lambda x: -x[1])

        player = "White" if str(board.current_player) == "Player.White" else "Black"
        print(f"Move {move_num+1} ({player}):")
        print(f"  Net value: {raw_value:+.4f}  |  MCTS value: {mcts_value:+.4f}")
        print(f"  Top 5 moves (mcts% / net%):")
        for m, mp, np_ in move_probs[:5]:
            marker = ">>" if _fmt_move(m) == _fmt_move(best_move) else "  "
            print(f"    {marker} {_fmt_move(m):20s}  mcts={mp:.1%}  net={np_:.1%}")

        board.make_move(best_move)
        print()
        print(board.display())
        print()

    winner = board.check_winner()
    if winner is not None:
        print(f"Result: {winner} wins in {board.move_count} moves")
    else:
        print(f"Result: Draw (reached move {board.move_count})")


def play_vs_heuristic(trainer, simulations, depth, max_moves=80):
    """Play network (White) vs heuristic (Black), printing each move."""
    trainer.network.eval()
    mcts = NetworkMCTS(
        trainer.network,
        simulations=simulations,
        c_puct=1.4,
        batch_size=trainer.mcts_batch_size,
        device=trainer.device,
        use_amp=trainer.use_amp,
    )
    engine = _RustMCTSEngine(simulations, 1.4)

    board = Board()
    print(f"Network (White, {simulations} sims) vs Heuristic (Black, depth {depth})")
    print(board.display())
    print()

    for move_num in range(max_moves):
        winner = board.check_winner()
        if winner is not None:
            break

        moves = board.generate_moves()
        if not moves:
            break

        is_white = str(board.current_player) == "Player.White"

        if is_white:
            # Network + MCTS
            net_probs, raw_value = _net_eval(trainer.network, board, engine, trainer.device)
            mcts_policy, mcts_value, best_move = _mcts_search(
                engine, board, mcts._batch_eval_fn, mcts.batch_size
            )
            print(f"Move {move_num+1} (Network/White): {_fmt_move(best_move)}  "
                  f"net_val={raw_value:+.4f}  mcts_val={mcts_value:+.4f}")
        else:
            # Heuristic alpha-beta
            result = engine.search_heuristic(board)
            best_move = result.best_move
            print(f"Move {move_num+1} (Heuristic/Black): {_fmt_move(best_move)}  "
                  f"value={result.root_value:+.4f}")

        board.make_move(best_move)
        print(board.display())
        print()

    winner = board.check_winner()
    if winner is not None:
        print(f"Result: {winner} wins in {board.move_count} moves")
    else:
        print(f"Result: Draw (reached move {board.move_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch AlphaZero play a game")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")
    parser.add_argument("--vs-heuristic", type=int, default=0,
                        help="Play vs heuristic at this depth (0 = self-play)")
    parser.add_argument("--max-moves", type=int, default=80,
                        help="Max moves before draw (default: 80)")
    args = parser.parse_args()

    trainer = AlphaZeroTrainer(device="auto")
    trainer.load(args.checkpoint)

    if args.vs_heuristic > 0:
        play_vs_heuristic(trainer, args.simulations, args.vs_heuristic, args.max_moves)
    else:
        play_self_game(trainer, args.simulations, args.max_moves)
