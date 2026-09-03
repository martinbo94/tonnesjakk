"""
AlphaZero training in small chunks for easy cancellation.

Runs a few iterations at a time, saving model + replay buffer after each chunk.
Ctrl+C between chunks loses at most one chunk of work.

Key features:
  - ResNet CNN architecture (default) for spatial pattern learning
  - Heuristic game seeding for value head bootstrapping
  - Decaying heuristic ratio: gradually shifts from imitation to self-play
  - Replay buffer persists across chunks via .buffer.npz sidecar

Usage:
  python scripts/train_alphazero.py
  python scripts/train_alphazero.py --network resnet --chunks 60
  python scripts/train_alphazero.py --network mlp --save-dir az_mlp
"""

import argparse
import time
from pathlib import Path

from tonnesjakk.alphazero import AlphaZeroTrainer, TrainingLogger


def main():
    parser = argparse.ArgumentParser(description="AlphaZero training in resumable chunks")
    # -- Outer loop: maximize games, not sims (Wang et al. 2020) --
    parser.add_argument("--chunks", type=int, default=20,
                        help="Number of chunks to run (default: 20)")
    parser.add_argument("--iters-per-chunk", type=int, default=3,
                        help="Iterations per chunk (default: 3)")
    parser.add_argument("--games-per-iter", type=int, default=100,
                        help="Self-play games per iteration (default: 100)")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move (default: 200)")

    # -- Network architecture --
    parser.add_argument("--hidden", type=int, default=64,
                        help="Network channels/hidden size (default: 64)")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "mlp"],
                        help="Network architecture (default: resnet)")
    parser.add_argument("--num-blocks", type=int, default=5,
                        help="Residual blocks for resnet (default: 5)")

    # -- Training: keep inner-loop params low (Wang et al. 2020) --
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--training-epochs", type=int, default=1,
                        help="Training epochs per iteration (default: 1)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Training batch size (default: 512)")
    parser.add_argument("--train-window", type=int, default=100000,
                        help="Max examples to train on per iteration (default: 100000)")
    parser.add_argument("--buffer-max", type=int, default=200000,
                        help="Max replay buffer size (default: 200000)")
    parser.add_argument("--policy-weight", type=float, default=0.5,
                        help="Policy loss weight; value-heavy improves small games "
                             "(Wang & Emmerich 2019) (default: 0.5)")

    # -- Self-play --
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Self-play temperature for exploratory moves (default: 0.8)")
    parser.add_argument("--temp-moves", type=int, default=3,
                        help="Number of moves with temperature exploration (default: 3)")
    parser.add_argument("--mcts-batch-size", type=int, default=16,
                        help="MCTS evaluation batch size (default: 16)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Mixed precision FP16 (default: on, use --no-amp to disable)")
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="PUCT exploration constant (default: 1.0)")
    parser.add_argument("--full-search-fraction", type=float, default=1.0,
                        help="Fraction of moves with full search (playout cap, default: 1.0 = off)")
    parser.add_argument("--cheap-sims", type=int, default=50,
                        help="Simulations for cheap-search moves (default: 50)")

    # -- Replay buffer --
    parser.add_argument("--buffer-min", type=int, default=20000,
                        help="Starting replay buffer size for growing buffer (default: 20000)")

    # -- Evaluation --
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per evaluation (default: 20)")
    parser.add_argument("--eval-depth", type=int, default=4,
                        help="Opponent depth for evaluation (default: 4)")
    parser.add_argument("--eval-simulations", type=int, default=0,
                        help="MCTS sims for eval (0 = use --simulations, default: 0)")
    parser.add_argument("--gate-threshold", type=float, default=0.0,
                        help="Model gating win-rate threshold (0 = disabled, default: 0.0)")

    # -- Heuristic bootstrapping: decay to 0 for max final strength --
    parser.add_argument("--heuristic-ratio", type=float, default=0.3,
                        help="Starting heuristic game fraction (default: 0.3)")
    parser.add_argument("--heuristic-ratio-end", type=float, default=0.1,
                        help="Ending heuristic game fraction (default: 0.1)")

    # -- Infrastructure --
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps (default: auto)")
    parser.add_argument("--save-dir", type=str, default="alphazero_checkpoints",
                        help="Checkpoint directory (default: alphazero_checkpoints)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel self-play workers (default: 1)")
    parser.add_argument("--bootstrap-games", type=int, default=0,
                        help="Alpha-beta games to generate before training (default: 0)")
    parser.add_argument("--bootstrap-depth", type=int, default=9,
                        help="Alpha-beta search depth for bootstrap games (default: 9)")

    # -- Gumbel AlphaZero (improved policy targets) --
    parser.add_argument("--gumbel", action="store_true", default=False,
                        help="Use Gumbel AlphaZero search (Sequential Halving + "
                             "completed-Q policy targets, replaces PUCT + Dirichlet at root)")
    parser.add_argument("--forward-only", action="store_true", default=False,
                        help="Remove backward barrel moves during self-play (reduces branching "
                             "factor and forces decisive games)")
    parser.add_argument("--repetition-penalty", type=float, default=0.0,
                        help="Penalise MCTS leaf values for positions seen earlier in game "
                             "(0.0=disabled, 0.3=recommended). Value shrinks by penalty*count.")
    parser.add_argument("--max-draw-fraction", type=float, default=1.0,
                        help="Cap draw games in training data (1.0=keep all, 0.33=max 33%% draws). "
                             "Excess draws are randomly discarded to strengthen value signal.")
    parser.add_argument("--mixed-opponent-depth", type=int, default=0,
                        help="Play some training games as network vs alpha-beta at this depth "
                             "(0=disabled). Breaks self-play ceiling by training against a "
                             "stronger external opponent.")
    parser.add_argument("--mixed-fraction", type=float, default=0.0,
                        help="Fraction of network games to play as network-vs-heuristic "
                             "(default: 0.0, e.g. 0.3 = 30%% of non-bootstrap games)")

    # -- Value target blending (dense signal from heuristic eval) --
    parser.add_argument("--value-blend-lambda", type=float, default=0.5,
                        help="Blend: lambda*game_outcome + (1-lambda)*search_score "
                             "(1.0=pure outcome, 0.0=pure search, default: 0.5)")
    # -- Game adjudication (end decisive games early) --
    parser.add_argument("--adjudication-threshold", type=float, default=0.6,
                        help="Adjudicate when |heuristic_eval| > threshold in tanh units "
                             "(0.0=disabled, default: 0.6 ≈ 360cp)")
    parser.add_argument("--adjudication-min-moves", type=int, default=30,
                        help="Minimum moves before adjudication can trigger (default: 30)")
    parser.add_argument("--max-moves", type=int, default=80,
                        help="Maximum moves per self-play game (default: 80)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    checkpoint = save_dir / "latest_model.pt"
    total_iters = args.chunks * args.iters_per_chunk

    print("=" * 60)
    print("ALPHAZERO CHUNKED TRAINING")
    print("=" * 60)
    print(f"  Network: {args.network} (blocks={args.num_blocks}, channels={args.hidden})")
    print(f"  Chunks: {args.chunks} x {args.iters_per_chunk} iters = {total_iters} total iterations")
    print(f"  Games/iter: {args.games_per_iter}")
    print(f"  Simulations: {args.simulations}")
    if args.eval_simulations > 0:
        print(f"  Eval simulations: {args.eval_simulations}")
    print(f"  c_puct: {args.c_puct}")
    print(f"  Workers: {args.workers}")
    print(f"  Training epochs: {args.training_epochs}")
    print(f"  Training batch size: {args.batch_size}")
    print(f"  MCTS batch size: {args.mcts_batch_size}")
    print(f"  Buffer: {args.buffer_min:,} -> {args.buffer_max:,} (growing)")
    print(f"  Train window: {args.train_window:,} examples/iter")
    print(f"  Mixed precision (AMP): {'FP16' if args.amp else 'off'}")
    print(f"  Temperature: {args.temperature} (first {args.temp_moves} moves)")
    if args.gumbel:
        print(f"  Gumbel AlphaZero: Sequential Halving + completed-Q policy targets")
    if args.forward_only:
        print(f"  Forward-only: backward barrel moves removed from self-play")
    if args.repetition_penalty > 0:
        print(f"  Repetition penalty: {args.repetition_penalty} (shrink value by penalty*count)")
    if args.max_draw_fraction < 1.0:
        print(f"  Draw filtering: max {args.max_draw_fraction:.0%} draws in training data")
    if args.mixed_opponent_depth > 0:
        print(f"  Mixed training: {args.mixed_fraction:.0%} of games vs heuristic depth {args.mixed_opponent_depth}")
    if args.full_search_fraction < 1.0:
        print(f"  Playout cap: {args.full_search_fraction:.0%} full ({args.simulations} sims), "
              f"{1-args.full_search_fraction:.0%} cheap ({args.cheap_sims} sims)")
    print(f"  Heuristic ratio: {args.heuristic_ratio:.0%} -> {args.heuristic_ratio_end:.0%}")
    if args.bootstrap_games > 0:
        print(f"  Bootstrap: {args.bootstrap_games:,} alpha-beta games (depth {args.bootstrap_depth})")
    if args.gate_threshold > 0:
        print(f"  Model gating: revert if win rate < {args.gate_threshold:.0%}")
    print(f"  Policy weight: {args.policy_weight}")
    print(f"  Value blend lambda: {args.value_blend_lambda} "
          f"({args.value_blend_lambda:.0%} outcome + {1-args.value_blend_lambda:.0%} search)")
    if args.adjudication_threshold > 0:
        print(f"  Adjudication: |eval| > {args.adjudication_threshold} after move {args.adjudication_min_moves}")
    print(f"  Max moves per game: {args.max_moves}")
    print(f"  Device: {args.device}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Ctrl+C between chunks to stop safely.")
    print("=" * 60)
    print()

    # Create ONE trainer that persists across all chunks
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
        buffer_min=args.buffer_min,
        train_window=args.train_window,
        network_type=args.network,
        num_blocks=args.num_blocks,
        policy_weight=args.policy_weight,
        device=args.device,
        mcts_batch_size=args.mcts_batch_size,
        use_amp=args.amp,
        num_workers=args.workers,
        full_search_fraction=args.full_search_fraction,
        cheap_sims=args.cheap_sims,
        gate_threshold=args.gate_threshold,
        temp_moves=args.temp_moves,
        value_blend_lambda=args.value_blend_lambda,
        adjudication_threshold=args.adjudication_threshold,
        adjudication_min_moves=args.adjudication_min_moves,
        max_moves=args.max_moves,
        use_gumbel=args.gumbel,
        forward_only=args.forward_only,
        repetition_penalty=args.repetition_penalty,
        max_draw_fraction=args.max_draw_fraction,
        eval_simulations=args.eval_simulations or None,
        mixed_opponent_depth=args.mixed_opponent_depth,
        mixed_fraction=args.mixed_fraction,
    )

    # Set LR schedule for total iterations
    trainer.set_lr_schedule(total_iters)

    # Create training logger
    logger = TrainingLogger(args.save_dir)
    logger.log_config(
        network=args.network, hidden=args.hidden, num_blocks=args.num_blocks,
        chunks=args.chunks, iters_per_chunk=args.iters_per_chunk,
        games_per_iter=args.games_per_iter, simulations=args.simulations,
        c_puct=args.c_puct, lr=args.lr, training_epochs=args.training_epochs,
        batch_size=args.batch_size, train_window=args.train_window,
        buffer_max=args.buffer_max, buffer_min=args.buffer_min,
        temperature=args.temperature, temp_moves=args.temp_moves,
        mcts_batch_size=args.mcts_batch_size, amp=args.amp,
        heuristic_ratio=args.heuristic_ratio,
        heuristic_ratio_end=args.heuristic_ratio_end,
        policy_weight=args.policy_weight, device=args.device,
        workers=args.workers,
        full_search_fraction=args.full_search_fraction,
        cheap_sims=args.cheap_sims,
        gate_threshold=args.gate_threshold,
        value_blend_lambda=args.value_blend_lambda,
        adjudication_threshold=args.adjudication_threshold,
        adjudication_min_moves=args.adjudication_min_moves,
        max_moves=args.max_moves,
        gumbel=args.gumbel,
        forward_only=args.forward_only,
        repetition_penalty=args.repetition_penalty,
        max_draw_fraction=args.max_draw_fraction,
        eval_simulations=args.eval_simulations or None,
        mixed_opponent_depth=args.mixed_opponent_depth,
        mixed_fraction=args.mixed_fraction,
    )

    # Resume from checkpoint if available (loads model + replay buffer)
    if checkpoint.exists():
        trainer.load(str(checkpoint))

    # Bootstrap: generate alpha-beta games before training (or load cached)
    if args.bootstrap_games > 0 and len(trainer.replay_buffer) == 0:
        bootstrap_cache = Path(f"bootstrap_d{args.bootstrap_depth}_{args.bootstrap_games}.npz")
        if bootstrap_cache.exists():
            import numpy as np
            print(f"Loading cached bootstrap games from {bootstrap_cache}...")
            data = np.load(bootstrap_cache)
            n_total = len(data["values"])
            effective_max = trainer._effective_buffer_max()
            # Only load last effective_max examples to avoid memory spike
            start = max(0, n_total - effective_max)
            boards = data["boards"][start:]
            policies = data["policies"][start:]
            values = data["values"][start:]
            search_scores = data["search_scores"][start:]
            for i in range(len(values)):
                trainer.replay_buffer.append((
                    boards[i], policies[i],
                    float(values[i]), float(search_scores[i]),
                ))
            del data, boards, policies, values, search_scores
            print(f"Loaded {len(trainer.replay_buffer):,}/{n_total:,} bootstrap examples into buffer.")
        else:
            trainer.generate_bootstrap_games(
                args.bootstrap_games, depth=args.bootstrap_depth,
            )
            # Cache bootstrap data for future runs
            import numpy as np
            boards = np.array([ex[0] for ex in trainer.replay_buffer])
            policies = np.array([ex[1] for ex in trainer.replay_buffer])
            values = np.array([ex[2] for ex in trainer.replay_buffer], dtype=np.float32)
            search_scores = np.array([ex[3] for ex in trainer.replay_buffer], dtype=np.float32)
            np.savez_compressed(bootstrap_cache, boards=boards, policies=policies,
                                values=values, search_scores=search_scores)
            print(f"Bootstrap cached to {bootstrap_cache}")
        # Save immediately so bootstrap games survive cancellation
        trainer._save(checkpoint)
        print(f"Bootstrap buffer saved to {checkpoint}")

    t0 = time.time()

    for chunk in range(1, args.chunks + 1):
        chunk_start = time.time()
        iter_start = (chunk - 1) * args.iters_per_chunk + 1
        iter_end = chunk * args.iters_per_chunk

        # Linearly decay heuristic ratio from start to end over all chunks
        progress = (chunk - 1) / max(1, args.chunks - 1)
        h_ratio = args.heuristic_ratio + progress * (args.heuristic_ratio_end - args.heuristic_ratio)

        print(f"--- Chunk {chunk}/{args.chunks} (iters {iter_start}-{iter_end}/{total_iters})"
              f" [heuristic: {h_ratio:.0%}] ---")

        # Evaluate on last iter of each chunk
        trainer.run(
            iterations=args.iters_per_chunk,
            eval_every=args.iters_per_chunk,
            eval_games=args.eval_games,
            eval_depth=args.eval_depth,
            save_dir=args.save_dir,
            heuristic_ratio=h_ratio,
            logger=logger,
        )

        # Save as latest (for next process to resume from)
        trainer._save(checkpoint)

        chunk_time = time.time() - chunk_start
        total_time = time.time() - t0
        remaining = chunk_time * (args.chunks - chunk)

        print(f"--- Chunk {chunk} done in {chunk_time:.0f}s "
              f"(total {total_time/60:.1f}m, ~{remaining/60:.0f}m remaining) ---")
        print()

    print(f"All {args.chunks} chunks complete in {(time.time()-t0)/60:.1f} minutes.")


if __name__ == "__main__":
    main()
