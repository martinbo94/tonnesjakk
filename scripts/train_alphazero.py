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
    parser.add_argument("--hidden", type=int, default=128,
                        help="Network channels/hidden size (default: 128)")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "mlp"],
                        help="Network architecture (default: resnet)")
    parser.add_argument("--num-blocks", type=int, default=5,
                        help="Residual blocks for resnet (default: 5)")

    # -- Training: keep inner-loop params low (Wang et al. 2020) --
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--training-epochs", type=int, default=2,
                        help="Training epochs per iteration (default: 2)")
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
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Self-play temperature for first 15 moves (default: 1.0)")
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
    print(f"  c_puct: {args.c_puct}")
    print(f"  Workers: {args.workers}")
    print(f"  Training epochs: {args.training_epochs}")
    print(f"  Training batch size: {args.batch_size}")
    print(f"  MCTS batch size: {args.mcts_batch_size}")
    print(f"  Buffer: {args.buffer_min:,} -> {args.buffer_max:,} (growing)")
    print(f"  Train window: {args.train_window:,} examples/iter")
    print(f"  Mixed precision (AMP): {'FP16' if args.amp else 'off'}")
    print(f"  Temperature: {args.temperature}")
    if args.full_search_fraction < 1.0:
        print(f"  Playout cap: {args.full_search_fraction:.0%} full ({args.simulations} sims), "
              f"{1-args.full_search_fraction:.0%} cheap ({args.cheap_sims} sims)")
    print(f"  Heuristic ratio: {args.heuristic_ratio:.0%} -> {args.heuristic_ratio_end:.0%}")
    if args.bootstrap_games > 0:
        print(f"  Bootstrap: {args.bootstrap_games:,} alpha-beta games (depth {args.bootstrap_depth})")
    if args.gate_threshold > 0:
        print(f"  Model gating: revert if win rate < {args.gate_threshold:.0%}")
    print(f"  Policy weight: {args.policy_weight}")
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
        temperature=args.temperature,
        mcts_batch_size=args.mcts_batch_size, amp=args.amp,
        heuristic_ratio=args.heuristic_ratio,
        heuristic_ratio_end=args.heuristic_ratio_end,
        policy_weight=args.policy_weight, device=args.device,
        workers=args.workers,
        full_search_fraction=args.full_search_fraction,
        cheap_sims=args.cheap_sims,
        gate_threshold=args.gate_threshold,
    )

    # Resume from checkpoint if available (loads model + replay buffer)
    if checkpoint.exists():
        trainer.load(str(checkpoint))

    # Bootstrap: generate alpha-beta games before training
    if args.bootstrap_games > 0 and len(trainer.replay_buffer) == 0:
        trainer.generate_bootstrap_games(
            args.bootstrap_games, depth=args.bootstrap_depth,
        )
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
