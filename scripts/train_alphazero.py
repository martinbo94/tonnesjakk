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
  .venv\Scripts\python.exe scripts\train_alphazero.py
  .venv\Scripts\python.exe scripts\train_alphazero.py --network resnet --chunks 60
  .venv\Scripts\python.exe scripts\train_alphazero.py --network mlp --save-dir az_mlp
"""

import argparse
import time
from pathlib import Path

from tonnesjakk.alphazero import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description="AlphaZero training in resumable chunks")
    parser.add_argument("--chunks", type=int, default=20,
                        help="Number of chunks to run (default: 20)")
    parser.add_argument("--iters-per-chunk", type=int, default=3,
                        help="Iterations per chunk (default: 3)")
    parser.add_argument("--games-per-iter", type=int, default=30,
                        help="Self-play games per iteration (default: 30)")
    parser.add_argument("--simulations", type=int, default=400,
                        help="MCTS simulations per move (default: 400)")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Network channels/hidden size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--training-epochs", type=int, default=10,
                        help="Training epochs per iteration (default: 10)")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Games per evaluation (default: 20)")
    parser.add_argument("--eval-depth", type=int, default=4,
                        help="Opponent depth for evaluation (default: 4)")
    parser.add_argument("--heuristic-ratio", type=float, default=0.5,
                        help="Starting heuristic game fraction (default: 0.5)")
    parser.add_argument("--heuristic-ratio-end", type=float, default=0.25,
                        help="Ending heuristic game fraction (default: 0.25)")
    parser.add_argument("--network", type=str, default="resnet", choices=["resnet", "mlp"],
                        help="Network architecture (default: resnet)")
    parser.add_argument("--num-blocks", type=int, default=5,
                        help="Residual blocks for resnet (default: 5)")
    parser.add_argument("--train-window", type=int, default=20000,
                        help="Max examples to train on per iteration (default: 20000)")
    parser.add_argument("--save-dir", type=str, default="alphazero_checkpoints",
                        help="Checkpoint directory (default: alphazero_checkpoints)")
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
    print(f"  Training epochs: {args.training_epochs}")
    print(f"  Train window: {args.train_window:,} examples/iter")
    print(f"  Heuristic ratio: {args.heuristic_ratio:.0%} -> {args.heuristic_ratio_end:.0%}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Ctrl+C between chunks to stop safely.")
    print("=" * 60)
    print()

    # Create ONE trainer that persists across all chunks
    trainer = AlphaZeroTrainer(
        hidden=args.hidden,
        simulations=args.simulations,
        c_puct=1.4,
        lr=args.lr,
        games_per_iter=args.games_per_iter,
        training_epochs=args.training_epochs,
        batch_size=256,
        temperature=1.0,
        buffer_max=100000,
        train_window=args.train_window,
        network_type=args.network,
        num_blocks=args.num_blocks,
    )

    # Resume from checkpoint if available (loads model + replay buffer)
    if checkpoint.exists():
        trainer.load(str(checkpoint))

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
