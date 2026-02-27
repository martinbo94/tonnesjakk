# AlphaZero Training Guide for Tonnesjakk

## Quick Start

```bash
python scripts/train_alphazero.py \
  --chunks 75 --iters-per-chunk 3 \
  --games-per-iter 100 --simulations 200 \
  --workers 10 --amp \
  --c-puct 1.0 --training-epochs 1 \
  --full-search-fraction 0.25 --cheap-sims 50 \
  --train-window 50000 --buffer-min 20000 --buffer-max 100000 \
  --bootstrap-games 10000 --bootstrap-depth 9 \
  --eval-games 20 --eval-depth 4
```

Resume from checkpoint is automatic — `Ctrl+C` between chunks to stop safely.

## Architecture

- **Network**: ResNet with 5 residual blocks, 128 channels (~1.6M parameters)
- **Input**: 5x6x6 planes — relative encoding (my barrels, opp barrels, my pail, opp pail, bias)
- **Policy head**: 1332 outputs (37 x 36 from/to square pairs)
- **Value head**: LeakyReLU + dropout, single scalar in [-1, +1]

## Hyperparameter Recommendations

Based on research on AlphaZero for small games:

### Outer loop > Inner loop (Wang et al. 2020)

The single most important finding: **maximize the number of self-play iterations**, not
simulations, epochs, or games per iteration. The outer loop subsumes the inner loop
parameters. Too much inner-loop training can actually hurt performance.

### Recommended settings

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--simulations` | 200 | Sufficient for b.f. ~16 (Wang et al. 2020) |
| `--c-puct` | 1.0 | Lower exploration suits small games (alpha-zero-general) |
| `--games-per-iter` | 100 | Balance: enough data per iter, many iters total |
| `--training-epochs` | 1 | 1 is optimal for small games (Wang et al. 2020) |
| `--batch-size` | 512 | Standard; buffer fills quickly |
| `--train-window` | 50000 | Train on recent data; old data from weaker network hurts (KataGo) |
| `--buffer-min` | 20000 | Growing buffer start (washes out early random data) |
| `--buffer-max` | 100000 | Growing buffer end; smaller keeps data fresh |
| `--full-search-fraction` | 0.25 | Playout cap: 25% full, 75% cheap (~4x speedup) |
| `--cheap-sims` | 50 | Cheap search simulations |
| `--gate-threshold` | 0.0 | Disabled by default; 0.55 can trap early training |
| `--policy-weight` | 0.5 | Value-heavy loss for small games (Wang & Emmerich 2019) |
| `--amp` | on | FP16 inference + training; ~20-30% faster, no quality loss |

### Training epochs (Wang et al. 2020)

For small games, **1 epoch per iteration is optimal** when combined with many
outer-loop iterations. More epochs cause the network to overfit to the current
buffer, producing narrow self-play data. The outer loop (diverse data from
improving network) matters more than squeezing each batch.

### Value-heavy loss (Wang & Emmerich 2019)

For small games, optimizing value loss alone outperforms the standard AlphaZero
combined loss. The `--policy-weight 0.5` setting biases the loss toward the value
head (total loss = 0.5 * policy_loss + 1.0 * value_loss).

Wang & Emmerich tested on Connect Four (b.f. ~7) and Othello 6x6 (b.f. ~6-8),
where value-only (policy_weight=0) was best. Tonnesjakk has b.f. ~16 — higher,
but still small enough that 200-sim MCTS covers most legal moves. A setting of
0.3-0.5 is a reasonable compromise; going to 0 may work but is untested here.

### Playout cap randomization (KataGo-style)

Playout cap randomization (Wu 2019) makes self-play ~4x faster. Each move randomly
uses either full search (200 sims) or cheap search (50 sims). Only full-search
positions generate training data, keeping data quality high while exploring more
game trajectories per unit of compute.

- `--full-search-fraction 0.25` — 25% of moves use full search
- `--cheap-sims 50` — cheap-search moves use 50 sims
- Default `1.0` disables playout cap (all moves use full search)

### Growing replay buffer

Early self-play data is low-quality (near-random network). A growing buffer starts
small to wash out this data quickly, then grows to retain more as training improves:

- `--buffer-min 20000` — initial buffer capacity
- `--buffer-max 200000` — final buffer capacity
- Buffer grows linearly from min to max over the total training iterations

### Model gating

If eval win rate drops below a threshold, training reverts to the best checkpoint
weights (preserving the replay buffer). This prevents catastrophic forgetting.

- `--gate-threshold 0.55` — revert if win rate < 55%
- Default `0.0` disables gating

### Exploration constant (c_puct)

The PUCT exploration constant controls the balance between exploration and
exploitation in MCTS. The default `1.0` (down from AlphaZero's 1.4) is better
suited to small games with lower branching factors, following alpha-zero-general
and KataGo conventions.

### Heuristic bootstrapping

The cold-start problem is well-documented: a random neural network produces
meaningless value estimates, so early MCTS is essentially random. The
`--heuristic-ratio` flag mixes in alpha-beta self-play games (pure Rust, fast,
decisive outcomes) to give the value head grounded training signal.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--heuristic-ratio` | 0.3 | 30% heuristic games at start (depth 7 in-training) |
| `--heuristic-ratio-end` | 0.0 | Decay to 0% (pure self-play) |
| `--bootstrap-games` | 10000 | Pre-fill buffer with strong play |
| `--bootstrap-depth` | 9 | Sweet spot: +322 ELO over D4, 33 min with 10 workers |

### Workers

For Apple M4 Pro (14 CPU cores, 20 GPU cores):
- `--workers 10` — leaves cores free for MPS GPU inference + OS
- Network games (Python NN callback) are the bottleneck, not heuristic games

## Training Dynamics

### The cold-start plateau

AlphaZero training on small games typically shows a **long flat period followed by
a sharp improvement**. This is because the policy and value heads form a
chicken-and-egg loop:

1. The policy head needs accurate values to learn which moves lead to wins
2. The value head needs good games (from good policy) to learn position evaluation
3. Early on, both are weak, so self-play games are near-random and uninformative

Once one head gets "good enough", the other follows quickly — creating the
characteristic inflection point.

### What to watch in the logs

The training log (`{save_dir}/training.log`) contains JSONL records. Key metrics:

| Metric | Healthy sign | Concern |
|--------|-------------|---------|
| `policy_loss` | Steadily decreasing | Flat or increasing |
| `value_loss` | Decreasing (may lag policy) | Increasing over many iters |
| `search_score_mae` | Decreasing toward 0.3-0.4 | Stuck above 0.6 |
| `search_score_std` | 0.4-0.6 (moderate confidence) | Near 0 (overconfident) or near 1 (random) |
| `search_score_mean` | Near 0 (balanced games) | Strong bias toward +1 or -1 |
| Eval W-D-L | First draws, then wins appear | 0W-0D-20L for 50+ chunks |

Parse with jq:
```bash
# Loss progression
jq -r 'select(.type=="iteration") | "\(.iteration) p=\(.policy_loss) v=\(.value_loss) mae=\(.search_score_mae)"' training.log

# Eval progression
jq -r 'select(.type=="eval") | "iter \(.iteration): \(.wins)W-\(.draws)D-\(.losses)L ELO=\(.elo)"' training.log
```

### Per-position search scores

The training log only stores aggregate search_score statistics per iteration.
For per-position analysis (e.g. tracing how the engine evaluated each move in a
game), load the replay buffer directly:

```python
import numpy as np

data = np.load("alphazero_checkpoints/latest_model.pt.buffer.npz")
search_scores = data["search_scores"]  # per-position search eval (current player perspective)
values = data["values"]                # game outcome per position

# Agreement between search eval and game outcome
mae = np.mean(np.abs(search_scores - values))
print(f"Search-outcome MAE: {mae:.3f}")

# Distribution of search confidence
print(f"Search score std: {np.std(search_scores):.3f}")
```

## If Training Stalls

If eval shows 0W-0D-20L after 50+ chunks (~150 iterations, ~15k games):

1. **Fresh bootstrap run**: Generate strong examples before self-play begins
   ```bash
   python scripts/train_alphazero.py \
     --bootstrap-games 10000 --bootstrap-depth 9 \
     --chunks 75 --iters-per-chunk 3 \
     --save-dir alphazero_fresh
   ```

2. **Lower eval depth**: Try `--eval-depth 3` or 2. You may already be beating
   a weaker opponent — depth 4 is a competent player.

3. **More simulations**: 400-800 sims compensates for a weak value head at the
   cost of slower games. Consider halving `--games-per-iter` to keep wall time
   constant.

4. **Larger train window**: `--train-window 50000` lets the network see more of
   its experience per iteration.

5. **Reduce exploration**: Lower `temp_moves` from 15 to 8-10 for stronger
   self-play sooner (currently hardcoded in game loops).

## Research Findings Summary

| Technique | Source | Effect | Default |
|-----------|--------|--------|---------|
| Maximize outer loop | Wang et al. 2020 | Most important factor for convergence | Many chunks x few iters |
| Value-heavy loss | Wang & Emmerich 2019 | Better for small games | `--policy-weight 0.5` |
| Heuristic bootstrapping | Wang et al. 2020 | Solves cold-start problem | `--heuristic-ratio 0.3` |
| Playout cap randomization | Wu 2019 (KataGo) | ~4x more games for same compute | `--full-search-fraction 0.25` |
| Lower c_puct | alpha-zero-general | Less exploration for small games | `--c-puct 1.0` |
| Growing replay buffer | KataGo, alpha-zero-general | Washes out low-quality early data | `--buffer-min 20000` |
| Model gating | alpha-zero-general | Prevents catastrophic forgetting | `--gate-threshold 0.55` |

## Timing Reference (M4 Pro, 10 workers)

| Configuration | Time/iteration |
|--------------|----------------|
| 30% heuristic, 200 sims, 100 games | ~120s |
| 0% heuristic, 200 sims, 100 games | ~245s |
| With playout cap (25% full) | ~60-80s (estimated) |
| Eval (20 games, depth 4) | ~20s |
| 3-iter chunk + eval | ~6-11 min |

## References

- Wang & Emmerich. "Policy or Value? Loss Function and Playing Strength in
  AlphaZero-like Self-play." CoG 2019.
  https://www.semanticscholar.org/paper/b125c8933d0264b9a103cb8fa80f226f8c9c3cdc

- Wang, Emmerich, Preuss & Plaat. "Analysis of Hyper-Parameters for Small Games:
  Iterations or Epochs in Self-Play?" 2020.
  https://arxiv.org/abs/2003.05988

- Wang, Emmerich, Preuss & Plaat. "Warm-Start AlphaZero Self-Play Search
  Enhancements." 2020.
  https://arxiv.org/abs/2004.12357

- Wu. "Accelerating Self-Play Learning in Go." 2019. (KataGo)
  https://arxiv.org/abs/1902.10565

- Gao, Muller & Plaat. "Adaptive Warm-Start MCTS in AlphaZero-Like Deep
  Reinforcement Learning." 2021.
  https://arxiv.org/abs/2105.06136

- Surag Nair, alpha-zero-general. 2018-2024.
  https://github.com/suragnair/alpha-zero-general

- Zhao et al. "Efficient Learning for AlphaZero via Path Consistency." ICML 2022.
  https://proceedings.mlr.press/v162/zhao22h/zhao22h.pdf

- Uiterwijk. "Perfectly Solving Domineering Boards." 2024.
  (Oracle Connect Four series for small-board game insights)
