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

## Value Target Blending (v8/v9)

Standard AlphaZero uses only the game outcome (+1/-1) as the value target. This is
problematic for tonnesjakk because barrels can move backwards, leading to indecisive
self-play games (~60-70% draws) that give near-zero value signal.

**Solution**: Blend the game outcome with the per-position heuristic search score:

```
value_target = lambda * game_outcome + (1 - lambda) * search_score
```

This is the same approach Stockfish NNUE uses (lambda=0.85), and is a simpler form
of the temporal consistency idea from Zhao et al. 2022. It provides dense, per-position
signal rather than sparse game-outcome signal.

- `--value-blend-lambda 0.5` — 50% outcome + 50% heuristic eval (default)
- `--value-blend-lambda 1.0` — pure game outcome (original AlphaZero)

## Game Adjudication (v8/v9)

End self-play games early when the heuristic eval indicates a decisive advantage.
Reduces aimless wandering that generates low-quality training data.

- `--adjudication-threshold 0.4` — adjudicate at ~254cp (tanh scale)
- `--adjudication-min-moves 30` — don't adjudicate before move 30
- `--max-moves 50` — hard cap on game length (was 80)

**Accuracy analysis** (from random positions, played out with depth-5 heuristic):

| Threshold | ~Centipawns | Accuracy (decisive games) |
|-----------|-------------|--------------------------|
| 0.3 | 186cp | 87% |
| 0.4 | 254cp | 90% |
| 0.5 | 330cp | 94% |
| 0.6 | 416cp | 95% |

Adjudication uses static eval (depth 0) which is essentially free. Depth-3 eval
would add ~28s/iteration overhead — not worth the marginal accuracy gain.

## Training Run Results

### v7 (baseline, no blending/adjudication)
- 314 iterations, value loss plateaued at 0.22
- search_score_mae stuck at ~0.6
- Best eval: 4W-0D-16L vs depth 4 (ELO -241)
- ~30% draws in self-play, straggler iterations up to 960s

### v8 (value blending lambda=0.5, adjudication threshold=0.6, max_moves=80)
- 234 iterations, value loss reached 0.077 (vs v7's 0.22)
- Best eval: 4W-1D-15L vs depth 4 (ELO -215)
- Straggler problem: some iterations 960-1020s due to max_moves=80
- Draw rate ~55%, down from v7's ~65%

### v9 (value blending lambda=0.5, adjudication threshold=0.4, max_moves=50)
- 224+ iterations (ongoing), value loss reached 0.077
- Best eval: **6W-0D-14L vs depth 4 (ELO -147)** at iteration ~108
- Zero straggler iterations — max_moves=50 fixed the long game problem
- Draw rate dropped to 43-52% in mid-training
- Eval shows consistent 1-5 wins from chunk 30 onward
- Best result across all training runs so far

### Key takeaways
- Value blending halved the value loss (0.22 → 0.077) — the value head actually learns
- Lower adjudication threshold (0.4 vs 0.6) + lower max_moves (50 vs 80) improved
  both speed (no stragglers) and eval results
- The network still peaks early and draw rate creeps up — may need further
  investigation (exploration decay, learning rate, or reward shaping)

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

---

## Diagnostic Investigation: v16 vs v9 (2026-03-15)

### Setup
Ran `scripts/diagnose_training.py` comparing v16 (128ch, Gumbel, forward-only, 100 sims) vs v9 (64ch, standard MCTS, 200 sims) in 20 games each against depth-4 heuristic.

### Bug Found: MCTS Tree Reuse Across Pail Sub-Moves

**Critical finding**: Python-level game loops (`watch_game.py`, diagnostic) were broken by MCTS tree reuse. Tønnesjakk has two-phase turns (pail placement → barrel placement). When `search_network_batched` was called twice per turn, the second call reused the stale pail-phase tree, causing MCTS to return pail moves instead of barrel moves. Games never progressed beyond placement phase.

**Not affected**: Training self-play and evaluation (`play_eval_match_impl`) handle pail sub-moves separately with `random_center_pail()` in Rust, so training was never broken.

**Fix**: Both `diagnose_training.py` and `watch_game.py` now handle pail sub-moves with center-biased random selection (matching Rust) and create fresh MCTS engines per barrel search.

### Eval Game Results (after fix)

| Metric | v16 (128ch, 100sim) | v9 (64ch, 200sim) |
|---|---|---|
| W-D-L vs depth-4 | 7-4-9 | 7-3-10 |
| Avg game length | 43 moves | 44 moves |
| Net barrel advancement | +0.684 rows/move | +0.733 rows/move |
| Heuristic advancement | +0.751 rows/move | +0.771 rows/move |
| 3-fold repetitions | 4/20 games | 3/20 games |

### Replay Buffer Analysis

| Metric | v16 | v9 |
|---|---|---|
| Buffer size | 123,679 | 69,600 |
| Value: Win/Draw/Loss | 17% / 65% / 18% | 19% / 59% / 22% |
| Policy top-1 accuracy | 51.0% | 46.0% |
| MCTS target entropy | 1.44 | 2.04 |
| Search score |score|>0.5 | 13.2% | 14.6% |

### Key Findings

1. **Both models are roughly equal** — neither clearly dominates. Both lose to depth-4 heuristic (~35% win rate).

2. **Heuristic advances barrels faster** — +0.75 rows/move vs +0.68-0.73 for network. Heuristic makes aggressive +4 cross-board jumps; network prefers cautious +1 steps.

3. **Value head is poorly calibrated and systematically optimistic** — In v16 Loss #2 (game 1), the network's MCTS value stays positive (+0.497 to +0.602) for 30+ moves while actually losing W4-B2. The heuristic correctly reads the position at +0.9 throughout. The network can't tell it's behind until it's too late.

4. **Too many draws → weak value signal** — 65% of v16's buffer (59% of v9's) are draws with value target ~0. Value head learns to output ~0 for most positions, destroying its ability to distinguish winning from losing.

5. **Raw policy priors are near-uniform** — In several critical positions, `net=0.000` for all moves (rounded from ~1/1332). With near-uniform priors, 200 simulations are spread across ~16 legal moves (~12 sims/move), giving very shallow effective search depth.

6. **Games collapse into repetition loops** — Network bounces barrels back and forth (e.g., `(2,2)->(2,3)` repeatedly) while thinking it's ahead. Heuristic also loops. Neither breaks out because 3-fold detection only terminates games — there's no cost for approaching repetition.

### Root Cause Analysis

The core issue is a **draw spiral**: too many draws → weak value targets → value head can't distinguish positions → network plays aimlessly → more draws. The heuristic breaks this because it has hand-crafted evaluation that knows barrel advancement and scoring potential from day one.

### Planned Interventions

1. **Repetition penalty in MCTS search** (highest priority) — Penalize positions that appear in the game's position history during MCTS tree evaluation. This directly breaks the shuffle-loops and produces more decisive training games. Implemented as a new `repetition_penalty` parameter on MCTSEngine.

2. **Policy bootstrapping from heuristic** (future) — During heuristic self-play fraction, record heuristic move preferences as soft policy targets. Gives the policy head a head start so MCTS searches more efficiently.

3. **Tune training signal density** (future) — Lower max_moves (50→40), increase early heuristic_ratio (50%→15% vs 30%→10%), increase value_blend_lambda toward 0.7.
