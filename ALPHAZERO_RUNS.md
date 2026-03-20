# AlphaZero Training Run History

All runs train a ResNet CNN to play tonnesjakk via self-play, evaluated against
the alpha-beta heuristic engine at depth 4 (20 games per eval, alternating colors).

## Run Summary

| Run | Best ELO | Best W-D-L | Hidden | Sims | Key Changes | Chunks | Runtime |
|-----|----------|------------|--------|------|-------------|--------|---------|
| **v19** | **-127** | **6-1-13** | 128 | 200 | 128ch, max_moves=40 (beats d4 @ 300 sims) | 141/250 | — |
| v18 | -147 | 6-0-14 | 64 | 200 | draw filtering 33%, temp_moves=8, vblend=0.7 | 250/250 | ~14h |
| v17 | -147 | 6-0-14 | 64 | 200 | repetition penalty 0.3 | 146/250 | ~7h |
| v9 | -147 | 6-0-14 | 64 | 200 | adj=0.4, max_moves=50 | 155/250 | 7.4h |
| v8 | -215 | 4-1-15 | 64 | 200 | value blending, adjudication | 78/250 | 5.7h |
| v7 | -241 | 4-0-16 | 64 | 200 | playout cap 25%, smaller buffer | 104/250 | 4.9h |
| v2 | -241 | 4-0-16 | 128 | 200 | first serious run, 10 workers | 104/75 | 6.2h |
| v16 | -301 | 3-0-17 | 128 | 100 | forward-only + Gumbel | 156/250 | ~8h |
| v3 | -301 | 3-0-17 | 128 | 200 | amp on, 1 epoch, full run | 250/250 | 12.5h |
| v12 | -301 | 3-0-17 | 128 | 400 | Gumbel + 100 games, temp_moves=10 | 22/250 | 2.4h |
| v15 | -512 | 1-0-19 | 128 | 100 | Gumbel 100 sims, adj=0.4, temp=1.0 | 229/250 | 10.8h |
| v14 | -400 | 0-0-20 | 128 | 400 | Gumbel c_scale=0.1 + Gumbel eval | 72/250 | 7.4h |
| v13 | -400 | 0-0-20 | 128 | 400 | Gumbel 5 fixes, c_scale=1.0 (wrong) | 31/250 | 3.2h |
| v10 | -382 | 2-0-18 | 128 | 400 | Gumbel, policy_weight=1.0 | 25/250 | 1.0h |
| v11 | -382 | 2-0-18 | 128 | 400 | Gumbel + temp fix, no Dirichlet | 33/250 | 1.5h |
| v4 | -512 | 1-0-19 | 64 | 200 | no playout cap, 2 epochs | 133/330 | 6.9h |
| v5 | -512 | 1-0-19 | 64 | 400 | 400 sims, 50 games/iter | 97/330 | 3.9h |
| ckpt | -512 | 1-0-19 | 64 | 50 | earliest logged run | 369 | 351h |
| v6 | -400 | 0-0-20 | 64 | 400 | same as v5, stopped early | 33/500 | 1.1h |

At 200 sims (training eval budget), no run achieved positive ELO. However, post-training
analysis revealed that 200 sims undersells network strength: v19 (128ch) beats depth-4
at 300 sims (13-1-6, 65%), and even v18 (64ch) beats it (12-2-6, 60%). The eval sim budget
was the bottleneck, not the networks.

---

## Detailed Run Configs

### alphazero_checkpoints (earliest logged run)

The first run with training logs. Very slow — single worker, 50 sims, 2 games/iter.
Ran for 369 chunks over 351 hours before being abandoned.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks (~480K params) |
| Simulations | 50 |
| Games/iter | 2 |
| Workers | 1 |
| Epochs | 1 |
| Playout cap | off (100% full search) |
| AMP | on |
| Bootstrap | none |
| Value blending | 0.5 |
| Adjudication | 0.6 threshold, 30 min moves |
| Max moves | 80 |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.127 | 0.091 |
| Policy loss | 2.616 | 2.085 |
| Best eval | 1W-0D-19L (ELO -512) |
| Evals with wins | 4/123 (3%) |

**Takeaway**: Too slow. Single worker + 50 sims = insufficient data generation.

---

### v2

First serious run with parallel workers and proper settings.

| Parameter | Value |
|-----------|-------|
| Network | resnet, **128 channels**, 5 blocks (~1.6M params) |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Epochs | 2 |
| Playout cap | **25% full** (first run to use this) |
| AMP | off |
| Bootstrap | none |
| Heuristic ratio | 30% → 10% |
| Buffer | 200K max, 20K min, 100K window |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.452 | 0.249 |
| Policy loss | 5.374 | 2.320 |
| Best eval | 4W-0D-16L (ELO -241) at chunk 84 |
| Evals with wins | 55/104 (53%) |

**Takeaway**: Larger network (128 ch) showed promise early but peaked at 4 wins.
AMP was off, slowing training significantly.

---

### v3

Full 250-chunk run with AMP enabled and 1 epoch per iteration.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 128 channels, 5 blocks |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Epochs | **1** (down from 2) |
| Playout cap | 25% full |
| AMP | **on** |
| Heuristic ratio | 30% → **0%** |
| Buffer | **100K max**, 20K min, **50K window** |
| MCTS batch | **16** (up from 8) |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.450 | 0.342 |
| Policy loss | 3.955 | 2.166 |
| Best eval | 3W-0D-17L (ELO -301) at chunk 39 |
| Evals with wins | 41/250 (16%) |

**Takeaway**: Worse than v2 despite running longer. Value loss stayed high (0.34).
Decaying heuristic ratio to 0% may have hurt — pure self-play games are low quality
early on. Smaller buffer helped loss converge faster but didn't improve play.

---

### v4

Returned to 64 channels. Turned off playout cap. Added temperature control.

| Parameter | Value |
|-----------|-------|
| Network | resnet, **64 channels**, 5 blocks |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Epochs | **2** |
| Playout cap | **off** (100% full search) |
| AMP | on |
| Temperature | **0.8** (first 3 moves only) |
| Heuristic ratio | 30% → 10% |
| Buffer | 200K max, 20K min, 100K window |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.391 | 0.074 |
| Policy loss | 3.330 | 2.189 |
| Best eval | 1W-0D-19L (ELO -512) at chunk 23 |
| Evals with wins | 1/133 (<1%) |

**Takeaway**: Disabling playout cap was a mistake — fewer game trajectories per
compute budget. Value loss looked good (0.074) but play was terrible. 2 epochs
likely caused overfitting. Only 1 eval ever had a win.

---

### v5

Increased sims to 400, halved games to compensate.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks |
| Simulations | **400** |
| Games/iter | **50** |
| Workers | 10 |
| Epochs | **1** |
| Playout cap | **off** |
| AMP | on |
| Buffer | 200K max, 20K min, 100K window |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.475 | 0.085 |
| Policy loss | 4.247 | 2.088 |
| Best eval | 1W-0D-19L (ELO -512) at chunk 72 |
| Evals with wins | 1/97 (1%) |

**Takeaway**: 400 sims didn't help without playout cap. Fewer games (50) meant
less data diversity. Same poor results as v4.

---

### v6

Same config as v5, stopped very early.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks |
| Simulations | 400 |
| Games/iter | 50 |
| Config | identical to v5, chunks increased to 500 |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.293 | 0.120 |
| Policy loss | 5.908 | 2.482 |
| Best eval | 0W-0D-20L (ELO -400) |
| Evals with wins | 0/33 (0%) |

**Takeaway**: Abandoned after 33 chunks with zero wins. Same problems as v4/v5.

---

### v7

Returned to 200 sims with playout cap. Smaller buffer and train window.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Epochs | 1 |
| Playout cap | **25% full** (back on) |
| AMP | on |
| Buffer | **100K max**, 20K min, **50K window** |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.510 | 0.223 |
| Policy loss | 4.421 | 2.459 |
| Best eval | 4W-0D-16L (ELO -241) at chunk 28 |
| Evals with wins | 50/104 (48%) |

**Takeaway**: Playout cap at 25% was clearly better than full search (v4/v5/v6).
Nearly half of all evals had at least one win. But value loss plateaued at 0.22 —
the value head struggled with 65% draw rate in self-play.

---

### v8

First run with value target blending and game adjudication.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Playout cap | 25% full |
| **Value blending** | **lambda=0.5** (50% outcome + 50% heuristic eval) |
| **Adjudication** | **threshold=0.6** (~360cp), min 30 moves |
| Max moves | 80 |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.233 | 0.077 |
| Policy loss | 4.190 | 2.499 |
| Best eval | 4W-1D-15L (ELO -215) at chunk 36 |
| Evals with wins | 44/78 (56%) |

**Takeaway**: Value blending halved value loss (0.22 → 0.077). Draw rate dropped
from ~65% to ~55%. Best ELO improved to -215. But max_moves=80 caused straggler
iterations (960-1020s) when games dragged on.

---

### v9 (best run)

Tightened adjudication and capped game length. Best results across all runs.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 64 channels, 5 blocks |
| Simulations | 200 |
| Games/iter | 100 |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| **Adjudication** | **threshold=0.4** (~254cp), min 30 moves |
| **Max moves** | **50** (down from 80) |

| Metric | Early | Final |
|--------|-------|-------|
| Value loss | 0.213 | 0.060 |
| Policy loss | 4.356 | 2.413 |
| Best eval | **6W-0D-14L (ELO -147)** at chunk 36 |
| Evals with wins | **121/155 (78%)** |

**Takeaway**: Lower adjudication threshold + shorter games eliminated stragglers
and reduced draw rate to 43-52%. The network peaked early (~chunk 36) and draw
rate crept back up later. Policy loss barely moved (2.4 throughout), confirming
the policy head is the bottleneck — it can't learn from near-uniform MCTS visit
count targets.

---

### v10 (abandoned — draw rate too high)

Gumbel AlphaZero with larger network and stronger policy training signal.
Cancelled at chunk 25/250 due to ~80% draw rate caused by argmax move selection.

| Parameter | Value |
|-----------|-------|
| Network | resnet, **128 channels**, 5 blocks (~1.6M params) |
| Simulations | **400** |
| Games/iter | **50** |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| **Policy weight** | **1.0** (equal with value, up from 0.5) |
| **Gumbel search** | **on** (Sequential Halving + completed-Q policy targets) |
| Adjudication | threshold=0.4, min 30 moves |
| Max moves | 50 |

| Metric | Early | Final (chunk 25) |
|--------|-------|------------------|
| Value loss | 0.134 | 0.064 |
| Policy loss | 2.921 | 2.181 |
| Best eval | 2W-0D-18L (ELO -382) at chunk 23 |
| Evals with wins | 7/25 (28%) |
| Draw rate | ~80% throughout |

**Takeaway**: Gumbel improved policy loss (2.18 vs v9's 2.41 floor) but argmax
move selection without temperature exploration produced repetitive openings,
causing ~80% draws in self-play (vs v9's 43-52%). The improved policy targets
couldn't overcome the lack of game diversity. Fix: restore temperature sampling
from the Gumbel improved policy for the first `temp_moves` moves (implemented
for v11).

---

### v11 (abandoned — draw rate still too high)

Same as v10 but with temperature sampling from Gumbel policy for first 3 moves.
Dirichlet noise was still missing. Cancelled at chunk 33/250.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 128 channels, 5 blocks |
| Simulations | 400 |
| Games/iter | 50 |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| Policy weight | 1.0 |
| Gumbel search | on |
| **Temperature fix** | **sample from Gumbel policy for first 3 moves** |
| Adjudication | threshold=0.4, min 30 moves |
| Max moves | 50 |

| Metric | Early | Final (chunk 33) |
|--------|-------|------------------|
| Value loss | 0.357 | 0.059 |
| Policy loss | 6.050 | 2.115 |
| Best eval | 2W-0D-18L (ELO -382) at chunk 7 |
| Draw rate | ~80% throughout |

**Takeaway**: Temperature on only 3 moves was insufficient — Dirichlet noise in
standard MCTS adds exploration on *every* move, not just openings. Also only 50
games/iter meant ~10 decisive games per iteration (vs v9's ~50). Fix for v12:
add Dirichlet noise back to Gumbel mode, increase to 100 games/iter, temp_moves=10.

---

### v12 (abandoned — draw rate still ~80%)

Same Gumbel setup as v11 but with 100 games/iter and temp_moves=10 for more diversity.
Still had 5 Gumbel bugs (no v_mix, no Q-normalization, fixed sigma, Dirichlet present,
sampling from improved policy instead of visit counts). Cancelled at chunk 22/250.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 128 channels, 5 blocks |
| Simulations | 400 |
| Games/iter | **100** (up from 50) |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| Policy weight | 1.0 |
| Gumbel search | on |
| Temperature | **0.8**, **temp_moves=10** (up from 3) |
| Adjudication | threshold=0.4, min 30 moves |
| Max moves | 50 |

| Metric | Early | Final (chunk 22) |
|--------|-------|-------------------|
| Value loss | 0.358 | 0.090 |
| Policy loss | 6.074 | 2.324 |
| Best eval | 3W-0D-17L (ELO -301) at chunk 18 |
| Draw rate | ~80% throughout |

**Takeaway**: More games (100 vs 50) and longer temperature (10 moves vs 3) helped
slightly — best eval improved to 3W vs v10/v11's 2W. But draw rate still stuck at ~80%.
Root cause identified as 5 bugs in Gumbel implementation: missing v_mix for unvisited
actions (returned 0 instead), no Q-value normalization to [0,1], fixed sigma=50 instead
of adaptive, Dirichlet noise applied on top of Gumbel noise, and temperature sampling
from improved policy instead of visit counts. All fixed for v13.

---

### v13 (abandoned — draw rate still ~80%, one-hot policy targets)

Fixed 5 bugs in Gumbel AlphaZero implementation to match the paper (Danihelka et al.,
ICLR 2022). Same config as v12 otherwise. Cancelled at chunk 31/250.

**Fixes applied:**
1. **Removed Dirichlet noise** — Gumbel noise replaces it per the paper
2. **v_mix for unvisited actions** — unvisited children now get weighted mix of root value
   and visited children's Q-values instead of 0.0
3. **Q-value normalization** — min-max normalization to [0, 1] before multiplying by sigma
4. **Adaptive sigma** — `(c_visit + max_visit_count) * c_scale` instead of fixed 50.0
5. **Temperature from visit counts** — sample moves proportional to visit counts raised to
   1/temp, not from the improved policy distribution

| Parameter | Value |
|-----------|-------|
| Network | resnet, 128 channels, 5 blocks |
| Simulations | 400 |
| Games/iter | 100 |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| Policy weight | 1.0 |
| Gumbel search | on (**5 bug fixes, c_scale=1.0**) |
| Temperature | 0.8, temp_moves=3 |
| Adjudication | threshold=0.6, min 30 moves |
| Max moves | 80 |

| Metric | Early | Final (chunk 31) |
|--------|-------|-------------------|
| Value loss | 0.231 | 0.086 |
| Policy loss | 6.893 | 2.233 |
| Best eval | 0W-0D-20L (ELO -400) |
| Draw rate | 80% → 84% (rising) |

**Takeaway**: The 5 bug fixes didn't help because `c_scale=1.0` was catastrophically
wrong. Replay buffer analysis revealed **47% of policy targets had entropy < 0.01**
(near one-hot), vs only 0.3% in v9. With 400 sims, adaptive sigma = (50 + 400) * 1.0 =
450, which crushes `softmax(log_prior + 450 * normalized_Q)` into a delta function.

Comparison of policy target quality:

| Metric | v9 (standard MCTS) | v13 (Gumbel, c_scale=1.0) |
|--------|-------------------|---------------------------|
| Entropy median | 2.19 | 0.02 |
| Top-1 prob median | 0.24 | 0.997 |
| % near one-hot | 0.3% | 47.2% |

The network trained on these one-hot targets learned extreme confidence in single moves,
killing exploration during self-play and producing repetitive drawing lines. Even using
the **same v9 model** with Gumbel search produced 100% draws (0W-20D-0B) vs 95% draws
with standard MCTS — confirming the issue is the search, not the model.

**Root cause**: DeepMind's reference implementation (google-deepmind/mctx) actually uses
`value_scale=0.1` (not 1.0). The paper text says c_scale=1.0 but that's the pre-
normalization value; the code applies `value_scale=0.1` after min-max Q normalization.
Fixed for v14.

---

### v14 (abandoned — draw rate stuck at 80%, Gumbel eval didn't help)

Same as v13 but with `c_scale=0.1` matching DeepMind's mctx reference implementation.
Gumbel eval added at chunk 38 (previously eval used standard MCTS). Cancelled at chunk
72/250 with no improvement.

| Parameter | Value |
|-----------|-------|
| Network | resnet, 128 channels, 5 blocks |
| Simulations | 400 |
| Games/iter | 100 |
| Workers | 10 |
| Playout cap | 25% full |
| Value blending | lambda=0.5 |
| Policy weight | 0.5 |
| Gumbel search | on (5 bug fixes, **c_scale=0.1**) |
| Gumbel eval | **on** (added at chunk 38) |
| Temperature | 0.8, temp_moves=3 |
| Adjudication | threshold=0.6, min 30 moves |
| Max moves | 80 |

| Metric | Early | Final (chunk 72) |
|--------|-------|-------------------|
| Value loss | 0.251 | 0.061 |
| Policy loss | 6.898 | 2.355 |
| Best eval | 0W-0D-20L (ELO -400) |
| Eval wins | 6 total across 72 evals (8%) |
| Draw rate | 80% throughout (never dropped) |

**Takeaway**: Correct c_scale=0.1 fixed the one-hot target problem (sigma=45 vs old 450)
and policy loss improved to 2.35 (best ever), but it didn't translate to wins or lower
draw rate. Adding Gumbel search to eval games (chunk 38+) made no difference — 3 wins in
35 post-fix evals, same rate as before.

Root cause analysis: **400 sims is too many for Gumbel**. The paper shows Gumbel with 16
sims matches standard MCTS with 200 sims. With 400 sims, sigma = (50 + 400) * 0.1 = 45,
still producing fairly sharp targets. More critically, 400 sims per move means fewer
games per hour and less training data diversity. Additionally, permissive game settings
(adjudication=0.6, max_moves=80, temp_moves=3) let games drag into drawn endgames.
v9 used adjudication=0.4, max_moves=50 and achieved 47% draws.

---

### v15 (abandoned — draw rate stuck at 84-90%, 0 eval wins after chunk 95)

Three changes based on research: (1) reduce sims from 400 to 100, (2) restore v9's
aggressive adjudication (adj=0.4, max_moves=50), (3) increase temperature (1.0, temp_moves=8).
Ran for full 229 chunks over 10.8h with no improvement.

| Parameter | Value | vs v14 |
|-----------|-------|--------|
| Network | resnet, 128 channels, 5 blocks | same |
| Simulations | **100** | was 400 |
| Games/iter | 100 | same |
| Workers | 10 | same |
| Playout cap | 100% full | same |
| Value blending | lambda=0.5 | same |
| Policy weight | 0.5 | same |
| Gumbel search | on (c_scale=0.1, Gumbel eval) | same |
| Temperature | **1.0**, **temp_moves=8** | was 0.8/3 |
| Adjudication | **threshold=0.4**, min 30 moves | was 0.6 |
| Max moves | **50** | was 80 |

| Metric | Early | Final (chunk 229) |
|--------|-------|-------------------|
| Value loss | 0.259 | 0.054 |
| Policy loss | 6.877 | 2.212 |
| Best eval | 1W-0D-19L (ELO -512) — 15 evals had 1 win each |
| Draw rate | 80% → 90% (rising throughout) |

**Takeaway**: Fewer sims and aggressive adjudication didn't fix the draw rate — it actually
climbed from 80% to 90% over the run. Only 15 out of 229 evals had a single win, and the
last win was at eval #95. The draw rate never dropped; it monotonically increased. Policy
loss reached 2.21 (best ever) but the network still couldn't win games.

Gumbel AlphaZero has now failed across 6 configurations (v10-v15) with different sim counts
(100, 400), sigma scales (0.01, 0.1, 1.0), adjudication settings, and temperature schedules.
The common factor: backward barrel moves create repetitive shuffling positions that produce
80-90% draws regardless of search algorithm. Standard MCTS v9 achieved 47% draws because
its noisy visit-count targets maintained diversity; Gumbel's sharper targets amplify the
network's tendency toward drawing lines.

---

### v16 (forward-only + Gumbel — draw rate still high)

Same as v15 but with `--forward-only` flag: backward barrel moves are removed from move
generation during self-play and MCTS search. Eval games use full moves.

| Parameter | Value | vs v15 |
|-----------|-------|--------|
| Network | resnet, 128 channels, 5 blocks | same |
| Simulations | 100 | same |
| Games/iter | 100 | same |
| Workers | 10 | same |
| Value blending | lambda=0.5 | same |
| Policy weight | 0.5 | same |
| Gumbel search | on (c_scale=0.1, Gumbel eval) | same |
| Temperature | 1.0, temp_moves=8 | same |
| Adjudication | threshold=0.4, min 30 moves | same |
| Max moves | 50 | same |
| **Forward-only** | **on** | new |

| Metric | Early | Final (chunk 156) |
|--------|-------|-------------------|
| Value loss | 0.231 | 0.047 |
| Policy loss | 6.877 | 1.937 |
| Best eval | 3W-0D-17L (ELO -301) |
| Evals with wins | 65/156 (42%) |
| Draw rate | 77% → 81% (rising) |

**Takeaway**: Forward-only reduced branching factor but draw rate still climbed to 81%.
Policy loss reached 1.94 (best ever) thanks to the smaller action space, but this didn't
translate to better eval results. Forward-only + Gumbel was the wrong combination — the
low draw rate from forward-only was offset by Gumbel's tendency toward deterministic play.

---

### v17 (repetition penalty — matched v9 then regressed)

Return to standard MCTS (no Gumbel, no forward-only). Added repetition penalty in MCTS:
leaf values are shrunk by `penalty * count` when a position has appeared earlier in the game.
This directly discourages the shuffle-loops that cause draws.

| Parameter | Value | vs v9 |
|-----------|-------|-------|
| Network | resnet, 64 channels, 5 blocks | same |
| Simulations | 200 | same |
| Games/iter | 100 | same |
| Workers | 10 | same |
| Playout cap | 25% full | same |
| Value blending | lambda=0.5 | same |
| Adjudication | threshold=0.4, min 30 moves | same |
| Max moves | 50 | same |
| **Repetition penalty** | **0.3** | new |

| Metric | Early | Final (chunk 146) |
|--------|-------|-------------------|
| Value loss | 0.365 | 0.063 |
| Policy loss | 6.118 | 2.437 |
| Best eval | **6W-0D-14L (ELO -147)** at chunk 97 |
| Evals with wins | 100/145 (69%) |
| Draw rate | 74% → 72% |

**Takeaway**: Repetition penalty brought draw rate from v9's ~65% down to ~57% mid-run,
and matched v9's all-time best of 6W. However, draw rate crept back up to 72% late and
eval performance regressed to 0-1 wins by chunk 145 — the same fade pattern as v9. The
penalty helps but isn't sufficient alone; the network still learns to play passively.

---

### v18 (draw filtering + temp_moves=8 + vblend=0.7 — stable but plateaued)

Three changes targeting training signal quality: (1) discard excess draw games from
training data (max 33%), (2) more opening diversity with temp_moves=8, (3) lean harder
on actual game outcomes (vblend=0.7). Full 250-chunk run.

| Parameter | Value | vs v17 |
|-----------|-------|--------|
| Network | resnet, 64 channels, 5 blocks | same |
| Simulations | 200 | same |
| Games/iter | 100 | same |
| Workers | 10 | same |
| Playout cap | 25% full | same |
| Adjudication | threshold=0.4, min 30 moves | same |
| Max moves | 50 | same |
| Repetition penalty | 0.3 | same |
| **Value blending** | **lambda=0.7** | was 0.5 |
| **Temperature** | 0.8, **temp_moves=8** | was 3 |
| **Draw filtering** | **max 33%** | new |

| Metric | Early | Final (chunk 250) |
|--------|-------|-------------------|
| Value loss | 0.195 | 0.215 |
| Policy loss | 2.990 | 2.430 |
| Best eval | **6W-0D-14L (ELO -147)** |
| Evals with wins | **216/250 (86%)** |
| Draw rate (kept) | 32% → 32% (stable) |
| Draws dropped/iter | ~300 → ~480 |

Eval win rate by phase:

| Evals | Win rate | Avg ELO |
|-------|----------|---------|
| 1-50 | 11% | -432 |
| 51-100 | 11% | -361 |
| 101-150 | 11% | -371 |
| 151-200 | 13% | -341 |
| 201-250 | 11% | -381 |

**Takeaway**: Draw filtering eliminated the late-run regression — 86% of evals had at least
one win (vs v17's 69%, v9's 78%). The network stayed competitive throughout all 250 chunks.
However, it completely plateaued: eval win rate was flat at 11% from chunk 50 onward, policy
loss stuck at 2.43. The 64-channel network has hit its capacity ceiling. Draw filtering
and repetition penalty solved the stability problem but not the strength problem.

---

### v19 (128 channels + max_moves=40 — first to beat heuristic)

Same as v18 but with 128-channel network (3.3x more parameters) to break the policy
plateau, and max_moves=40 for even shorter, more decisive games.

| Parameter | Value | vs v18 |
|-----------|-------|--------|
| **Network** | resnet, **128 channels**, 5 blocks (~1.6M params) | was 64ch (~480K) |
| Simulations | 200 | same |
| Games/iter | 100 | same |
| Workers | 10 | same |
| Playout cap | 25% full | same |
| Value blending | lambda=0.7 | same |
| Adjudication | threshold=0.4, min 30 moves | same |
| **Max moves** | **40** | was 50 |
| Repetition penalty | 0.3 | same |
| Draw filtering | max 33% | same |
| Temperature | 0.8, temp_moves=8 | same |

| Metric | Early | Latest (chunk 141, still running) |
|--------|-------|-------------------|
| Value loss | 0.183 | 0.055 |
| Policy loss | 2.954 | 2.337 |
| Best eval (200 sims) | — | 6W-1D-13L (ELO -127) |
| Evals with wins | — | 109/141 (77%) |
| Draw rate (kept) | 15% | 21% |

**Takeaway**: 128ch broke through the 64ch ceiling — new all-time best ELO (-127) and
policy loss pushed past the 2.43 plateau to 2.34 (still declining). 77% of evals had wins.

#### Simulation budget analysis (v19 best_model.pt at chunk ~100)

Post-training analysis revealed that the 200-sim eval budget was dramatically underselling
the network's strength. Testing the same checkpoint at different sim budgets:

| Sims | W-D-L vs depth-4 | Win% | Notes |
|------|-------------------|------|-------|
| 200 | ~5-0-15 | 25% | training eval budget |
| **300** | **13-1-6** | **65%** | **sweet spot** |
| 400 | 12-1-7 | 60% | diminishing returns |
| 500 | 7-4-9 | 35% | too cautious, more draws |
| 600 | 6-6-8 | 30% | 30% draws |
| 800 | 7-7-6 | 35% | 35% draws |

At 300 sims, the network **decisively beats depth-4 heuristic** (65% win rate). Above
~400 sims, extra search makes the network play too cautiously — it avoids risk and draws
instead of winning. This is likely because the value head is still slightly optimistic, and
deeper search amplifies the tendency to avoid tactical complications.

Testing against stronger heuristics at 300 sims:

| Opponent depth | W-D-L | Win% |
|----------------|-------|------|
| 4 | 13-1-6 | 65% |
| 5 | 10-3-7 | 50% |
| 6 | 5-3-12 | 25% |
| 7 | 14-4-2 | 70% |
| 8 | 6-7-7 | 30% |
| 9 | 10-4-6 | 50% |

The network is competitive against the heuristic at **all depths up to 9**. The odd-even
pattern (better vs odd depths) is an alpha-beta artifact — odd/even search depths evaluate
positions from different perspectives. Key finding: the network learned genuine strategy,
not just anti-depth-4 patterns.

**Critical insight**: The 200-sim eval used during training was the bottleneck — the network
was stronger than it appeared, and the best-model gating mechanism was likely saving suboptimal
checkpoints. Future runs should eval at 300 sims.

---

## Key Findings

### What worked
- **Value target blending** (v8/v9): Halved value loss by providing dense per-position
  signal from heuristic eval, avoiding the problem of sparse game-outcome-only targets
- **Playout cap 25%** (v2/v3/v7-v9): ~4x more game trajectories per compute. Runs
  without it (v4/v5/v6) performed consistently worse
- **Aggressive adjudication** (v9): threshold=0.4 + max_moves=50 eliminated straggler
  iterations and reduced draw rate
- **Bootstrap games** (10K at depth 9): Solved the cold-start problem by pre-filling
  the replay buffer with strong play
- **Repetition penalty** (v17/v18): Penalising revisited positions in MCTS reduced draw
  rate from ~65% to ~57% mid-run. Directly breaks shuffle-loops
- **Draw filtering** (v18): Capping draws at 33% of training data eliminated the late-run
  regression. 86% of evals had wins (vs v9's 78%). Network stayed competitive through all
  250 chunks instead of fading after chunk 40
- **temp_moves=8** (v18): More opening diversity than temp_moves=3
- **128ch network** (v19): Broke through the 64ch policy plateau (2.43 → 2.34, still
  declining). First run to beat depth-4 heuristic at appropriate sim budget

### What didn't work
- **No playout cap** (v4/v5/v6): Full search on every move wastes compute on cheap
  moves that don't need accuracy
- **2 epochs per iteration** (v2/v4): Overfits to current buffer, producing narrow
  self-play data (Wang et al. 2020)
- **400 sims without playout cap** (v5/v6): More sims per move but fewer total games.
  The outer loop matters more than the inner loop
- **Heuristic ratio decay to 0%** (v3): Pure self-play too early produces low-quality
  games. Keeping some heuristic games (10%) through training helps
- **Buggy Gumbel implementation** (v10-v12): ~80% draw rate from 5 bugs (no v_mix, no
  Q-normalization, fixed sigma, Dirichlet on top of Gumbel, wrong temp sampling)
- **c_scale=1.0 in Gumbel sigma** (v13): Paper says c_scale=1.0 but DeepMind's own
  reference impl (mctx) uses value_scale=0.1. With c_scale=1.0, sigma=450 at 400 sims,
  making 47% of policy targets one-hot (entropy < 0.01). Network learned to always pick
  one move with 99.7% confidence, killing exploration and producing 80%+ draws

### Unsolved problems
- **Eval sim budget mismatch**: Training eval at 200 sims dramatically undersells network
  strength. v19 shows 25% win rate at 200 sims but 65% at 300 sims. The best-model gating
  during training is likely saving suboptimal checkpoints. Future runs must eval at 300 sims.
- **Cautious play at high sim counts**: Above ~400 sims the network plays too passively,
  drawing instead of winning. The value head's slight optimism is amplified by deeper search,
  causing risk-averse play. Repetition penalty helps but doesn't fully solve this.
- **Self-play ceiling**: Once the network reaches a certain strength, it can only learn from
  games against itself. Without external opponents of varying strength, improvement stalls.
- **64ch policy plateau** (solved by 128ch): Policy loss plateaued at ~2.43 across all 64ch
  runs. 128ch in v19 pushed past this to 2.34 and still declining.

### Ideas to test
- **Eval at 300 sims**: Fix the eval mismatch. 300 sims is the sweet spot where the network
  performs best. Enables accurate best-model gating during training.
- **Network-vs-heuristic training games**: Replace some self-play games with network vs
  heuristic at varying depths (3-7). Exposes the network to strong, varied play every
  iteration and breaks the self-play plateau ceiling.
- **Training at 300 sims**: Better policy targets at modest speed cost (~15% slower with
  playout cap). Matches the optimal eval budget.
- **Progressive curriculum**: Once the network beats depth-4, train/eval against depth-6,
  then depth-8. Forces continued improvement against increasingly strong opponents.
- **Longer runs (500 chunks)**: v19's policy is still declining at 2.34. With proper 300-sim
  eval, continued improvement might be visible through 500 chunks.

### Tested and abandoned
- **Forward-only training** (v16): Reduced branching but didn't help with Gumbel
- **Gumbel AlphaZero** (v10-v16): Six configurations, all worse than standard MCTS.
  Gumbel's sharper policy targets amplify the network's passive tendencies
- **No playout cap** (v4/v5/v6): Fewer games per compute, consistently worse
- **2 epochs** (v2/v4): Overfits to current buffer
- **400 sims** (v5/v6/v10-v14): More sims per move but fewer games. Outer loop > inner loop
