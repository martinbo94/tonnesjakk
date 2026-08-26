# Engine Roadmap & Measurement Log

Strategy adopted 2026-08-20 after reviewing the AlphaZero (23 runs) and NNUE
efforts: **make the classical engine sound and measurable first, then attach a
properly trained NNUE. No MCTS at play time.**

## Diagnosis (why previous efforts stalled)

1. **The game had no draw rules in the engine.** Barrels move backward, so
   games are infinite by rule; every "draw" in earlier data was a max-move-cap
   artifact. Self-play collapsed into shuffle loops (80–90% "draws"), and
   search could neither avoid nor exploit repetitions.
2. **20-game evals are ±150–300 Elo of noise.** Many run-to-run "conclusions"
   in ALPHAZERO_RUNS.md were sampling noise (v20–v23 evals at 300 sims bounce
   0–4 wins with identical configs).
3. **Wrong tools for the game's shape.** Tønnesjakk is a short, sharp race
   (measured: ~27 plies/game, 98–99% decisive under real rules). Deep
   alpha-beta eats shallow MCTS here. NNUE was trained on ~1M narrow
   positions and tested at equal *depth* (needs equal *time*).

## Rules now defined (implemented in core, 2026-08-20)

- **Threefold repetition = draw.** Search scores any repetition of a
  search-path or game-history position as a draw (checked before TT probe).
- **No-progress rule: 60 plies** without an irreversible event (barrel
  placement, pail placement, scoring) = draw. `halfmove_clock` on the board.
- **Zobrist fix:** off-board barrel counts are hashed (they weren't — a
  scored barrel vs an in-hand barrel previously hashed identically).
- Engine API: `set_game_history(hashes)`, `contempt` (cp, root-side draw
  aversion), `no_progress_limit`. `full_reset()` now truly wipes the TT
  (deterministic games; the old O(1) clear leaked entries across games).

## Measurement infrastructure

- `scripts/match.py` — the gate for every change. Paired random openings
  (each played with both colors), real draw rules, fixed time or depth,
  parallel workers, Elo ± 95% CI over pair scores. Engines are deterministic:
  a self-match scores exactly 50% with CI ±0.
- `scripts/spsa_tune.py` — SPSA tuning of all eval weights via fast matches
  with common-random-number openings.

## Measured facts (2026-08-20, post-draw-rules)

| Match | Result | Elo |
|---|---|---|
| d4 vs d6 | 36.2% (400 games) | −98 [−128, −70] |
| d6 vs d8 | 35.0% (300) | −108 [−136, −81] |
| d8 vs d10 | 38.8% (200) | −80 [−113, −48] |
| 50ms vs 100ms | 40.5% (300) | −67 [−94, −40] |
| 100ms vs 200ms | 40.0% (300) | −70 [−97, −45] |

- **~40–55 Elo per ply of depth; ~70 Elo per time doubling. Not saturating
  through depth 10** — search keeps paying, eval quality is the multiplier.
- **Games are 98–99% decisive** with correct rules (draws were an artifact).
- **First-move advantage: White scores 57%** (~+50 Elo). Paired openings
  neutralize this in measurement.
- Avg game length ~27 plies: it's a race, tempo matters.

## Phase 2 results (2026-08-20)

- **SPSA (250 iters × 24 pairs @ d5) validated: +24 Elo @ d5 [+4,+45],
  +25 @ d7 [+2,+49], +9 @ d9 [−21,+39]** vs old defaults. Tuned weights are
  now the engine defaults (`BitBoardEngine::new`). The optimizer moved
  weights only slightly — the hand-tuned values were already near a local
  optimum for this feature set. Bigger eval gains must come from NNUE.
- **A/B: `weight_tempo=20` → −12 Elo [−38,+13]** — no value, left at 0.
- **A/B: `weight_pail_in_hand=30` → literally zero effect** (every game
  identical), which exposed a rules bug: `generate_moves` forced pail
  placement as the very first sub-move of the game.

## Rules correction (2026-08-24) — deferred pail placement

Confirmed by the rules-owner: **the pail may be placed once per game, on any
of your turns, as an optional sub-move BEFORE the barrel move.** Implemented
in both move generators (BitBoard + Board). Search adjustments: NMP/razoring/
futility now only disabled at `awaiting_barrel` (mid-turn) nodes; pail
placements are never futility-pruned and excluded from quiescence. Web AI
searches the whole turn (pail sub-move + barrel) instead of random-placing;
AlphaZero paths keep the old random turn-1 pail (their models assume it).

⚠️ Everything measured before this date was the forced-pail VARIANT: the
depth curve, SPSA tuning, and `training_gen1_d8_FORCEDPAIL_VARIANT.bin`
(12k games, quarantined — do not train on it).

Match-harness openings are now BARREL-ONLY randomization (match.py +
nnue.py): a random pail burn in the opening would erase the strategic
dimension the rules fix unlocked.

### Re-baseline under real rules (2026-08-24, barrel-only openings)

| Match | Result | Elo |
|---|---|---|
| d4 vs d6 | 29.3% (300) | −153 [−190, −119] |
| d6 vs d8 | 33.3% (300) | −120 [−158, −85] |
| pail_in_hand=30 A/B @ d5 | 50.6% (400) | +4 [−14, +23] |
| pail_in_hand=80 A/B @ d5 | 50.1% (400) | +1 [−24, +26] |

- **Depth is worth MORE under real rules** (d4v6 −98 → −153): pail timing
  is a deep resource that deeper search exploits.
- **`weight_pail_in_hand` adds nothing** (kept at 0): search itself prices
  the option; a static scalar doesn't help at these depths.
- Draws now 3–5% (threefold), up from ~1%; games still ~28 plies.
- **Engines DO exercise the pail option**: at d6 with 6-ply openings,
  ~40% place on their first turn, the rest delay 1–4 moves (median
  move 7, max 10, all 48 pails placed before game end).

## Plan

1. ~~Draw rules + measurement harness~~ ✅
2. ~~SPSA-tune eval weights, validate at d5/d7/d9~~ ✅ (+24/+25 Elo, applied)
3. **NNUE, the proven recipe** (all local, M4 Pro):
   - ➤ RUNNING: `training_gen1_d8.bin` — 150k games @ d8, tuned engine,
     random-moves 6, 4 nice'd workers, checkpoints every 1000 games,
     auto-resumes. First 25k games were generated on the pre-qs2 binary at
     3.1 games/s; after the quiescence fix the SAME 4 workers run at
     **~46 games/s** (~15x) — the whole run now takes ~1h. 2-label format
     (search score + outcome). Data from both engine versions is mixed
     (labels are d8 search scores either way).
   - Train value-only net (PyTorch, MPS), export, load in Rust.
   - Gate at **equal time** via match.py. Iterate generations: once
     NNUE-engine > heuristic-engine, regenerate data with it.
4. Later: endgame tablebases keyed on (white_scored, black_scored) phases;
   opening book; contempt tuning for match play vs humans.

## Research survey (2026-08-24) — validation + technique imports

Two literature sweeps (held-resource option value; race-game engine
architectures). Full agent reports in the session log; distilled:

**Our architecture bet is what the evidence says wins.**
- Vanilla MCTS/AlphaZero demonstrably fails in no-capture race games:
  tabula-rasa AlphaZero for Chinese checkers scored 0/100 (Liu et al. 2019,
  arXiv:1903.01747) — random playouts never terminate; MCTS "shallow trap"
  literature (Ramanujan/Sabharwal/Selman) predicts averaging backups fail in
  trap-dense 28-ply races. Our 23 failed runs were the expected outcome.
- Breakthrough (best-studied race game): alpha-beta+eval beat even enhanced
  MCTS at ≤1s/move (Lanctot et al. 2014); the current Computer-Olympiad
  champion (Athénan) is minimax + learned value net trained ~48h on modest
  hardware — no MCTS, no policy net.
- **Fairy-Stockfish measured NNUE at +355 Elo on 6x6 Breakthrough and +374
  on Racing Kings** with hobbyist-scale training — our exact game class and
  board size. NNUE lane is validated; expect saturation somewhere in the
  20–100M position range.
- Chinese checkers is strongly solved through 6x6/6 pieces (Sturtevant
  2019): **every solved size is a first-player win by exactly 2 moves**, and
  the solve REQUIRED repetition + camping rules (cf. Dodgem: 4x4/5x5 never
  end under perfect play without them). Our Phase-0 draw rules are
  load-bearing. Tønnesjakk's state space is far smaller than the solved CC
  board → **solving it on the M4 Pro is plausibly feasible** (retrograde,
  2 bits/state, phase DAG on scored counts).

**Pail-in-hand (option value) findings.**
- Quoridor literature (walls-in-hand) matches our +0 Elo A/B exactly:
  path-difference eval dominates; walls-remaining weight ~10x smaller and
  unstable (Glendenning 2002). Rule of thumb across backgammon cube /
  Scrabble leaves / chess threats: **an explicit in-hand eval term pays only
  when the option's exercise point lies beyond the search horizon** — ours
  doesn't at d5–8.
- For the NNUE: held resources belong in the FEATURES (shogi hand pieces in
  HalfKP, CrazyAra pocket planes) so the net learns the *conditional*
  premium. Our HalfPail design already encodes own pail (bucket incl.
  "not placed") and enemy pail (feature type) per perspective — correct.
- Quoridor's key optimization transfers: **prune pail placements to squares
  that change a race/jump distance** (Quoridor: ≥1 shortest-path change),
  instead of all 36 empties.

**Imported technique queue (ranked, from strongest evidence):**
1. **Single-agent race distance table**: exact min-plies for one side to
   score all remaining barrels ignoring the opponent (C(36,≤4) configs per
   scored-count — tiny). Use the two-sided difference as the core eval term
   (Roschke & Sturtevant: beat everything in CC) and as **exact win
   adjudication once armies disentangle** (side to move wins iff its count ≤
   opponent's) — search cutoff + perfect endgame labels for NNUE.
2. **Decisive-move handling + race quiescence**: detect immediate-win and
   must-block moves in movegen/ordering; quiesce on runners within 2–3
   tempi of scoring (no captures to quiesce on).
3. **NNUE output buckets keyed on (my_scored, opp_scored)** — the kingless
   analog of king buckets; eval semantics shift sharply as barrels leave.
   Consider the `bullet` trainer (Rust, ships an Ataxx example) as an
   alternative to the PyTorch pipeline.
4. **Forward-biased ordering + straggler tie-break** (prefer advancing the
   hindmost barrel among near-equal moves); forward-only filtering proven in
   CC playouts/tree (~2x branching cut). NB: forward-ONLY was already tried
   in AZ v16 — the win is in *ordering/reductions* for alpha-beta, not rule
   changes.
5. **Pail-placement pruning** (Quoridor-style relevance filter) + optional
   Janowski-style functional option value for the classical eval:
   eval += x · (best_pail_placement_eval − eval), x≈0.6–0.7, leaves only.

## Technique A/B results (2026-08-24, real rules, gated via match.py)

| Change | Result | Verdict |
|---|---|---|
| `weight_race=80` (single-agent distance diff) | **+36 @ d5 [+6,+67], +34 @ 50ms [+2,+66], +63 @ d7 [+24,+104]**; 40 → +13(ns), 120 → +5(ns) | ✅ **default = 80**; scales UP with depth |
| Win-distance scoring (WIN−ply + TT mate-score conversion) | correctness fix (engine previously couldn't prefer faster wins; risked shuffling won positions into repetition draws) | ✅ applied (untested-by-match; standard) |
| `pail_filter` (placements within 2 of a barrel) | −2 [−32,+27] @ 50ms | ❌ off — no speed win measured |
| `weight_straggler=6` (ordering) | −20 [−47,+8] @ 50ms | ❌ off |

The race table is the biggest single eval gain so far and grows with depth —
consistent with it being *exact* long-horizon information the ~28-ply search
can't derive itself.

## Testing methodology (adopted 2026-08-24, from Stockfish/fishtest research)

**The rule: games under real time control are the only acceptance criterion
for search changes.** Modern pruning is deliberately unsound — nobody proves
a pruning rule "safe"; SPRT games measure whether the pruned leaves mattered
on net. nodes/sec and depth-reached are diagnostics only (a pruning patch
trivially inflates both while possibly discarding decisive lines).

- `match.py --sprt ELO0 ELO1` implements a GSPRT stop rule over PAIR scores
  (pentanomial-style variance, α=β=0.05, LLR ±2.94). Bounds while gains are
  large: **gainer [0, 10]**, tightening to [0, 5] later; **simplification
  [-5, 0]**. Our ~0% draw rate maximizes per-game variance (σ²≈0.25), so we
  need ~2x the games per Elo that chess does — but real gains are big now.
- **Pruning/reduction/margin patches must ALSO pass at a slower TC**
  (Stockfish lesson: 1-in-40 LTC survival; pruning patches flip sign with
  depth). Screening TC = 50ms, confirm TC = 200ms+.
- Fixed-depth matches only for eval-only changes; fixed-time otherwise.
- Deterministic engines: never replay an opening pair — every pair must use
  a fresh seed (match.py already does this).
- Max ~4 retries per idea, then archive. Every ~10 merged changes, run a
  fixed regression match vs a frozen older build; keep a ladder of frozen
  versions to detect sibling-exploitation (discount self-play Elo ~30%).
- **NNUE promotion rule (2026-08-26): a new net must beat the heuristic AND
  every frozen rung in `models/ladder/` at equal time** (`scripts/ladder.py
  --candidate …`), not just its parent. Strength is not transitive; a net
  trained on its parent's games can exploit the parent specifically.
  Periodic `--round-robin` over the rungs checks the ladder stays monotone.
- Debug tool: a build with all pruning disabled must reproduce plain
  alpha-beta results at equal depth (validates implementation; SPRT
  validates the policy).

## Search audit vs modern engines (2026-08-24)

Already correct/modern: PVS+ID+TT(3-entry clusters, depth+age), NMP with
eval/depth-scaled R, futility (alpha side), razoring, IIR (not IID — right
choice), killer+butterfly+1-ply continuation history, LMR (ln·ln + history
modulation), correction history (2024-era!), win-distance scoring + TT
mate-score conversion (added today).

Missing / added behind flags — SPRT gate results (2026-08-24):
- `asp_mode=1` (one-sided geometric aspiration widening): **PASSED both
  gates — +42 [+20,+64] @ 50ms, +31 [+13,+50] @ 200ms → DEFAULT.**
- `lmp_base=6` (late move pruning): **DEFAULT.** SPRT PASS @ 50ms
  (+27 [+10,+45]); at 200ms accepted on CI rather than SPRT bound:
  +14 [+3,+26] over 1600 games (two independent LTC runs +16/+14 — the
  true value sits mid-band of SPRT(0,10), so the LLR couldn't resolve,
  but the CI excludes zero at both time controls).
- `rfp_margin=120` (reverse futility): inconclusive (+2, 1000 games @
  50ms). Retry other margins later (≤4 attempts, then archive).
- `keep_killers=1`: null (+1 in 2000 games) — archived. Plausibly because
  aspiration re-searches within one iteration already reuse killers where
  it matters.

**Quiescence was the hidden cost centre (found 2026-08-25).** Profiling a
d8 midgame search: ~300 main nodes vs ~118,000 quiescence nodes (360x).
The legacy filter (any move landing within 2 rows of goal, 8 plies deep,
unordered) is a full-width extension once barrels advance — 99.7% of all
search effort, while raw speed is fine (~23M nodes/s). `qs_mode` knob:
1 = scoring + immediate-threat moves, cap 6 (6x faster at d8);
2 = scoring moves only, cap 4 (36x faster). Scoring moves ordered first.
Gates @ 50ms: **qs1 +31 [+13,+50] PASS; qs2 +53 [+28,+78] PASS.**
Slow-TC gate: **qs2 +62 [+35,+91] @ 200ms PASS**; head-to-head qs2 vs
qs1 +29 [+11,+47] PASS. → **`qs_mode=2` is the default.** Biggest single
search gain of the project. Lesson: measure WHERE nodes go before touching
number formats/SIMD.

## NNUE architecture tournament infrastructure (2026-08-25)

- Rust `SparseNNUE` (src/nnue.rs): ONE evaluator generic over a
  `NnueConfig` — feature set (`halfpail` 3996 / `plain` 144), `mirror_black`
  (black perspective sees a flipped board so shared weights are orientation-
  consistent — the legacy net did NOT do this), optional 20 dense features,
  `output_buckets` 1 or 25 keyed on (white_scored, black_scored).
  Incremental updates are a generic before/after feature-set diff (no
  per-move special cases); a perspective is recomputed only when its bucket
  changed. Loads v2 JSON and legacy HalfPail JSON. Tests: incremental ==
  scratch along random games for all 8 configs; mirror symmetry; dense-row
  round trip.
- Python `nnue_arch.py`: `NnueArch` + `SparseNNUE` model (bucketed heads via
  wide linear + gather), Rust batch decoder `decode_sparse_batch` (indices
  from the SAME Rust feature code the engine uses), pre-decode-to-device
  trainer (MPS/CUDA), `export_sparse_json`, parity helper.
  CLI: `python -m tonnesjakk.nnue --load-data X.bin --feature-set plain
  --mirror --output-buckets 25 --no-dense --arch 256 32 --output runs/<tag>`.
- **Verified**: 3 architectures trained on 200k real positions (~2–3 s per
  2 epochs on MPS), exported, loaded in Rust: **0 cp Python↔Rust difference**
  on 400 positions; search runs with NNUE loaded.
- Tournament plan (equal-TIME gates via match.py `--nnue-a/--nnue-b`, plus
  each vs heuristic): halfpail vs plain; ±mirror; ±dense; buckets 1 vs 25;
  width 128/256/512; λ ∈ {0.2, 0.5, 0.8}. Run once gen-1 data is complete.

Backlog (bigger builds): improving flag (needs static-eval stack),
2-ply continuation history + history pruning, singular extensions +
multicut (TT-move-based — no captures needed, suits forced race lines),
more aggressive NMP (game is near-zugzwang-free), TT static-eval storage,
time management (bestmove stability) when we play timed matches.
Not applicable (capture-dependent): SEE, ProbCut, MVV-LVA, capture history.
Simplification candidates to test: razoring; killers once conthist is 2-ply.

## THE NNUE ROOT CAUSE — inverted labels (found 2026-08-25)

Gen-1 diagnostics before any architecture work: stored search-score labels
had **r = −0.002 with game outcome** and r = −0.025 with the score
differential — impossible for real d8 scores; their only correlate was side
to move (r = +0.21). Cause: `play_game` negated the engine's score for
black-to-move positions "to convert to White's perspective", but the
minimax score is ALREADY White-perspective. Half of all labels were sign-
inverted. **This bug predates this work and plausibly explains every
earlier NNUE failure** (the old "val loss floor 0.5867", −115..−230 Elo).
Fix: generator corrected; gen-1 repaired in place (3.1M labels un-negated;
`*_y.bin.preflip_backup` kept). After repair: r = +0.75 with outcome; a
linear model on the 20 dense features gets R² = 0.45 (was 0.06).

Two more fixes on the way to the first result:
- `dedupe_rows` permuted labels by the inverse of the row permutation
  (val loss pinned at ln 2 = 0.693 for every candidate — the tell).
- Rust evaluator returned tanh·1000 instead of 600·atanh (labels are
  tanh(cp/600)), a compressed scale that broke every search margin.

**First NNUE win:** `halfpail_m_d20_256x32` (mirror, dense), 60 epochs
(39 s on MPS) on deduped gen-1 (1.61M unique positions, 74% dups removed):
val 0.5453 (entropy floor ≈ 0.54) → **+165 Elo [+107, +233] vs heuristic
at equal 100 ms** (84-5-31), despite reaching depth 8 vs 9 (NNUE 1.3 Mnps vs
3.6). Plain features without dense/side-to-move: val 0.6287, −585 Elo —
architecture matters enormously here.

### Architecture tournament, round 1 (gen-1 repaired, 150 epochs, 400 games @ 100ms vs heuristic)

| # | architecture | Elo vs heuristic | val loss |
|---|---|---|---|
| 1 | **plain + mirror + dense 20, 128×32** | **+203 [+168,+241]** | 0.5367 |
| 2 | halfpail + mirror + dense, 128×32 | +177 [+146,+211] | 0.5458 |
| 3 | halfpail + dense, 128×32 (legacy incumbent) | +171 [+141,+203] | 0.5480 |
| 4 | plain + mirror, no dense, 256×32, 25 buckets | +163 [+132,+197] | 0.5580 |
| 5 | halfpail + mirror + dense, 256×32, 25 buckets | +154 [+125,+186] | 0.5446 |
| 6 | plain + mirror, no dense, 512×32, 25 buckets | +118 [+88,+150] | 0.5584 |
| 7–13 | plain, no dense, no buckets (any width/λ/loss/dedupe) | −436 … −619 | 0.58–0.64 |

Readings: the net MUST see game phase + side to move (dense features or
scored-count buckets) — without either it is hopeless; with dense features
the simple plain piece-square encoding beats HalfPail buckets (3996-feature
embedding is too data-hungry for 1.6M unique positions); bigger nets are
worse at this data size (128 > 256 > 512) ⇒ data-limited ⇒ gen-1b (diverse
data) is the next lever. λ / loss / dedupe ran on the failing no-dense
family (confounded) → round 2 re-tests them on the winner.

### Round 2 (variants of the winner; same gate)

| architecture | Elo vs heuristic | val loss |
|---|---|---|
| plain+m+d20 128×32 **λ=0.5** | +207 [+176,+243] | 0.4874 |
| … MSE loss | +185 [+153,+221] | (mse) |
| … 25 output buckets | +183 [+153,+215] | 0.5386 |
| … no dedupe | +181 [+149,+216] | 0.5588 |
| … 64×32 | +181 [+150,+214] | 0.5384 |
| … λ=1.0 | +179 [+149,+213] | 0.5641 |
| … (winner re-run, identical config) | +176 [+144,+211] | 0.5370 |
| … 192×32 | +173 [+142,+206] | 0.5372 |
| … λ=0.65 | +171 [+141,+203] | 0.5136 |
| … 128×64 | +171 [+139,+205] | 0.5368 |

**Everything within +171..+207 with overlapping CIs.** The identical-config
re-run (+176 vs +203 in round 1) puts the run-to-run noise floor at ~30 Elo,
so no round-2 knob is a proven gain; λ=0.5 is suggestive at best. The family
is saturated on 1.6M unique positions ⇒ confirmed data-limited. Nine of nine
400-game gates beat the heuristic decisively — the first NNUE wins in the
project's history. Next levers, in order: gen-1b (diverse data), then
self-generated data (gen-2 labeled by the best net, gated vs heuristic AND
previous net), then width.

### Round 3 (gen-1 + gen-1b = 16.3M rows → 10.5M unique; 100 epochs)

Gen-1b (300k games, random-moves 10, noise 0.15) cut duplicates from 74% to
35% and moved the loss for the first time: net-1 config 0.4874 → **0.4607**.

| architecture | Elo vs heuristic @100ms | val loss |
|---|---|---|
| plain+m+d20 128×32 λ0.5, 100% data | +201 [+170,+236] | 0.4607 |
| … 50% data | +201 [+170,+237] | 0.4648 |
| … 256×32 | +198 [+167,+232] | 0.4603 |
| … λ0.8 control | +192 [+160,+227] | 0.4833 |
| … 25% data | +187 [+157,+221] | 0.4705 |
| halfpail+m+d20 256×32 λ0.5 | +181 [+150,+214] | 0.4679 |
| … 256×64 | +177 [+146,+211] | 0.4608 |
| … 512×32 | **+146** [+116,+178] | 0.4614 |

**Loss and strength have decoupled.** Elo vs heuristic is flat ~+200 across
25%→100% data and 128→256 width while loss improves; 512-wide has the SAME
loss as 128 but −55 Elo — the wider net is slower per node, so at fixed time
it searches less. ⇒ The bottleneck is now the NNUE engine's SEARCH side
(speed; margins/LMP/qsearch tuned for the heuristic; no race term), not the
eval's fit. Levers: SPSA-retune search with the net loaded, eval speed
(quantization now earns its keep), head-to-head net gates (a fixed weaker
opponent has lost resolution), deeper labels / gen-2.
Head-to-head 100%-data vs 25%-data net (same arch/speed): **+16.5 [−7,+40]**
over 400 games — 4x data ≈ nothing at fixed time. Draws 17% in net-vs-net
(vs ~7% vs heuristic): similar evals → balanced games.
NPS: heuristic 3.5M; NNUE 128×32 1.5M, 256 0.95M, 512 0.6M ⇒ depth 15 vs
13/13/12 @100ms. **Round 4 = speed**: smaller nets, dense-feature cost,
then quantization.

### Round 4 (speed; gated HEAD-TO-HEAD vs net-1 = plain_m_d20_128x32_l0.5, 100ms)

| architecture | Elo vs net-1 | val loss |
|---|---|---|
| **plain+m+d20 96×16 λ0.5** | **+36 [+13,+59]** | 0.4615 |
| plain+m+d20 64×16 | +33 [+9,+58] | 0.4633 |
| plain+m, no dense, 64×16, 25 buckets | +10 [−15,+36] | 0.4813 |
| plain+m+d20 128×16 | −3 [−26,+21] | 0.4615 |
| plain+m+d20 64×32 | −3 [−30,+23] | 0.4624 |
| plain+m+d20 32×16 | −10 [−35,+14] | 0.4662 |

Speed hypothesis confirmed: hidden2=16 (half the FC2 cost) beats net-1
despite equal-or-worse loss; 32-wide is too small (eval quality loss wins).
Dense features still pay (64×16 d20 +33 vs d0+buckets +10) despite their
~25% throughput cost. NPS: net-1 1.46M, 96×16 2.18M, 64×16 2.38M.
**Slow-TC gate PASSED: 96×16 vs net-1 @200ms +43 [+21,+65].**

## net-1b = `models/net1b_plain_m_d20_96x16_l05.json` (2026-08-25)

plain piece-square features, mirrored black perspective, 20 dense features,
96×16, single head, λ=0.5, trained 100 epochs on gen-1+gen-1b deduped
(10.5M unique positions). Chain of evidence: ≈+200 Elo vs heuristic @100ms
(family), +225 @200ms (128×32 sibling), +36/+43 vs net-1 @100/200ms. Web
UI's heuristic engine now loads it by default when present.

**Gen-2 running**: 300k games @ d8 labeled BY net-1b (--use-nnue), random-
moves 10, noise 0.15, 10 workers → `training_gen2_d8.bin`. First turn of the
self-improvement loop. Gate for net-2: vs net-1b AND vs heuristic, 100+200ms.
Next speed lever after that: quantized inference (int16 accumulator / int8
FC2) — FC2 and dense features dominate NNUE node cost.

## Engine speed work (2026-08-26, while gen-2 generates)

- **TT stores BitMove** (was the heap-allocating Python `Move` + conversion on
  probe): tree bit-identical (bench signature 88680/123548), ~+10% nps both
  engines.
- **Dense features**: integer sorts, single pass; 0/3000 evals differ.
- **Quantized inference**: i16 accumulators (scale 128), i16 FC2 weights
  (scale 1024, |w|≤3), `i16x16::dot` → i32 (16 MACs/op vs 8 for f32).
  net-1b: mean 3.6 cp / p95 10 cp / max 32 cp vs f32 reference, corr
  0.99996; incremental updates now bit-exact. NNUE nps 2.1–2.3 → 2.68
  (heuristic/NNUE ratio 0.55 → 0.77). hidden1 must be a multiple of 16.
  Sanity gate: quantized net-1b vs heuristic @100ms **+204 [+173,+238]**
  (286-39-75) under heavy contention (10 datagen + 4 match workers) — no
  regression. Precise quantization Elo gain needs an idle-machine A/B
  (expected ~+25 from the speed/Elo relation seen in round 4).

**Slow-TC gate PASSED:** `plain_m_d20_128x32_l0.5` vs heuristic @ 200ms:
**+225 Elo [+187, +268]** (225-21-54, 300 games) — larger than at 100ms
(+207), i.e. the NNUE's edge GROWS with time (eval quality compounds with
depth; the per-node speed deficit matters less). Both gates cleared: this is
**net-1 candidate** pending training on gen-1 + gen-1b.

## Cleanup (2026-08-25)

Removed: A/B-rejected knobs (`weight_pail_in_hand`, `weight_tempo`,
`weight_straggler`, `pail_filter`, `keep_killers`) and the legacy code paths
behind `asp_mode=0` / `qs_mode=0`; the Python-side HalfPail feature functions
and `decode_halfpail*` Rust exports (superseded by `decode_sparse_batch`);
`nnue.py`'s 50-game no-draw-rules `--compare*` tooling and training-history
tracking (measurement belongs to `match.py`); `--halfpail`, `--num-workers`,
`--test-halfpail`; superseded scripts (`depth_tournament`, `tune_heuristic`,
`bench_depth`, `benchmark_depths`, `test_model`, `diagnose_*`,
`test_submove_matches`, one-off suite runners) and `NNUE_NEXT_STEPS.md` /
`nnue_history.json`. Kept: AlphaZero core (documented research line),
`bench_engine.py`, `watch_game.py`, `inspect_model.py`.

## Facts to remember

- Depth-4 heuristic (the old AlphaZero eval opponent) is ~−300 Elo below the
  depth-10 heuristic. "Beats depth-4" was never a high bar.
- Never trust a result without a CI from match.py. 200+ games minimum.
