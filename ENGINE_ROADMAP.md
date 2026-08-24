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
     random-moves 6, 10 workers (~3.8 games/s ⇒ ~11h; checkpoints every
     2000 games, auto-resumes). Expect ~5M augmented positions, 2-label
     format (search score + outcome).
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

Backlog (bigger builds): improving flag (needs static-eval stack),
2-ply continuation history + history pruning, singular extensions +
multicut (TT-move-based — no captures needed, suits forced race lines),
more aggressive NMP (game is near-zugzwang-free), TT static-eval storage,
time management (bestmove stability) when we play timed matches.
Not applicable (capture-dependent): SEE, ProbCut, MVV-LVA, capture history.
Simplification candidates to test: razoring; killers once conthist is 2-ply.

## Facts to remember

- Depth-4 heuristic (the old AlphaZero eval opponent) is ~−300 Elo below the
  depth-10 heuristic. "Beats depth-4" was never a high bar.
- Never trust a result without a CI from match.py. 200+ games minimum.
