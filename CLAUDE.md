# Tonnesjakk - NNUE Barrel Chess Engine

## Project Overview

Tonnesjakk is a 6x6 barrel chess game ("tønnesjakk" in Norwegian) with a custom
Rust engine and NNUE neural network evaluation. The game has 4 piece types
(white/black barrels and pails) on a 6x6 board. Goal: push your barrels to the
opponent's back row to score. First to score 4 barrels wins.

## Architecture

- **Rust engine** (`src/lib.rs`, ~4000 lines): Bitboard-based engine with
  alpha-beta search, PVS, iterative deepening, TT, null move pruning, LMR,
  futility pruning, killer moves, history heuristic, quiescence search.
  Exposed to Python via PyO3/maturin.

- **Python NNUE training** (`python/tonnesjakk/nnue.py`): Training pipeline
  for the NNUE network. Generates games via self-play, trains PyTorch model,
  exports weights to JSON for Rust consumption.

- **NNUE in Rust** (`src/lib.rs` lines 2051-2627): Incremental NNUE evaluation
  with SIMD-accelerated f32 operations. AccumulatorStack for make/unmake.
  157 features → 64 → 32 → 1 architecture (144 base + 13 relational).

## Build & Test

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Build Rust extension
maturin develop --release

# Run training with improved pipeline
.venv\Scripts\python.exe -m tonnesjakk.nnue --games 20000 --depth 7 --lambda 0.85 --workers 8 --save-data training_v2.npz

# Compare NNUE vs heuristic (time-based)
.venv\Scripts\python.exe scripts\test_model.py nnue_weights.json --time-ms 200 --games 50

# Compare NNUE vs heuristic (fixed depth)
.venv\Scripts\python.exe -m tonnesjakk.nnue --compare nnue_weights.json heuristic --compare-games 50 --depth 7
```

## Key Engine Functions (src/lib.rs)

| Function | Line | Purpose |
|----------|------|---------|
| `minimax()` | ~3490 | Main alpha-beta with PVS search |
| `quiesce()` | ~3355 | Quiescence search for tactical stability |
| `order_moves()` | ~3264 | Move ordering: TT > killers > history |
| `evaluate()` | ~3118 | Dispatches to heuristic or NNUE eval |
| `search_iterative()` | ~3904 | Iterative deepening with aspiration windows |
| `compute_relational_features()` | ~2285 | 13 relational NNUE features (distances, scored, pails) |
| `add_relational_features()` | ~2340 | SIMD-accelerated relational feature injection |
| `null_move pruning` | ~3632 | Skip-turn pruning, R=2-3 |
| `futility pruning` | ~3655 | Extended to depth 8 with tuned margins + razoring |
| `LMR` | ~3726 | Late move reductions (log table + history modulation) |

## Important Constraints

- The engine is a PyO3 module: changes to `src/lib.rs` require `maturin develop --release`
- The function `minimax()` is the core search — most improvements go here
- Bitboard representation uses 4 x u64 (white_barrels, black_barrels, white_pails, black_pails)
- Board is 6x6 = 36 squares, stored in the low 36 bits of each u64
- Moves are encoded as `BitMove` with from/to squares and move type
- `FUTILITY_MARGINS` is `[0, 80, 160, 250, 350, 450, 600, 750, 950]` (depth 0-8)
- `self.history` is `[[i32; 36]; 36]` (from-square to-square butterfly history)
- `self.killers` is `[[Option<BitMove>; 2]; MAX_DEPTH]`

## NNUE Training Pipeline

The training pipeline (`python/tonnesjakk/nnue.py`) generates self-play data and
trains a 157→64→32→1 NNUE network. Key features:

- **Score normalization**: `tanh(score / 600)` — smooth sigmoid mapping that
  preserves information across the full eval range
- **Quiet position filtering**: Skip first 4 moves of each game and positions
  with |score| > 3000 to avoid noisy/random/mating positions
- **Lambda blend**: `--lambda 0.85` mixes search scores (85%) with game outcomes
  (15%) for training labels. Balances positional accuracy with game result signal.
- **13 relational features**: Barrel-to-goal distances (4+4), scored barrels (2),
  pails placed (2), current player (1) — computed identically in Python and Rust
- **Multi-worker generation**: `--workers 8` for parallel self-play data generation

### Feature layout (157 total)
- Features 0-143: 144 base board features (6x6 grid x 4 piece planes)
- Features 144-147: White barrel distances to goal (sorted, normalized)
- Features 148-151: Black barrel distances to goal (sorted, normalized)
- Features 152-153: White/black scored barrels (normalized to 0-1)
- Features 154-155: White/black pails placed (0 or 1)
- Feature 156: Current player (+1 white, -1 black)

## Current Performance

- Heuristic: ~500K nodes/sec
- NNUE 64x32: ~100K nodes/sec
- Depth 7 games: ~0.54s/game (heuristic, single-thread)
- With `--workers 8`: ~10 games/sec at depth 7

## Engine Improvement Plan

See `ENGINE_IMPROVEMENTS.md` for the full list of 18 ranked improvements.

**P0 improvements — ALL IMPLEMENTED (2026-02-10):**
Combined result: **-74.1% node reduction** at depth 8 (1,447K → 374K nodes).

1. **Log LMR table** (line ~3726): Precomputed log table (divisor 1.0) + history modulation (-0.6%)
2. **Null move tuning** (line ~3632): R=2-3 + depth/eval boosts at 150cp (-0.6%)
3. **Extended futility** (line ~3655): Depth 8 with tuned margins + razoring (-31.0%)
4. **IIR + continuation history** (line ~3490, ~3264): Biggest win (-69.3%)

**Next up (P1):** TT clustering, endgame detection

## Scripts & Data

- `scripts/test_model.py` — Time-based NNUE vs heuristic comparison framework
- `scripts/bench_engine.py` — Engine benchmark (nodes/sec, depth stats)
- `scripts/diagnose_wdl.py` — Per-move diagnostic tool comparing NNUE and heuristic evals
- `training_d7.npz` — 7.7M positions at depth 7 (old /1000 normalization, needs rescaling)
- `nnue_weights_gen1.json` — Gen 1 model (157 features, +275 ELO at equal depth 5)

## Git Conventions

- Main branch: `master`
- Feature branches: `feature/<name>` (e.g., `feature/lmr-table`)
- Commit messages: Describe what changed and why
- Always `maturin develop --release` after Rust changes to verify they compile
