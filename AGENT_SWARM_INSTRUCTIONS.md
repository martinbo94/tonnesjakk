# Agent Swarm Instructions: NNUE Training & Comparison Pipeline

## Project Overview

Tonnesjakk is a Rust/Python game engine for a Norwegian board game (6x6, barrels + pails race game). The engine has a fast heuristic evaluator (~500K nodes/sec) and we want a neural network (NNUE) evaluator that is strong enough to beat the heuristic at equivalent time controls.

**Current state**: The engine core (search + heuristic) is production-ready. NNUE training works but comparing/benchmarking NNUE models is broken - games take forever because the comparison function has no proper time control and games hit the 100-move limit.

## Critical Issues to Fix FIRST (Before Any Training)

### Issue 1: `compare_nnue()` games never finish
**File**: `python/tonnesjakk/nnue.py`, function `compare_nnue()` (line ~826)
**Problem**: Games hit the `move_count < 100` limit at depth 5+, taking minutes per game. At depth 4, games all draw (not deep enough to find wins).
**Fix needed**:
- Reduce max moves to 50 (game should be decided by then, or it's a draw)
- Add a per-game time limit (e.g., 60 seconds, then declare draw)
- Print progress after EVERY game (not every 10), including time per game
- The function already has `verbose` parameter - make it useful

### Issue 2: Dead quantized NNUE code
**File**: `src/lib.rs`
**Problem**: `QuantizedNNUE` struct uses hardcoded `HIDDEN1_SIZE = 64`, crashes on 32x16/16x8 networks. Currently disabled: `use_quantized: false` and `self.qnnue = None`.
**Fix needed**: Either remove all QuantizedNNUE code (recommended) or make it dynamic. It's ~200 lines of dead code that causes confusion.

### Issue 3: README and TODO are outdated
**Files**: `README.md`, `todo.md`
**Problem**: README says 144 features (actually 147 with 3 relational). Lists search features as future work but they're all implemented. Doesn't mention transposed weights optimization.

## Architecture

```
Python Layer:
  python/tonnesjakk/nnue.py    - Training, data generation, comparison
  train_compare_sizes.py        - Multi-size training script
  benchmark_nnue.py             - Speed benchmark

Rust Layer (src/lib.rs, ~4000 lines):
  BitBoard                      - Board representation (u64 bitboards)
  BitBoardEngine                - Search (minimax, alpha-beta, PVS, etc.)
  IncrementalNNUE               - Float NNUE with incremental accumulator
  QuantizedNNUE                 - DEAD CODE, disabled, crashes on non-64x32

Build:
  cargo build --release         - Build Rust
  pip install -e .              - Build + install Python bindings (uses maturin)
  .venv/Scripts/python.exe      - Python in venv (Windows)
```

## NNUE Architecture

```
Input (147) -> Hidden1 (variable) -> ReLU -> Hidden2 (variable) -> ReLU -> Output (1) -> Tanh

Feature encoding:
  - 144 base features: 36 squares x 4 piece types (one-hot)
  - 3 relational features: white_scored/4, black_scored/4, current_player (+1/-1)

Incremental updates:
  - Layer 1 accumulator is cached and updated per-move (add/remove ~2-4 features)
  - Weights stored in transposed layout for cache-friendly SIMD access
  - SIMD f32x8 for add_feature/remove_feature operations
```

## Team Structure (5 Teammates)

### Teammate 1: "Fixer" - Fix compare_nnue and cleanup dead code
**Priority**: HIGHEST - must complete before others can test
**Files**: `python/tonnesjakk/nnue.py`, `src/lib.rs`
**Tasks**:
1. Fix `compare_nnue()` function:
   - Change `move_count < 100` to `move_count < 50`
   - Add per-game time limit (60 sec timeout -> declare draw)
   - Print after every game with move count, time, and running score
   - Flush output (`flush=True`) so it appears in real-time
2. Remove all QuantizedNNUE dead code from `src/lib.rs`:
   - Remove `QuantizedNNUE` struct and impl
   - Remove `QuantizedAccumulator` struct and impl
   - Remove `QuantizedAccumulatorStack` struct and impl
   - Remove `qnnue` and `qacc_stack` fields from `BitBoardEngine`
   - Remove `use_quantized` field and all branches checking it
   - Remove the quantization constants (`WEIGHT_SCALE_L1`, `WEIGHT_SCALE_L2`, `ACTIVATION_SCALE`, `HIDDEN1_SIZE`, `HIDDEN2_SIZE`)
   - Remove `use wide::i16x8` import (keep `f32x8`)
3. Rebuild and verify: `cargo build --release && pip install -e .`
4. Test: `python -m tonnesjakk.nnue --compare nnue_64x32/nnue_weights.json nnue_32x16/nnue_weights.json --compare-games 10 --depth 5`
   - Should complete in under 5 minutes with per-game output

### Teammate 2: "Trainer" - Train high-quality networks
**Depends on**: Nothing (can run in parallel)
**Files**: `python/tonnesjakk/nnue.py`, training scripts
**Tasks**:
1. Generate high-quality training data:
   ```
   python -m tonnesjakk.nnue --games 10000 --depth 7 --save-data training_data_10k_d7.npz --no-compare --arch 64 32
   ```
2. Train three network sizes on the same data:
   ```
   python -m tonnesjakk.nnue --load-data training_data_10k_d7.npz --arch 64 32 --epochs 80 --output nnue_64x32 --no-compare --no-history
   python -m tonnesjakk.nnue --load-data training_data_10k_d7.npz --arch 32 16 --epochs 80 --output nnue_32x16 --no-compare --no-history
   python -m tonnesjakk.nnue --load-data training_data_10k_d7.npz --arch 16 8 --epochs 80 --output nnue_16x8 --no-compare --no-history
   ```
3. Log training loss for each size, note convergence
4. Save results to `training_results.json`

### Teammate 3: "Benchmarker" - Speed benchmarks after Fixer is done
**Depends on**: Teammate 1 (Fixer) must finish first
**Files**: `benchmark_nnue.py`, `train_compare_sizes.py`
**Tasks**:
1. After Fixer rebuilds, run speed benchmarks for all 3 sizes + heuristic
2. Run head-to-head comparisons:
   - 64x32 vs 32x16 (20 games, depth 5)
   - 64x32 vs 16x8 (20 games, depth 5)
   - 32x16 vs 16x8 (20 games, depth 5)
   - Best NNUE vs heuristic (20 games, depth 5)
3. Collect and summarize:
   - Nodes/sec per architecture
   - Speed ratio vs heuristic
   - Win rates and ELO differences
   - Time per game

### Teammate 4: "Documenter" - Update README and TODO
**Depends on**: Teammate 1 (Fixer) for accurate code state
**Files**: `README.md`, `todo.md`
**Tasks**:
1. Update README:
   - Fix architecture diagram: 147 features (144 base + 3 relational)
   - Mark all search optimizations as implemented (PVS, LMR, null-move, etc.)
   - Add NNUE optimization section (transposed weights, SIMD)
   - Update file structure section
   - Add training instructions
2. Update TODO:
   - Move completed items to "Fullfort" section
   - Add current priorities (NNUE training, self-play loop)
   - Remove items that are no longer relevant

### Teammate 5: "Self-Play Architect" - Design the self-improvement loop
**Depends on**: Nothing (research task)
**Files**: Read-only exploration of `python/tonnesjakk/nnue.py`, `src/lib.rs`
**Tasks**:
1. Research and design a self-play improvement loop:
   - Generation 0: Train NNUE from heuristic-generated games (current approach)
   - Generation N: Use NNUE from Gen N-1 to generate training data for Gen N
   - How to detect if a new generation is actually better (gatekeeper matches)
   - How many games/depth per generation
2. Write a plan document `SELF_PLAY_PLAN.md` with:
   - Recommended pipeline (commands to run)
   - Success criteria (when to stop iterating)
   - How to handle the speed problem (NNUE is slower than heuristic for data generation)
   - Whether to use time-based or depth-based search for self-play
3. Consider: should we generate data with heuristic (fast) and train NNUE to predict heuristic + learn patterns beyond it?

## Key Technical Details for All Teammates

### Building
```bash
# Activate venv
.venv/Scripts/activate  # Windows

# Build Rust + install Python package
cargo build --release
pip install -e .
```

### Running tests
```bash
python benchmark_nnue.py                    # Speed comparison
python -m tonnesjakk.nnue --compare A B     # Head-to-head (A, B = paths or "heuristic")
python train_compare_sizes.py               # Full comparison pipeline
```

### Important Rust code locations (src/lib.rs)
- `IncrementalNNUE::load()` ~line 2600: Loads JSON weights, creates transposed matrix
- `IncrementalNNUE::add_feature()` ~line 2770: SIMD feature update (hot path)
- `IncrementalNNUE::evaluate_full()` ~line 2710: Full board evaluation
- `BitBoardEngine::evaluate()` ~line 3438: Main eval dispatch (NNUE or heuristic)
- `BitBoardEngine::minimax()` ~line 3844: Main search function
- `BitBoardEngine::search_depth()` ~line 3630: Root search with aspiration windows

### Important Python code locations (python/tonnesjakk/nnue.py)
- `TonnesjakkNNUE` class ~line 55: PyTorch model definition
- `DataGenerator.generate_dataset()` ~line 230: Self-play data generation
- `train_model()` ~line 550: Training loop
- `train_nnue()` ~line 629: Full pipeline
- `compare_nnue()` ~line 826: Head-to-head comparison (BROKEN - see Issue 1)

### Game rules summary
- 6x6 board, 4 barrels + 1 pail per player
- Goal: get barrels to opponent's starting row (they score and are removed)
- Each turn: move a barrel (1 step or jump over adjacent piece)
- Pail: placed once per game, acts as a blocker
- When pail is placed, you also move a barrel that turn
- First to score all 4 barrels wins, or most scored when opponent can't move
</content>
</invoke>