# Tonnesjakk AI

An AI engine for **Tonnesjakk** ("barrel chess"), a Norwegian board game from TV2's *Farmen Kjendis*. Built as a Rust/Python hybrid with a game tree search engine and a neural network evaluation function trained through self-play.

## The Game

Tonnesjakk is played on a 6x6 board. Two players (white and black) each have **4 barrels** and **1 pail** (a blocking piece). The goal is to push your barrels across the board to the opponent's back row — when a barrel reaches the far side, it scores and is removed. First player to score all 4 barrels wins.

**Movement rules:**
- Barrels start off the board and are placed from your own back row
- Barrels move one square in any direction (orthogonal or diagonal — 8 directions total)
- Barrels can **jump over** adjacent pieces, landing on the empty square behind them (like checkers)
- Each player gets one **pail** per game — a piece that can be placed anywhere on the board as a permanent blocker

The game is deceptively tactical: barrel chains create jump sequences, the pail can block key lanes, and the 6x6 board means every move matters.

## How It Works

The AI has two main components: a **search algorithm** that explores possible future moves (game tree search), and an **evaluation function** that estimates how good a position is without searching further.

### Search: Looking Ahead

The engine uses the same fundamental approach as chess engines like Stockfish: build a tree of all possible move sequences, evaluate the positions at the leaves, and propagate the best scores back up.

**Core algorithm — Alpha-Beta with PVS:**
The base is [minimax](https://en.wikipedia.org/wiki/Minimax) with [alpha-beta pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning). Alpha-beta eliminates branches that provably can't affect the outcome, cutting the number of positions examined by ~95% compared to brute-force minimax. On top of this, [Principal Variation Search (PVS)](https://www.chessprogramming.org/Principal_Variation_Search) uses a zero-width window to test non-best moves, narrowing the search further when move ordering is good.

**Making search faster — pruning and reductions:**

The engine uses several techniques to search deeper within the same time budget. The common theme: spend time on moves that matter, skip or reduce effort on moves that probably don't.

| Technique | What it does |
|-----------|-------------|
| **Transposition table** | A hash map that stores previously evaluated positions. If we reach the same board state through a different move order, we reuse the cached result instead of re-searching. Uses [Zobrist hashing](https://www.chessprogramming.org/Zobrist_Hashing) for fast, collision-resistant position hashing. |
| **Null move pruning** | "What if I just skip my turn?" If giving the opponent a free move still results in a position so good we wouldn't play it, the real position must be even better — so we can prune the whole subtree. Saves a lot of time in positions where one side has a clear advantage. |
| **Late move reductions (LMR)** | Moves are ordered by estimated quality. The first few moves (likely the best ones) are searched at full depth, but later moves get reduced depth. If a reduced-depth search finds a surprising result, we re-search at full depth. This is based on the observation that good move ordering means the best move is usually found early. |
| **Futility pruning** | At shallow depths, if the static evaluation is so far below alpha (the current best) that no single move could plausibly recover, skip those moves entirely. Extended to depth 8 with tuned margins. |
| **Razoring** | An aggressive extension of futility pruning: at very shallow depths (1-3), if the evaluation is far enough below alpha, drop directly into quiescence search without trying any moves at all. |
| **IIR (Internal Iterative Reductions)** | If we have no transposition table hit for a position (meaning we have no idea which move to try first), reduce the search depth by 1. The idea: searching without a good first guess is less productive, so invest less. |
| **Killer moves** | Remember which moves caused beta cutoffs (pruning) at each depth level. Try these "killer moves" early at sibling nodes, since a move that's great in one position is often great in similar positions at the same depth. |
| **History heuristic** | Long-term memory of which (from-square, to-square) combinations have historically been good. Moves with high history scores are searched first. The continuation history variant also tracks which moves are good *after* a specific previous move. |
| **Aspiration windows** | Instead of searching with infinite alpha-beta bounds, start with a narrow window around the expected score. If the true score falls outside the window, re-search with wider bounds. This gamble usually pays off because iterative deepening gives a good prediction of the next iteration's score. |
| **Quiescence search** | At leaf nodes, don't just stop and evaluate — continue searching "noisy" tactical moves (pieces near the goal line that might score). This prevents the [horizon effect](https://www.chessprogramming.org/Horizon_Effect), where the engine thinks a position is safe simply because the opponent's winning move falls just beyond the search depth. |

**Result:** These techniques combine to reduce the search tree by about **74%** at depth 8, allowing the engine to search roughly 2x deeper in the same time.

### Evaluation: Scoring a Position

When the search reaches a leaf node, it needs a score. There are two options:

**1. Heuristic evaluation (handcrafted):**
A simple formula: how far have my barrels advanced (+100 per row), how many have I scored (+500 each), is my pail well-placed (+10 for center). Fast (~500K evaluations/sec) but limited in understanding.

**2. NNUE (neural network):**
A small neural network trained on millions of self-play positions. More on this below.

## NNUE: The Neural Network

[NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) is a technique pioneered in Shogi engines and adopted by Stockfish in 2020. The key insight: a small neural network can be dramatically faster than a large one if you design it so the first layer only needs **incremental updates** when a move is made.

### Architecture

```
Input (157 features) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Tanh
                                                                              ↓
                                                                        Score: [-1, +1]
```

The network has ~12,000 parameters — tiny by modern standards. This is intentional: the network is evaluated millions of times per second during search, so every microsecond matters.

### Features (What the Network Sees)

The 157 input features encode the board state:

| Features | Count | Description |
|----------|-------|-------------|
| Board planes | 144 | 6x6 grid with 4 binary channels (white barrel, black barrel, white pail, black pail present at each square) |
| Barrel distances | 8 | How close each barrel is to scoring (4 per side, sorted, normalized 0-1) |
| Scored barrels | 2 | How many barrels each side has scored (normalized to 0-1) |
| Pails placed | 2 | Whether each side has placed their pail (0 or 1) |
| Current player | 1 | Who moves next (+1 white, -1 black) |

The board planes give the network raw spatial information, while the relational features provide pre-computed strategic summaries (like "how close is the nearest barrel to scoring?").

### Incremental Updates (Why NNUE is Fast)

The first layer (`157 → 64`) is the bottleneck — it's a 157x64 matrix multiply. But here's the trick: when a piece moves from square A to square B, only ~2-4 of the 157 input features change. Instead of recomputing the entire first layer, we:

1. Subtract the contribution of features that turned off
2. Add the contribution of features that turned on

This turns an O(157 * 64) operation into an O(4 * 64) operation — roughly **40x faster**. The engine maintains an "accumulator stack" that caches the first-layer output and can be efficiently updated on make/unmake move.

The relational features (13 values like barrel distances) can't be incrementally updated since they depend on global board state, so these are recomputed each evaluation and injected via a separate SIMD-accelerated path.

### SIMD Acceleration

The inner loops use 8-wide SIMD (Single Instruction, Multiple Data) via the [`wide`](https://crates.io/crates/wide) crate — specifically `f32x8` operations that process 8 float values in a single CPU instruction. This applies to:
- Accumulator updates (adding/removing feature columns)
- Relational feature injection
- ReLU activation and layer-2/layer-3 forward pass

### Training

The NNUE is trained in PyTorch on data generated by self-play:

**1. Data generation:**
The engine plays thousands of games against itself at a fixed search depth (typically depth 7). For each game, it records every position along with:
- The **search score** — what the engine's search thought about the position
- The **game outcome** — who actually won (+1, -1, or 0 for draw)

Some positions are filtered out to reduce noise:
- First 4 moves of each game (random opening phase)
- Positions with extreme scores (|score| > 3000, likely won/lost already)

**2. Label construction (lambda blend):**
The training label for each position is a weighted mix:

```
label = lambda * search_score + (1 - lambda) * game_outcome
```

With lambda=0.85 (default), labels are 85% search score and 15% game outcome. This balances two signals: the search score is a precise position-by-position assessment, while the game outcome provides a ground-truth correction that prevents the network from perpetuating any systematic search biases.

**3. Score normalization:**
Raw search scores (integers like +350, -1200) are mapped to the [-1, +1] range using `tanh(score / 600)`. This sigmoid-like mapping has nice properties:
- Scores near 0 (roughly equal positions) get the most resolution
- Large scores (winning/losing) asymptotically approach +/-1
- No information is lost to clipping, unlike a linear normalization

**4. Training loop:**
Standard supervised learning: MSE loss, Adam optimizer, ~50 epochs. The trained PyTorch model is exported as a JSON weight file that the Rust engine loads at runtime.

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                 Python (web/server.py)                │
│                  FastAPI + Uvicorn                    │
│           Serves web UI for human vs AI play         │
└───────────────────────┬──────────────────────────────┘
                        │ PyO3
┌───────────────────────▼──────────────────────────────┐
│                Rust Core (src/lib.rs)                 │
│                                                      │
│  Board / BitBoard        Game state & move generation│
│  BitBoardEngine          Search (PVS, LMR, TT, ...) │
│  IncrementalNNUE         SIMD neural net evaluation  │
│  Engine                  Python-facing API wrapper   │
└──────────────────────────────────────────────────────┘
                        │ PyO3
┌───────────────────────▼──────────────────────────────┐
│           Python (python/tonnesjakk/nnue.py)         │
│                                                      │
│  Self-play data generation (parallel workers)        │
│  PyTorch NNUE training & export to JSON              │
│  Model comparison & benchmarking                     │
└──────────────────────────────────────────────────────┘
```

The Rust engine (~4,000 lines) handles everything performance-critical: board representation, move generation, search, and NNUE inference. Python handles the training pipeline and the web UI. The two communicate through [PyO3](https://pyo3.rs/) bindings, compiled with [maturin](https://github.com/PyO3/maturin).

### Board Representation: Bitboards

The board uses 4 unsigned 64-bit integers — one per piece type (white barrels, black barrels, white pail, black pail). Each integer uses its low 36 bits to represent the 6x6 grid, where bit `n` being set means a piece exists at square `n` (with `square = row * 6 + col`).

This representation makes move generation and board queries extremely fast through bitwise operations. For example, "are there any white barrels in row 0?" is a single AND with a precomputed mask.

## Getting Started

### Prerequisites

- Python 3.10+
- Rust toolchain (install from [rustup.rs](https://rustup.rs/))
- PyTorch (for NNUE training only)

### Installation

```bash
git clone <repo>
cd tonnesjakk

python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/Mac

pip install maturin torch fastapi uvicorn
maturin develop --release
```

### Play via Web UI

```bash
cd web
python server.py
# Open http://localhost:8000
```

### Use from Python

```python
from tonnesjakk import Board, Engine

board = Board()
engine = Engine()
engine.load_nnue("nnue_weights_gen1.json")

result = engine.search(board, depth=7)
print(f"Best move: {result.best_move}, Score: {result.score}")
board.make_move(result.best_move)
```

### Train a New NNUE

```bash
# Generate self-play data and train (20K games, depth 7, 8 parallel workers)
python -m tonnesjakk.nnue --games 20000 --depth 7 --lambda 0.85 --workers 8

# Save training data for later reuse
python -m tonnesjakk.nnue --games 20000 --depth 7 --lambda 0.85 --workers 8 --save-data training.npz

# Train on existing data
python -m tonnesjakk.nnue --load-data training.npz --epochs 50

# Compare NNUE vs heuristic (time-limited, more realistic)
python scripts/test_model.py nnue_weights.json --time-ms 200 --games 50

# Compare at fixed depth
python -m tonnesjakk.nnue --compare nnue_weights.json heuristic --compare-games 100 --depth 7
```

## Performance

On a modern CPU (single-threaded):

| Metric | Heuristic eval | NNUE eval |
|--------|---------------|-----------|
| Nodes/sec | ~500K | ~100K |
| Typical depth in 1s | 10-12 | 7-9 |

The NNUE is ~5x slower per node but compensates with better positional understanding, leading to more efficient pruning and stronger play at equal time controls.

## Project Structure

```
tonnesjakk/
├── src/lib.rs                    # Rust engine (~4,000 lines)
├── python/tonnesjakk/
│   ├── nnue.py                   # NNUE training pipeline (PyTorch)
│   ├── __init__.py               # Python package init
│   └── export_nnue.py            # Weight export utilities
├── web/
│   ├── server.py                 # FastAPI web backend
│   └── index.html                # Browser-based game UI
├── scripts/
│   ├── test_model.py             # Time-based model comparison
│   ├── bench_engine.py           # Engine speed benchmarks
│   └── diagnose_wdl.py           # Per-move NNUE diagnostic tool
├── train_nnue.py                 # Convenience wrapper for training
├── nnue_weights_gen1.json        # Trained NNUE weights
├── ENGINE_IMPROVEMENTS.md        # Search optimization roadmap
├── CLAUDE.md                     # AI assistant project context
├── Cargo.toml                    # Rust dependencies
└── pyproject.toml                # Python/maturin build config
```

## Technologies

- **Rust** — Engine core (speed + safety)
- **PyO3 / maturin** — Rust-to-Python bindings
- **wide** — Portable SIMD (`f32x8`) for NNUE inference
- **PyTorch** — Neural network training
- **FastAPI** — Web backend
- **Vanilla JS** — Web frontend

## License

MIT

---

*Inspired by [Stockfish](https://stockfishchess.org/), [Shogi NNUE](https://www.chessprogramming.org/NNUE), and TV2's Farmen Kjendis*
