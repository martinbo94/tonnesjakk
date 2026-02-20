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

### Architecture: HalfPail NNUE

The current NNUE uses a **dual-perspective** architecture inspired by Stockfish's HalfKP. Instead of encoding the board as a flat feature vector, it uses the pail position as a "bucket" to specialize the network:

```
Sparse features (3996 per perspective)
     ↓ EmbeddingBag(3996, 128)
     ↓ ReLU
     ↓
  concat(white_128, black_128, dense_20) = 276 features
     ↓ Linear(276, 32)
     ↓ ReLU
     ↓ Linear(32, 1)
     ↓ Tanh
     ↓
Score: [-1, +1]
```

The network has **~520,000 parameters**. The sparse first layer uses `EmbeddingBag` — only the active features (typically 3-4 per perspective) are looked up, making it efficient despite the large feature space.

### Features (What the Network Sees)

**Sparse features (3996 per perspective):**
Each perspective (white/black) encodes pieces relative to the pail position. The pail serves as a "bucket" (37 possible states: 36 squares + no pail placed), and for each bucket, there are 36 squares × 3 piece types = 108 features. Total: 37 × 108 = 3996. Piece types are: friendly barrel, enemy barrel, enemy pail.

**Dense features (20):**

These are the same relational features used in the legacy NNUE training data (features 144-163):

| Features | Description |
|----------|-------------|
| 4 | White barrel distances to goal (sorted, normalized /5) |
| 4 | Black barrel distances to goal (sorted, normalized /5) |
| 2 | White/black scored barrels (normalized /4) |
| 2 | White/black pails placed (0 or 1) |
| 1 | Current player (+1 white, -1 black) |
| 2 | White/black immediate threats (barrels 1 step from scoring, /4) |
| 1 | Score differential (white_scored - black_scored) /4 |
| 2 | White/black barrels on board (/4) |
| 2 | White/black pail blocking count (/4) |

The dual-perspective design means the network sees the position from both sides simultaneously, which helps it evaluate asymmetric positions more accurately.

### Incremental Updates (Why NNUE is Fast)

The first layer is the bottleneck — it maps 3996 sparse features to 128 hidden units. But here's the trick: when a piece moves from square A to square B, only ~2-4 sparse feature indices change per perspective. Instead of recomputing the entire embedding, we:

1. Subtract the embedding vectors for features that turned off
2. Add the embedding vectors for features that turned on

This turns the first-layer computation from a full lookup into a small incremental update. The engine maintains a "dual accumulator stack" that caches both perspectives and can be efficiently updated on make/unmake move.

The 20 dense features depend on global board state and are recomputed each evaluation.

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

## AlphaZero Training

In addition to the supervised NNUE approach, the project includes a full [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero)-style training pipeline that learns entirely from self-play using Monte Carlo Tree Search (MCTS) guided by a neural network.

### How It Works

1. **Self-play:** The current network plays games against itself using MCTS. At each move, the tree search runs hundreds of simulations, using the network's policy head to guide exploration and value head to evaluate leaf positions.
2. **Data collection:** Each position records the board state, the MCTS visit count distribution (policy target), and the eventual game outcome (value target).
3. **Training:** The network is trained on the accumulated replay buffer to better predict the MCTS policy and game outcomes.
4. **Repeat:** The improved network produces better self-play data, creating a positive feedback loop.

### Network Architecture

Two architectures are available:

**ResNet CNN (default)** — 5 residual blocks with 128 channels (~1.6M parameters). Uses 3x3 convolutions over a 6-plane spatial board representation (white barrels, black barrels, white pail, black pail, current player, bias). The convolutional structure learns spatial patterns like jump sequences and pail blocking.

**MLP** — 3-layer fully-connected network with 128 hidden units (~240K parameters). Faster inference but no spatial inductive bias.

Both have separate policy (1332 move logits) and value (scalar in [-1, +1]) heads.

### MCTS Engine (Rust)

The MCTS implementation (`src/mcts.rs`) runs in Rust for speed:

- **Arena-based tree:** Nodes stored in a `Vec<MCTSNode>` with `u32` indices for cache-friendly traversal
- **PUCT selection:** AlphaZero's variant of UCB, using network policy priors to focus exploration
- **Batched evaluation with virtual loss:** Multiple MCTS leaves are selected per batch using virtual loss to force path diversity, then evaluated in a single network forward pass. This reduces Python-Rust round-trips by ~16x, achieving a 4.4x wall-clock speedup for ResNet inference
- **Heuristic mode:** Can also use the hand-crafted heuristic for leaf evaluation (~700K simulations/sec), used for bootstrapping early in training

### Training Features

- **Heuristic game seeding:** Early training mixes fast heuristic MCTS games (which produce decisive wins/losses) with network self-play to bootstrap the value head
- **Decaying heuristic ratio:** Linearly shifts from imitation to pure self-play over the training run
- **Windowed training:** Samples a fixed window of examples per iteration (biased 75% toward recent data), keeping training time constant as the replay buffer grows
- **Chunked training:** Training runs in resumable chunks with checkpoint + replay buffer persistence

### Running AlphaZero Training

```bash
# Full training run (ResNet, ~8h overnight)
python scripts/train_alphazero.py --chunks 60 --network resnet --simulations 200 --save-dir alphazero_resnet

# Quick test run (MLP, ~30min)
python scripts/train_alphazero.py --chunks 5 --network mlp --simulations 200 --save-dir az_test

# Evaluate a trained model
python -m tonnesjakk.alphazero --evaluate alphazero_resnet/best_model.pt --games 50 --opponent-depth 5
```

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                 Python (web/server.py)                │
│                  FastAPI + Uvicorn                    │
│     Serves web UI for human vs AI (heuristic or AZ)  │
└───────────────────────┬──────────────────────────────┘
                        │ PyO3
┌───────────────────────▼──────────────────────────────┐
│         Rust Core (src/ — 4 modules + lib.rs)        │
│                                                      │
│  board.rs    Board, BitBoard, moves, Zobrist hashing │
│  nnue.rs     IncrementalNNUE, QuantizedNNUE,         │
│              HalfPailNNUE, EvalCache                 │
│  search.rs   BitBoardEngine, TT, Engine (Python API) │
│  mcts.rs     MCTSEngine with batched NN eval         │
│  lib.rs      Module glue, re-exports, pymodule       │
└──────────────────────────────────────────────────────┘
                        │ PyO3
┌───────────────────────▼──────────────────────────────┐
│           Python Training Pipelines                   │
│                                                      │
│  nnue.py       Supervised NNUE training & export     │
│  alphazero.py  AlphaZero self-play + ResNet/MLP      │
│  mcts.py       Python MCTS (reference implementation)│
│  utils.py      Shared helpers (ELO, device detect)   │
└──────────────────────────────────────────────────────┘
```

The Rust engine (~5,600 lines across 4 modules) handles everything performance-critical: board representation, move generation, search, NNUE inference, and MCTS tree search. Python handles neural network training and the web UI. The two communicate through [PyO3](https://pyo3.rs/) bindings, compiled with [maturin](https://github.com/PyO3/maturin).

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
engine.load_nnue("nnue_halfpail/nnue_weights.json")

result = engine.search(board, depth=7)
print(f"Best move: {result.best_move}, Score: {result.score}")
board.make_move(result.best_move)
```

### Train a New NNUE

```bash
# Generate self-play data (streaming mode, 8 parallel workers)
python -m tonnesjakk.nnue --games 250000 --depth 7 --lambda 1.0 --workers 8 --save-data training_d7.bin --save-every 5000

# Train HalfPail NNUE on existing data
python -m tonnesjakk.nnue --load-data training_consolidator_d9.bin --epochs 50 --halfpail

# Compare NNUE vs heuristic (time-limited, more realistic)
python scripts/test_model.py nnue_halfpail/nnue_weights.json --time-ms 200 --games 50

# Compare at fixed depth
python -m tonnesjakk.nnue --compare nnue_halfpail/nnue_weights.json heuristic --compare-games 50 --depth 5
```

## Performance

On a modern CPU (single-threaded):

| Metric | Heuristic eval | HalfPail NNUE |
|--------|---------------|---------------|
| Nodes/sec | ~500K | ~54K (undertrained) |
| Typical depth in 1s | 10-12 | 5-7 |

The HalfPail NNUE is currently undertrained (5 epochs). Poor eval quality causes inefficient pruning, which reduces effective NPS. Extended training should improve both eval quality and search efficiency simultaneously.

## Project Structure

```
tonnesjakk/
├── src/
│   ├── board.rs                  # Board, BitBoard, moves, Zobrist (~1600 lines)
│   ├── nnue.rs                   # NNUE variants + EvalCache (~1860 lines)
│   ├── search.rs                 # Alpha-beta engine + TT (~1700 lines)
│   ├── mcts.rs                   # MCTS with batched NN eval (~1550 lines)
│   └── lib.rs                    # Module glue, re-exports, pymodule (~490 lines)
├── python/tonnesjakk/
│   ├── nnue.py                   # Supervised NNUE training pipeline
│   ├── alphazero.py              # AlphaZero self-play training
│   ├── mcts.py                   # Python MCTS (reference impl)
│   ├── utils.py                  # Shared helpers (ELO, device, etc.)
│   └── __init__.py               # Python package init
├── web/
│   ├── server.py                 # FastAPI web backend (heuristic + AlphaZero)
│   └── index.html                # Browser-based game UI with engine selector
├── scripts/
│   ├── train_alphazero.py        # Chunked AlphaZero training runner
│   ├── test_model.py             # Time-based model comparison
│   ├── bench_engine.py           # Engine speed benchmarks
│   └── diagnose_wdl.py           # Per-move NNUE diagnostic tool
├── nnue_halfpail/
│   ├── nnue_weights.json         # Trained HalfPail NNUE weights
│   └── nnue_model.pt             # PyTorch model checkpoint
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
