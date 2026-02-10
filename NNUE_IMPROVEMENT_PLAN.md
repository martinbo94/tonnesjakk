# NNUE Improvement Plan for Tonnesjakk

## Current Status (Gen 3 pending)
- **Gen 1**: 144 features, depth 8, +275 ELO vs older model
- **Gen 2**: 157 features, depth 6, -182 ELO vs heuristic (too shallow)
- **Gen 3**: 157 features, depth 9, **PENDING** - should beat heuristic

## Phase 1: Quick Wins (Implement First)

### 1.1 Quiet Position Filtering
**Expected gain: +50-100 ELO**

Skip training positions that are tactically volatile:

```python
def is_quiet_position(board, search_result) -> bool:
    """Filter out noisy positions that hurt training."""

    # Skip if a barrel is about to score (1 row from goal)
    # These positions are too "obvious" - doesn't help learning

    # Skip if best move is a scoring move
    # The eval will swing wildly on these

    # Skip if eval is extreme (|score| > 0.9)
    # Already decided games don't teach much

    # Skip if position has very few moves (< 3)
    # Forced positions are noisy

    return True  # Position is quiet enough
```

**Files to modify:**
- `python/tonnesjakk/nnue.py`: Add filtering in `play_game()` method

### 1.2 Wider First Layer
**Expected gain: +20-50 ELO**

The first layer learns piece-square relationships. More neurons = more patterns.

```python
# Current
TonnesjakkNNUE(hidden1=64, hidden2=32)   # 12,225 params

# Proposed
TonnesjakkNNUE(hidden1=128, hidden2=64)  # 30,817 params
```

**Files to modify:**
- `python/tonnesjakk/nnue.py`: Change defaults
- `src/lib.rs`: Update Accumulator size if needed

### 1.3 Learning Rate Schedule
**Expected gain: +10-20 ELO**

```python
# Current: Fixed LR
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Proposed: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

**Files to modify:**
- `python/tonnesjakk/nnue.py`: Add scheduler in `train_model()`

---

## Phase 2: Feature Engineering

### 2.1 Blocking Features (4 features)
**Expected gain: +30-50 ELO**

For each player's barrels, compute:
- Number of barrels with clear path to goal
- Number of barrels blocked by opponent pieces
- Number of barrels blocked by own pieces
- Number of "chain push" opportunities

```python
def compute_blocking_features(board_array, white_scored, black_scored):
    """
    Compute path-blocking features for each side.
    """
    features = []

    for player in [1, -1]:  # White, Black
        goal_row = 0 if player == 1 else 5
        barrels = find_barrels(board_array, player)

        clear_paths = 0
        blocked_by_opponent = 0
        blocked_by_own = 0

        for barrel_pos in barrels:
            path_status = analyze_path_to_goal(board_array, barrel_pos, goal_row)
            # ... count blocking situations

        features.extend([clear_paths/4, blocked_by_opponent/4, blocked_by_own/4])

    return features  # 6 features total
```

**Files to modify:**
- `python/tonnesjakk/nnue.py`: Extend `board_to_tensor()`
- `src/lib.rs`: Update `RELATIONAL_FEATURES` constant and `add_relational_features()`

### 2.2 Threat/Push Features (4 features)
**Expected gain: +20-40 ELO**

Which pieces can interact:
- Can white push a black barrel?
- Can black push a white barrel?
- Push opportunities toward goal
- Push opportunities away from goal

### 2.3 Tempo Features (2 features)
**Expected gain: +10-20 ELO**

- Moves until white can score (min distance)
- Moves until black can score (min distance)

---

## Phase 3: Architecture Experiments

### 3.1 Skip Connection (Residual)
```
Input(157) → L1(128) → L2(64) + skip(L1) → L3(32) → Output(1)
```

### 3.2 Dual Accumulators
Separate feature transformers for white/black perspective, then combine.

### 3.3 Larger Network
```
157 → 256 → 128 → 64 → 1  (if we have enough data)
```

---

## Phase 4: Training Pipeline

### 4.1 Position Diversity Sampling
Ensure balanced representation:
- 30% early game (move 1-15)
- 40% mid game (move 16-40)
- 30% late game (move 41+)

### 4.2 Eval Distribution Balance
Target distribution of labels:
- 20% strong white advantage (> 0.5)
- 20% slight white advantage (0.1 to 0.5)
- 20% equal (-0.1 to 0.1)
- 20% slight black advantage (-0.5 to -0.1)
- 20% strong black advantage (< -0.5)

### 4.3 Self-Play League
Instead of always self-play:
- 50% games vs current best
- 25% games vs previous generation
- 25% games vs heuristic

---

## Implementation Priority

| Priority | Task | Expected ELO | Effort |
|----------|------|--------------|--------|
| 1 | Quiet position filtering | +50-100 | Low |
| 2 | Wider first layer (128) | +20-50 | Low |
| 3 | Learning rate schedule | +10-20 | Low |
| 4 | Blocking features | +30-50 | Medium |
| 5 | More training data (20K+) | +20-40 | Low |
| 6 | Threat/push features | +20-40 | Medium |
| 7 | Skip connections | +10-30 | Medium |
| 8 | Position diversity | +10-20 | Medium |

---

## Testing Protocol

After each change:
1. Train with 10K games, depth 9
2. Compare vs previous best: `--compare nnue_weights.json nnue_weights_prev.json`
3. Compare vs heuristic: `--compare nnue_weights.json heuristic`
4. Only keep if improvement is statistically significant (>50 games, >55% win rate)

---

## Notes

- Always save training data with `--save-data` for reproducibility
- Track all experiments in `nnue_history.json`
- Commit working models to git with descriptive messages
