# Debug Notes - Tonnesjakk Engine Crash (RESOLVED)

## Problem Summary
The Rust engine crashed (STATUS_ILLEGAL_INSTRUCTION / exit code 0xC000001D) after playing multiple games with **random/varied positions**. The crash was non-deterministic, occurring after ~3-11 games.

## ROOT CAUSE & FIX

**The crash was caused by unbounded quiescence search depth causing stack overflow.**

### The Problem
The `quiesce()` function had no depth limit and could recurse indefinitely when both players kept making "tactical" moves (moves within 1 row of the goal). With varied/random positions, some games reached positions where quiescence search went extremely deep, causing stack overflow which manifested as STATUS_ILLEGAL_INSTRUCTION.

### The Fix
Added a `qsdepth` parameter to `quiesce()` with `MAX_QSEARCH_DEPTH = 8` limit:

```rust
fn quiesce(&mut self, bb: &BitBoard, mut alpha: i32, beta: i32, maximizing: bool, qsdepth: u8) -> i32 {
    const MAX_QSEARCH_DEPTH: u8 = 8; // Prevent stack overflow

    // Prevent stack overflow from unbounded quiescence search
    if qsdepth >= MAX_QSEARCH_DEPTH {
        return stand_pat;
    }

    // Recursive calls now pass qsdepth + 1
}
```

### Why Random Positions Triggered It
- Deterministic positions followed predictable paths that rarely led to deep quiescence
- Random positions explored diverse game states, some of which had many tactical moves back and forth
- The more varied the positions, the higher the chance of hitting a deep quiescence chain

### Verification
- 50+ games with random positions completed successfully
- Works with both debug and release (LTO) builds

## Debug Process Summary

Things that were **ruled out** as causes:
1. **Null Move Pruning** - Disabling it didn't fix the crash
2. **TT Size** - Smaller TT made it crash faster (more re-computation = faster stack exhaustion)
3. **Memory clearing** - `full_reset()` between games didn't help
4. **Position storage** - Crash happened even without storing positions
5. **LTO** - Crash persisted with LTO disabled
6. **SIMD/wide crate** - That code only runs with NNUE loaded (not in tests)

## Performance Results

With NMP + quiescence depth limit:
- Depth 12: 2.85s, 3.2M nodes (vs 6.9s, 7.2M without NMP)
- **59% speedup, 56% fewer nodes**
