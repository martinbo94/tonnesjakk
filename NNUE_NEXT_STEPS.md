# NNUE Next Steps — If 20 Dense Features Don't Help

Written 2026-02-18. Context: HalfPail NNUE with 6 dense features hit a val loss
floor of 0.5867 across multiple architectures (128x32, 128x64). Playing strength
was -115 to -230 ELO vs heuristic at equal depth 5. We expanded to 20 dense
features (distances, threats, score differential, blocking) to give the model the
same information the heuristic uses. If this doesn't work, here's the plan.

## Step 1: Diagnose WHERE the model fails

Before changing anything, understand what's going wrong.

```bash
# Play 5 games, save game records with per-move evals
.venv\Scripts\python.exe scripts\test_model.py nnue_halfpail_d20\nnue_weights.json --time-ms 500 --games 5

# Analyze eval quality from game records
.venv\Scripts\python.exe scripts\eval_analysis.py
```

Key questions:
- Is the NNUE "delusional" (thinks it's winning when losing)?
- Are errors concentrated in openings, midgame, or endgame?
- Does the NNUE miss specific tactical patterns (threats, blocking)?

## Step 2: Check if the loss function is the problem

WDL cross-entropy may not optimize for playing strength. Try MSE:

```bash
.venv\Scripts\python.exe -m tonnesjakk.nnue --load-data training_consolidator_d9.bin --halfpail --epochs 20 --batch-size 8192 --lr 0.001 --arch 128 32 --output nnue_test_mse --no-compare --loss mse
```

Compare val loss trajectory. If MSE breaks through where WDL-CE didn't,
the loss function was the bottleneck.

## Step 3: Check if the training target is the problem

We train against heuristic search scores (lambda=1.0). The NNUE is trying
to replicate the heuristic — but why would a replica beat the original?

Try mixing in game outcomes (lambda=0.7):

```bash
.venv\Scripts\python.exe -m tonnesjakk.nnue --load-data training_consolidator_d9.bin --halfpail --epochs 20 --batch-size 8192 --lr 0.001 --arch 128 32 --output nnue_test_lambda07 --no-compare --lambda 0.7
```

If the NNUE learns patterns the heuristic misses (from game outcomes),
it could surpass the heuristic even with worse raw score prediction.

## Step 4: Try a simpler architecture first

The HalfPail sparse features (3996 per perspective) may be too sparse for
the amount of training data per feature. Most features are rarely active.

Test the old 164-feature dense NNUE with the full 20 relational features
and a larger network:

```bash
.venv\Scripts\python.exe -m tonnesjakk.nnue --load-data training_consolidator_d9.bin --epochs 20 --batch-size 8192 --lr 0.001 --arch 128 32 --output nnue_test_dense --no-compare
```

If this beats HalfPail at equal depth, the sparse architecture is the
bottleneck and we should focus on the dense approach first.

## Step 5: Learning rate and batch size sweep

The current setup (lr=0.001, batch=8192) may not be optimal. Quick tests:

| Test | Change | Why |
|------|--------|-----|
| Higher LR | `--lr 0.003` | Escape shallow local minimum |
| Lower LR | `--lr 0.0003` | More precise convergence |
| Smaller batch | `--batch-size 2048` | More gradient updates per epoch (4x) |
| Larger batch | `--batch-size 16384` | Smoother gradients, possibly worse |

Run each for 10 epochs and compare val loss at epoch 10.

## Step 6: Feature importance analysis

After training, check which of the 20 dense features actually matter:

```python
import torch, json
d = json.load(open("nnue_halfpail_d20/nnue_weights.json"))
fc2 = torch.tensor(d["weights"]["fc2_weight"])  # [H2, 2*H1+20]
dense_weights = fc2[:, -20:]  # last 20 columns = dense feature weights
importance = dense_weights.abs().sum(dim=0)
feature_names = [
    "w_dist0","w_dist1","w_dist2","w_dist3",
    "b_dist0","b_dist1","b_dist2","b_dist3",
    "w_scored","b_scored","w_pail","b_pail",
    "player","w_threats","b_threats","score_diff",
    "w_barrels","b_barrels","w_blocks","b_blocks"
]
for name, imp in sorted(zip(feature_names, importance.tolist()), key=lambda x: -x[1]):
    print(f"  {name:12s} {imp:.3f}")
```

If the new features (distances, threats, blocking) have low importance,
the model isn't learning from them and we need a different approach.

## Step 7: Data quality check

The training data is heuristic self-play at depth 9. If the heuristic
plays poorly in certain situations, the training data is polluted.

- Check game outcome distribution: is it heavily skewed to one side?
- Check position diversity: are there enough endgame positions?
- Check score distribution: are most scores near 0 (all draws)?

## Priority order

1. Diagnose (Step 1) — always understand before changing
2. Loss function (Step 2) — quick test, high potential
3. Lambda blend (Step 3) — quick test, different training signal
4. Dense architecture (Step 4) — tests if HalfPail sparse is the issue
5. LR/batch sweep (Step 5) — systematic hyperparameter search
6. Feature importance (Step 6) — informs architecture decisions
7. Data quality (Step 7) — last resort, hardest to fix
