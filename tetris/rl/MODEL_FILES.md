# Model Files Explained

## Two Models Are Saved During Training

### 1. Final Model (`tetris_rl_final.zip`)
**Location:** `./rl_logs/tetris_rl_final.zip`

**What it is:**
- Saved at the **end** of training
- Contains the **final weights** after all timesteps
- This is what you get after training completes

**When to use:**
- If you want the model from the end of training
- If training completed successfully
- Default choice for most cases

**Example:**
```bash
uv run python -m tetris.rl.training evaluate \
    --model ./rl_logs/tetris_rl_final.zip
```

### 2. Best Model (`checkpoints/best/best_model.zip`)
**Location:** `./rl_logs/checkpoints/best/best_model.zip`

**What it is:**
- Saved **during** training based on evaluation performance
- Contains the **best performing weights** seen during training
- Updated whenever evaluation shows better performance

**When to use:**
- If training was interrupted
- If you want the peak performance model (might be better than final)
- If training went on too long and performance degraded

**Example:**
```bash
uv run python -m tetris.rl.training evaluate \
    --model ./rl_logs/checkpoints/best/best_model.zip
```

## Which Should You Use?

### Use Final Model If:
- ✅ Training completed successfully
- ✅ You want the latest learned weights
- ✅ Training didn't overfit

### Use Best Model If:
- ✅ Training was interrupted
- ✅ You suspect overfitting (performance decreased at end)
- ✅ You want the peak performance

## How to Check Which Is Better

```bash
# Test final model
uv run python -m tetris.rl.training evaluate \
    --model ./rl_logs/tetris_rl_final.zip \
    --episodes 20

# Test best model
uv run python -m tetris.rl.training evaluate \
    --model ./rl_logs/checkpoints/best/best_model.zip \
    --episodes 20

# Compare the results!
```

## Checkpoint Files

During training, checkpoints are also saved:
- `./rl_logs/checkpoints/tetris_rl_50000.zip` - Model at 50k timesteps
- `./rl_logs/checkpoints/tetris_rl_100000.zip` - Model at 100k timesteps
- etc.

These let you:
- Resume training from a checkpoint
- Evaluate performance at different training stages
- Compare learning progress

## Summary

**Yes, `evaluate` and `visualize` use the weights from the model file you specify.**

- `--model ./rl_logs/tetris_rl_final.zip` → Uses final training weights
- `--model ./rl_logs/checkpoints/best/best_model.zip` → Uses best evaluation weights

Both commands load the model file you provide and use those exact weights for evaluation/visualization.
