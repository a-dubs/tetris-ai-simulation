# Quickest RL Test

## Fastest Test (30 seconds)

```bash
uv run python -m tetris.rl.test_env
```

This runs:
- Import checks
- Environment creation
- Feature extraction
- Reward calculation
- Mini training step (100 timesteps)

## Even Quicker - Just Test Environment (10 seconds)

```python
uv run python -c "from tetris.rl.env import TetrisEnv; env = TetrisEnv(); obs, _ = env.reset(); print('✓ Works!', obs.shape)"
```

## Quick Training Test (2-3 minutes)

```bash
uv run python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config default \
    --timesteps 1000
```

Then check TensorBoard:
```bash
tensorboard --logdir ./rl_logs/tensorboard
```
