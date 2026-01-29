# RL Quick Start Guide

This guide will help you get started with reinforcement learning for Tetris.

## Installation

Install the RL dependencies:

```bash
# Basic RL dependencies
uv pip install -e ".[rl]"

# Or with full visualization tools
uv pip install -e ".[rl-full]"
```

## Quick Start

### 1. Train an RL Agent

Train a basic agent for 100k timesteps:

```bash
python -m tetris.rl.training train --timesteps 100000
```

Train with custom parameters:

```bash
python -m tetris.rl.training train \
    --timesteps 1000000 \
    --lr 3e-4 \
    --batch-size 64 \
    --model-name my_tetris_agent \
    --log-dir ./my_training_logs \
    --checkpoint-freq 50000 \
    --eval-freq 10000
```

### 2. Evaluate Trained Agent

Evaluate a trained model:

```bash
python -m tetris.rl.training evaluate \
    --model ./rl_logs/checkpoints/tetris_rl_best.zip \
    --episodes 10
```

Evaluate with rendering:

```bash
python -m tetris.rl.training evaluate \
    --model ./rl_logs/checkpoints/tetris_rl_best.zip \
    --episodes 5 \
    --render
```

### 3. Visualize Agent Playing

Watch the agent play:

```bash
python -m tetris.rl.training visualize \
    --model ./rl_logs/checkpoints/tetris_rl_best.zip \
    --episodes 3
```

### 4. View Training Progress

Plot training metrics:

```bash
python -m tetris.rl.visualization plot \
    --metrics-file ./rl_logs/metrics.json \
    --output training_plot.png
```

View statistics:

```bash
python -m tetris.rl.visualization stats \
    --recent 100
```

Create interactive dashboard:

```bash
python -m tetris.rl.visualization dashboard \
    --metrics-file ./rl_logs/metrics.json \
    --output dashboard.html
```

### 5. View TensorBoard Logs

Monitor training in real-time:

```bash
tensorboard --logdir ./rl_logs/tensorboard
```

Then open http://localhost:6006 in your browser.

## Training Tips

### Hyperparameter Tuning

Start with default parameters, then experiment:

- **Learning Rate**: Try `1e-4` to `1e-3`
- **Batch Size**: Try `32`, `64`, `128`
- **Gamma (discount)**: Try `0.95` to `0.99`
- **n_steps**: Try `1024`, `2048`, `4096`

### Reward Function Tuning

The reward function in `tetris/rl/env.py` can be customized:

- Adjust line clearing rewards (currently 100, 300, 500, 800)
- Modify height penalties
- Change hole penalties
- Add rewards for specific strategies

### Training Time Estimates

- **10k timesteps**: ~15-30 minutes (basic performance)
- **100k timesteps**: ~2-4 hours (decent performance)
- **1M timesteps**: ~1-2 days (strong performance)

### Monitoring Progress

Watch for:
- **Increasing average reward**: Agent is learning
- **Increasing score/lines**: Agent is improving
- **Stable episode length**: Agent is consistent
- **Plateau**: May need hyperparameter adjustment

## Architecture Overview

### Environment (`TetrisEnv`)

- **Action Space**: Discrete (choose placement, max 20 actions)
- **Observation Space**: Box (23 features)
- **Reward**: Shaped reward function

### State Features

The agent receives ~23 engineered features:
- Column heights (10)
- Aggregate statistics (4)
- Holes count (1)
- Active piece info (4)
- Next piece (1)
- Game progress (3)

### Reward Structure

- **+100-800**: Lines cleared (exponential)
- **+50**: Level progression
- **+score*0.01**: Score increase
- **-100**: Game over
- **-height*2**: High stacks (if >15)
- **-holes*5**: Holes penalty
- **-variance*0.5**: Height variance penalty

## Example Workflow

### Complete Training Session

```bash
# 1. Start training
python -m tetris.rl.training train \
    --timesteps 500000 \
    --model-name experiment1 \
    --log-dir ./experiments/exp1

# 2. In another terminal, monitor with TensorBoard
tensorboard --logdir ./experiments/exp1/tensorboard

# 3. After training, evaluate
python -m tetris.rl.training evaluate \
    --model ./experiments/exp1/checkpoints/tetris_rl_best.zip \
    --episodes 20

# 4. Visualize best episodes
python -m tetris.rl.training visualize \
    --model ./experiments/exp1/checkpoints/tetris_rl_best.zip \
    --episodes 5

# 5. Generate plots
python -m tetris.rl.visualization plot \
    --metrics-file ./experiments/exp1/metrics.json \
    --output exp1_progress.png
```

## Troubleshooting

### "gymnasium not installed"

Install dependencies:
```bash
uv pip install -e ".[rl]"
```

### "CUDA out of memory"

Reduce batch size or use CPU:
- Add `device="cpu"` in model creation
- Or reduce `--batch-size`

### Training is slow

- Use headless mode (no rendering during training)
- Reduce `--eval-freq` and `--checkpoint-freq`
- Use vectorized environments (already implemented)

### Agent not learning

- Check reward function (should be balanced)
- Try different learning rates
- Increase training time
- Check if observations are normalized properly

### Visualization not working

- Ensure pygame is installed: `uv pip install pygame`
- Check display settings (for headless servers)
- Try `render_mode=None` for headless training

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Customize reward function**: Adjust rewards for your goals
3. **Add features**: Extend state representation
4. **Try other algorithms**: DQN, A2C, SAC
5. **Curriculum learning**: Start easy, increase difficulty
6. **Compare with baselines**: Benchmark against greedy/random AI

## Files Overview

- `tetris/rl/env.py`: Core RL environment and utilities
- `tetris/rl/training.py`: Main training script
- `tetris/rl/visualization.py`: Visualization tools
- `RL_FEASIBILITY_ANALYSIS.md`: Detailed analysis and architecture

## Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
