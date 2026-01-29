# Weight Optimization for FastAI Heuristics

This document describes approaches for iteratively improving the heuristic weights used by the FastAI agent. The FastAI agent uses a set of 14 weight parameters that control how different board features are evaluated. While the heuristic structure is solid, finding optimal weights is challenging.

## Overview

**Important**: All optimization methods follow the same pattern:
1. **Set weights** (fixed for entire game)
2. **Play full game** from start to game over
3. **Get final score/performance**
4. **Adjust weights** based on performance
5. **Repeat** with new weights

**No weight adjustments occur during gameplay** - weights are constant for the entire duration of each game.

The FastAI agent evaluates placements using these weighted features:
- **Stack height penalties**: Keep the stack low (`m_stack_w`, `m_stack_e`, `a_stack_w`, `a_stack_e`)
- **Cliff penalties**: Avoid creating holes (`cliff_l_w`, `cliff_l_e`, `cliff_h_w`, `cliff_h_e`)
- **Stack danger**: Penalize high stacks (`stack_d_w`, `stack_d_e`, `stack_d_thresh`)
- **Score rewards**: Prioritize line clears (`score_w`, `tetris_bonus`)
- **Game over penalty**: Strong penalty for losing (`go_w`)

## Approaches

### 1. Evolutionary Strategy (CMA-ES) ⭐ Recommended for Quick Results

**Best for**: Finding good weights quickly with minimal setup

Uses Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to optimize weights. This is a gradient-free optimization method that works well for black-box optimization problems.

**Pros**:
- Fast convergence
- No gradients needed
- Handles noisy evaluations well
- Good for 10-50 parameter optimization

**Cons**:
- Requires many evaluations (games)
- Can get stuck in local optima

**Usage**:
```bash
# Install dependencies
uv pip install cma scikit-optimize
# Or: uv sync --extra optimization

# Run optimization (50 iterations, 20 games per evaluation)
uv run python -m tetris.ai.weight_optimization evolve \
    --runs 50 \
    --games-per-run 20 \
    --population-size 10 \
    --seed 42 \
    --output optimized_weights.json
```

**Expected time**: ~10-30 minutes depending on games per run

### 2. Bayesian Optimization ⭐ Best for Sample Efficiency

**Best for**: When you want to minimize the number of games played

Uses Gaussian Process-based Bayesian Optimization to intelligently explore the weight space. More sample-efficient than random search or grid search.

**Pros**:
- Very sample-efficient (needs fewer evaluations)
- Handles uncertainty well
- Good exploration/exploitation balance

**Cons**:
- Slower per iteration (GP fitting)
- May struggle with >20 parameters

**Usage**:
```bash
# Install dependencies (if not already installed)
uv pip install scikit-optimize
# Or: uv sync --extra optimization

# Run optimization (30 iterations, 20 games per iteration)
uv run python -m tetris.ai.weight_optimization bayesian \
    --iterations 30 \
    --games-per-iter 20 \
    --seed 42 \
    --output optimized_weights.json
```

**Expected time**: ~15-45 minutes

### 3. Meta-RL (RL Learning Weights)

**Best for**: When you want RL to learn weight selection over time

Creates an episodic RL environment where:
- Each episode: Agent sets weights → Play full game(s) → Get score → Episode ends
- State: Previous weights + previous performance
- Action: New weight values to try
- Reward: Performance from playing full game(s) with those weights

**Pros**:
- Can learn complex weight selection policies
- Adapts over training
- Could generalize to different scenarios

**Cons**:
- Slower than direct optimization
- More complex setup
- Requires RL dependencies

**Usage**:
```bash
# Install RL dependencies (if not already installed)
uv sync --extra rl
# Or: uv pip install gymnasium stable-baselines3

# Train meta-RL agent
uv run python -m tetris.ai.weight_optimization meta_rl \
    --timesteps 100000 \
    --games-per-eval 5 \
    --seed 42 \
    --output optimized_weights.json
```

**Expected time**: ~30-60 minutes

### 4. Hybrid RL (RL Using FastAI Features)

**Best for**: Learning placement selection while leveraging FastAI's domain knowledge

Instead of optimizing weights, this approach uses FastAI's heuristic features as input to an RL agent. The RL agent learns to select placements using FastAI's feature representation.

**Pros**:
- Combines FastAI domain knowledge with RL learning
- Can learn non-linear weight combinations
- More flexible than fixed weights

**Cons**:
- Requires training from scratch
- More complex than weight optimization
- May not outperform optimized FastAI

**Usage**:
```bash
# Train hybrid RL agent
uv run python -m tetris.ai.hybrid_rl \
    --timesteps 500000 \
    --learning-rate 3e-4 \
    --model-name hybrid_rl \
    --log-dir ./rl_logs \
    --seed 42

# Or without heuristic features (pure RL)
uv run python -m tetris.ai.hybrid_rl \
    --timesteps 500000 \
    --no-heuristic-features
```

**Expected time**: ~1-2 hours for good performance

## Comparison

| Approach | Setup Time | Training Time | Best For |
|----------|------------|---------------|----------|
| CMA-ES | Low | 10-30 min | Quick optimization |
| Bayesian | Low | 15-45 min | Sample efficiency |
| Meta-RL | Medium | 30-60 min | Adaptive weights |
| Hybrid RL | Medium | 1-2 hours | Feature learning |

## Using Optimized Weights

After optimization, you can use the weights in FastAI:

```python
import json
from tetris.ai.fast import FastAIPlayer

# Load optimized weights
with open("optimized_weights.json", "r") as f:
    data = json.load(f)
    optimized_params = data["params"]

# Create FastAI with optimized weights
ai = FastAIPlayer(params=optimized_params)

# Use in simulation
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.core.game_engine import GameEngine
from tetris.simulation.factory import create_renderer

config = SimulationConfig(
    ai_type="fast",
    headless=False,
    max_moves=10000,
)
engine = GameEngine()
renderer = create_renderer(False)
sim = Simulator(engine, ai, renderer, config)
result = sim.run()
```

Or via command line:
```bash
# Benchmark with optimized weights
uv run python main.py benchmark \
    --ai fast \
    --runs 20 \
    --seed 42 \
    --params-file optimized_weights.json
```

## Weight Bounds

The optimization respects these bounds (defined in `WeightBounds`):

```python
m_stack_w: (-15.0, -1.0)      # Max stack weight
m_stack_e: (0.5, 3.0)          # Max stack exponent
a_stack_w: (-10.0, -1.0)       # Avg stack weight
a_stack_e: (0.5, 3.0)          # Avg stack exponent
cliff_l_w: (-10.0, -1.0)       # Cliff length weight
cliff_l_e: (0.5, 3.0)          # Cliff length exponent
cliff_h_w: (-12.0, -1.0)       # Cliff height weight
cliff_h_e: (0.5, 3.0)          # Cliff height exponent
stack_d_w: (-8.0, -1.0)        # Stack danger weight
stack_d_e: (0.5, 3.0)          # Stack danger exponent
stack_d_thresh: (10, 20)       # Stack danger threshold
score_w: (50.0, 200.0)         # Score weight
tetris_bonus: (100.0, 500.0)   # Tetris bonus
go_w: (-2000.0, -100.0)        # Game over weight
```

## Evaluation Metrics

All approaches use a fitness function that combines:
- **Score** (50% weight): Average game score
- **Lines cleared** (40% weight): Average lines cleared
- **Moves survived** (10% weight): Average moves before game over

The fitness is normalized to balance these metrics:
```python
fitness = 0.5 * (score / 10000) + 0.4 * (lines / 50) + 0.1 * (moves / 1000)
```

## Tips

1. **Start with CMA-ES**: It's the fastest way to get improved weights
2. **Use more games per evaluation**: More games = less noisy fitness, better optimization
3. **Run multiple seeds**: Different seeds may find different local optima
4. **Compare with baseline**: Always benchmark optimized weights against defaults
5. **Hybrid RL for exploration**: If you want to learn beyond fixed weights, try Hybrid RL

## Example Workflow

```bash
# 1. Quick optimization with CMA-ES
uv run python -m tetris.ai.weight_optimization evolve \
    --runs 50 \
    --games-per-run 20 \
    --output weights_cmaes.json

# 2. Benchmark optimized weights
uv run python main.py benchmark \
    --ai fast \
    --runs 20 \
    --seed 42 \
    --params-file weights_cmaes.json \
    --output-file benchmark_optimized.json

# 3. Compare with baseline
uv run python main.py benchmark \
    --ai fast \
    --runs 20 \
    --seed 42 \
    --output-file benchmark_baseline.json

# 4. If results are good, use in production
# Update FastAI defaults or use params file
```

## Troubleshooting

**Optimization is slow**:
- Reduce `games_per_run` or `games_per_iter`
- Use fewer iterations/runs
- Try Bayesian optimization (more sample-efficient)

**Results are noisy**:
- Increase `games_per_run` or `games_per_iter`
- Use a fixed seed for reproducibility
- Run multiple optimizations and average

**Weights don't improve**:
- Check weight bounds (may be too restrictive)
- Try different initial weights
- Consider that defaults may already be near-optimal

**Meta-RL not converging**:
- Increase `games_per_eval`
- Adjust learning rate
- Try longer training (`--timesteps`)

## Future Improvements

Potential enhancements:
1. **Multi-objective optimization**: Optimize for score, lines, and survival separately
2. **Transfer learning**: Use weights from one level to initialize another
3. **Ensemble methods**: Combine multiple optimized weight sets
4. **Online learning**: Adapt weights during gameplay
5. **Differentiable heuristics**: Make evaluation differentiable for gradient-based optimization
