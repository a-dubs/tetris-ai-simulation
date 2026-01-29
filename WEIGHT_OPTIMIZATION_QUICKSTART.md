# Weight Optimization Quick Start

Quick guide to optimizing FastAI heuristic weights.

## Quick Start: CMA-ES (Recommended)

Fastest way to get improved weights:

```bash
# Install dependencies
uv pip install cma scikit-optimize

# Or install the optimization optional dependency group:
uv sync --extra optimization

# Run optimization (takes ~10-30 minutes)
uv run python -m tetris.ai.weight_optimization evolve \
    --runs 50 \
    --games-per-run 20 \
    --output optimized_weights.json

# Test the optimized weights
uv run python main.py benchmark \
    --ai fast \
    --runs 20 \
    --seed 42 \
    --params-file optimized_weights.json
```

## What Gets Optimized?

The FastAI agent uses 14 weight parameters:
- Stack height penalties (4 params)
- Cliff penalties (4 params)  
- Stack danger (3 params)
- Score rewards (2 params)
- Game over penalty (1 param)

These control how the AI evaluates placements. The optimization finds weights that maximize:
- Score (50% weight)
- Lines cleared (40% weight)
- Moves survived (10% weight)

## Other Methods

### Bayesian Optimization (More Sample-Efficient)
```bash
# Dependencies already installed above
uv run python -m tetris.ai.weight_optimization bayesian \
    --iterations 30 \
    --games-per-iter 20 \
    --output optimized_weights.json
```

### Meta-RL (RL Learns Weights)
```bash
# Install RL dependencies if not already installed
uv sync --extra rl

uv run python -m tetris.ai.weight_optimization meta_rl \
    --timesteps 100000 \
    --games-per-eval 5 \
    --output optimized_weights.json
```

### Hybrid RL (RL Uses FastAI Features)
```bash
# Install RL dependencies if not already installed
uv sync --extra rl

uv run python -m tetris.ai.hybrid_rl \
    --timesteps 500000 \
    --model-name hybrid_rl
```

## Using Optimized Weights

After optimization, use weights in any command:

```bash
# Run single game
uv run python main.py run \
    --ai fast \
    --params-file optimized_weights.json

# Benchmark
uv run python main.py benchmark \
    --ai fast \
    --runs 20 \
    --params-file optimized_weights.json
```

## Tips

1. **More games = better optimization**: Increase `--games-per-run` for less noisy results
2. **Multiple seeds**: Run with different seeds to find different local optima
3. **Compare baselines**: Always benchmark optimized weights vs defaults
4. **Start with CMA-ES**: It's the fastest way to get improvements

See `WEIGHT_OPTIMIZATION.md` for detailed documentation.
