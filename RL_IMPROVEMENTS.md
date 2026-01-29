# RL Environment Improvements

This document describes the recent improvements to the Tetris RL environment, implementing best practices for RL-based Tetris agents.

## Overview

The RL environment has been enhanced with:
1. **Multi-channel board observations** for CNN-based policies
2. **Next queue support** (configurable number of pieces ahead)
3. **Simplified reward function** for easier training
4. **Per-piece placement actions** (already implemented, now improved)
5. **Configurable observation modes**

## Observation Modes

### 1. Feature-based Observations (`observation_mode="features"`)

The default mode uses handcrafted features:
- Column heights (10 features)
- Aggregate statistics (4 features)
- Holes count (1 feature)
- Active piece info (4 features)
- Next queue (N × 1 feature per piece)
- Game progress (3 features)

**Total:** ~23 + N features (where N is `next_queue_size`)

**Best for:** MLP policies, fast training, interpretability

### 2. Multi-channel Board Observations (`observation_mode="board_channels"`)

Image-like representation with separate channels:
- **Channel 0:** Locked blocks (0/1)
- **Channel 1:** Active piece mask (0/1)
- **Channel 2:** Ghost piece landing mask (0/1) [optional]

Plus:
- Current piece one-hot encoding (7 values)
- Next queue one-hot encodings (N × 7 values)

**Total:** `(H × W × C) + 7 + (N × 7)` where:
- H = `TOTAL_PLAYFIELD_HEIGHT` (24)
- W = `PLAYFIELD_WIDTH` (10)
- C = 2 or 3 (depending on `include_ghost`)
- N = `next_queue_size`

**Best for:** CNN policies, learning spatial patterns

## Action Space

The environment uses **per-piece placement** actions (recommended approach):

- Action = index into list of valid placements for current piece
- Environment automatically executes the sequence of moves to reach that placement
- Action space size is configurable via `max_placements` parameter
- Invalid actions are wrapped to valid range (prevents episode termination)

**Benefits:**
- Much shorter horizon (one decision per piece vs many ticks)
- Cleaner credit assignment
- Easier to train
- Still strategically rich

## Reward Functions

### Minimal Reward (`use_minimal_reward=True`) ⭐ **SIMPLEST**

**Ultra-minimal reward: ONLY stack height and holes.**

```python
reward = (
    aggregate_height * stack_height_penalty    # Penalty for high stacks
    + holes * holes_penalty                    # Penalty for holes
    + game_over_penalty (if done)              # Terminal penalty
)
```

**Default parameters:**
- `stack_height_penalty`: -0.1 per unit of aggregate height
- `holes_penalty`: -1.0 per hole
- `game_over`: -100.0

**Key insight:** No explicit reward for clearing lines! The agent must learn that clearing lines is good because it reduces stack height and holes.

**Best for:** 
- Simplest possible reward structure
- Understanding if agent can learn line clearing implicitly
- Debugging reward shaping issues
- Research into emergent behavior

### Simplified Reward (`use_simple_reward=True`)

Clean, interpretable reward structure:

```python
reward = (
    lines_cleared * lines_weight           # Base reward per line
    + (tetris_bonus if 4-line clear)       # Extra bonus for Tetris
    + new_holes * holes_penalty            # Penalty for creating holes
    + aggregate_height * height_penalty    # Penalty for stack height
    + game_over_penalty (if done)          # Large penalty on game over
)
```

**Default parameters:**
- `lines_cleared`: 1.0 per line
- `tetris_bonus`: 2.0x multiplier for 4-line clear
- `holes_penalty`: -0.5 per new hole
- `aggregate_height_penalty`: -0.01 per unit of aggregate height
- `game_over`: -1000.0

**Best for:** Getting started, debugging, interpretability

### Advanced Reward (`use_simple_reward=False`, `use_minimal_reward=False`)

More sophisticated reward with:
- Exponential stack height penalties
- Cliff detection (horizontal and vertical)
- Stack danger zones
- Level progression bonuses

**Best for:** Fine-tuning performance, matching heuristic AI behavior

## Usage Examples

### Basic Training with Features

```python
from tetris.rl.env import TetrisEnv

env = TetrisEnv(
    observation_mode="features",
    next_queue_size=5,
    use_simple_reward=True,
    max_placements=50,
)
```

### CNN Training with Board Channels

```python
env = TetrisEnv(
    observation_mode="board_channels",
    next_queue_size=5,
    include_ghost=True,
    use_simple_reward=True,
    max_placements=50,
)

# Use CnnPolicy with stable-baselines3
from stable_baselines3 import PPO
model = PPO("CnnPolicy", env, ...)
```

### Minimal Reward Training (Simplest Possible)

```python
env = TetrisEnv(
    observation_mode="features",
    use_minimal_reward=True,  # ONLY stack height + holes
    next_queue_size=5,
    max_placements=50,
)

# Customize minimal reward parameters
env = TetrisEnv(
    use_minimal_reward=True,
    reward_params={
        "stack_height_penalty": -0.05,  # Lighter penalty
        "holes_penalty": -0.5,           # Lighter penalty
        "game_over": -50.0,              # Smaller terminal penalty
    }
)
```

### Custom Reward Parameters

```python
env = TetrisEnv(
    reward_params={
        "lines_cleared": 2.0,      # Higher reward for lines
        "holes_penalty": -1.0,     # Stronger hole penalty
        "aggregate_height_penalty": -0.02,
        "game_over": -2000.0,
    },
    use_simple_reward=True,
)
```

## Design Rationale

### Why Per-Piece Placement?

1. **Shorter horizon:** One decision per piece vs many ticks per piece
2. **Better credit assignment:** Reward directly relates to placement choice
3. **Easier training:** Less exploration needed, faster convergence
4. **Still strategic:** All strategic decisions (where to place) are preserved

### Why Multi-Channel Board?

1. **CNN-friendly:** Separate channels help CNNs learn spatial patterns
2. **Ghost piece:** Helps agent understand landing position
3. **Interpretable:** Can visualize what agent sees
4. **Proven:** Common in successful RL Tetris implementations

### Why Simplified Reward?

1. **Easier to tune:** Fewer hyperparameters
2. **Interpretable:** Clear relationship between actions and rewards
3. **Stable:** Less risk of reward explosion or vanishing gradients
4. **Good baseline:** Can always add complexity later

## Next Steps

1. **Try both observation modes:** Compare MLP vs CNN performance
2. **Tune reward parameters:** Adjust based on training behavior
3. **Experiment with next_queue_size:** More pieces = more planning, but larger observation space
4. **Add hold support:** Future enhancement for more strategic play

## References

- [Tetris RL Best Practices](https://github.com/your-repo/docs)
- Stable-Baselines3 documentation
- Gymnasium environment API
