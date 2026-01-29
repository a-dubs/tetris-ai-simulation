# RL Test Recommendations

## Quick Test (Already Done ✅)
```bash
uv run python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config default \
    --timesteps 1000
```
**Result**: Works! But too short to see learning.

## Proper Test: See Real Learning (Recommended)

### Option 1: Short Learning Test (5-10 minutes)
```bash
uv run python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config default \
    --timesteps 50000 \
    --eval-freq 10000
```

**What to expect:**
- ~5-10 minutes of training
- ~50,000 timesteps (~2,700 episodes)
- You should see improvement in:
  - `ep_len_mean` increasing (surviving longer)
  - `ep_rew_mean` becoming less negative (better play)
  - Score increasing over time

**Monitor progress:**
```bash
# In another terminal:
tensorboard --logdir ./rl_logs/tensorboard
```
Then open http://localhost:6006 to see:
- Episode reward over time
- Episode length over time
- Training metrics

### Option 2: Medium Test (15-30 minutes)
```bash
uv run python -m tetris.rl.training train \
    --agent-config default \
    --scenario-config default \
    --timesteps 200000 \
    --eval-freq 20000
```

**What to expect:**
- ~15-30 minutes
- ~200,000 timesteps (~11,000 episodes)
- Clear learning curve visible
- Agent should start clearing lines regularly

### Option 3: Full Test (1-2 hours)
```bash
uv run python -m tetris.rl.training train \
    --agent-config large_network \
    --scenario-config aggressive_rewards \
    --timesteps 1000000 \
    --eval-freq 50000
```

**What to expect:**
- ~1-2 hours
- ~1,000,000 timesteps (~55,000 episodes)
- Strong performance
- Should be competitive with baseline AIs

## After Training: Evaluate Your Agent

### 1. Evaluate Performance
```bash
uv run python -m tetris.rl.training evaluate \
    --model ./rl_logs/tetris_rl_final.zip \
    --episodes 20
```

### 2. Visualize Agent Playing
```bash
uv run python -m tetris.rl.training visualize \
    --model ./rl_logs/tetris_rl_final.zip \
    --episodes 3
```

### 3. Compare with Baselines
```bash
uv run python -m tetris.rl.benchmark \
    --model ./rl_logs/tetris_rl_final.zip \
    --games 10
```

## What to Look For

### Early Training (First 10k timesteps)
- `ep_len_mean`: Should increase from ~18 to ~25-30
- `ep_rew_mean`: Should improve from -3,500 to -1,000 or better
- Agent learning basic survival

### Mid Training (50k-200k timesteps)
- `ep_len_mean`: Should reach 50-100+ pieces
- `ep_rew_mean`: Should become positive (+500 to +2,000)
- Agent clearing lines regularly
- Score increasing consistently

### Advanced Training (500k+ timesteps)
- `ep_len_mean`: 100-200+ pieces
- `ep_rew_mean`: +2,000 to +5,000+
- Agent playing strategically
- Competitive with rule-based AIs

## Recommended First Real Test

Start with **Option 1** (50k timesteps):

```bash
# Terminal 1: Start training
uv run python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config default \
    --timesteps 50000 \
    --eval-freq 10000

# Terminal 2: Monitor with TensorBoard
tensorboard --logdir ./rl_logs/tensorboard
```

Then watch TensorBoard at http://localhost:6006 to see:
- **rollout/ep_rew_mean** - Should trend upward
- **rollout/ep_len_mean** - Should increase
- **time/total_timesteps** - Training progress

This gives you a good sense of whether learning is happening without waiting too long!
