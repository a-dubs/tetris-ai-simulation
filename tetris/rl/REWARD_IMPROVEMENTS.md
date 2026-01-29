# RL Reward Function Improvements

## Problem Identified

The RL agent was dropping pieces straight down without learning proper Tetris strategy. Analysis revealed the reward function was too weak compared to the successful `fast.py` AI.

## What Changed

### 1. **Stronger Game Over Penalty**
- **Before**: -100.0
- **After**: -1000.0 (matching fast.py)
- **Why**: Agent needs to strongly avoid game-ending situations

### 2. **Exponential Stack Height Penalties**
- **Before**: Only penalized heights above row 15, linear penalty of 2.0 per unit
- **After**: Penalizes ALL heights exponentially:
  - Max height: `-8.0 * (height^1.5)` 
  - Average height: `-5.0 * (avg^1.2)`
  - Dangerous heights (above 15): `-4.0 * (danger^1.2)`
- **Why**: Encourages keeping the stack low from the start, not just avoiding danger

### 3. **Cliff Detection (Better Than Simple Hole Counting)**
- **Before**: Simple hole counting (penalty of 5.0 per hole)
- **After**: Cliff penalties that detect the structure creating holes:
  - Horizontal cliffs (overhangs): `-6.0 * (length^1.3)`
  - Vertical cliffs (holes): `-7.0 * (depth^1.5)`
- **Why**: Detects the patterns that create holes, not just counts them after the fact

### 4. **Enhanced Line Clear Rewards**
- Added Tetris bonus: +300 extra for 4-line clears
- **Why**: Makes Tetris clears more attractive, encouraging better stacking

## Current Reward Structure

### Positive Rewards
- **Line clears**: 100/300/500/800 for 1/2/3/4 lines
- **Tetris bonus**: +300 extra for 4-line clear
- **Score**: 0.01x score increase
- **Level up**: +50

### Negative Rewards (Penalties)
- **Game over**: -1000 (very strong!)
- **Max stack height**: `-8.0 * (height^1.5)` (exponential)
- **Average stack height**: `-5.0 * (avg^1.2)` (exponential)
- **Stack danger**: `-4.0 * (danger^1.2)` for heights above row 15
- **Horizontal cliffs**: `-6.0 * (length^1.3)` (overhangs)
- **Vertical cliffs**: `-7.0 * (depth^1.5)` (holes)

## How This Matches Fast.py AI

The improved rewards now match the philosophy of `fast.py`:
1. **Strong penalties** discourage bad play immediately
2. **Exponential penalties** make small mistakes costly
3. **Cliff detection** identifies structural problems (holes/cliffs) better than simple counting
4. **Low stack priority** encourages keeping the board clean

## Expected Behavior Changes

With these improvements, the RL agent should:
1. ✅ Learn to rotate pieces (to avoid cliffs/holes)
2. ✅ Learn to position pieces horizontally (to avoid overhangs)
3. ✅ Prioritize keeping the stack low
4. ✅ Avoid creating holes/cliffs
5. ✅ Go for line clears when possible

## Testing

Retrain with the improved rewards:
```bash
uv run python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config default \
    --timesteps 200000
```

The agent should show much better learning behavior, learning to rotate and position pieces properly rather than just dropping them straight down.
