# How Rewards Work in Tetris RL

## The Key Concept: Rewards After Each Piece Placement

In Tetris RL, **each timestep = one piece placement**, and we give a reward **immediately after each piece is placed**. This is called "shaped rewards" - we provide feedback for every action, not just at the end of the game.

## Timeline Example

Let's trace through a few pieces:

### Piece 1: I-piece placed
- **Action**: Agent chooses placement #5 (drops I-piece in column 3)
- **Result**: Piece lands, no lines cleared
- **Reward calculation**:
  - Lines cleared: 0 → **+0**
  - Score increase: +10 → **+0.1** (score * 0.01)
  - Height penalty: Stack height = 2 → **-0** (below threshold)
  - Holes: 0 → **-0**
  - **Total reward: +0.1** ✅ (small positive for surviving)

### Piece 2: O-piece placed
- **Action**: Agent places O-piece, clears 1 line!
- **Result**: 1 line cleared, score increases
- **Reward calculation**:
  - Lines cleared: 1 → **+100** 🎉 (big reward!)
  - Score increase: +500 → **+5**
  - Height improved → **+0**
  - **Total reward: +105** ✅✅✅

### Piece 3: T-piece placed poorly
- **Action**: Agent places T-piece, creates a hole
- **Result**: No lines cleared, hole created, stack getting high
- **Reward calculation**:
  - Lines cleared: 0 → **+0**
  - Score increase: +10 → **+0.1**
  - Height penalty: Stack height = 16 → **-2** (16 - 15 = 1, * 2.0)
  - Holes: 1 → **-5** (bad!)
  - Height variance: High → **-1.5**
  - **Total reward: -8.4** ❌ (penalty for bad placement)

### Piece 4: Game Over
- **Action**: Agent places piece, but playfield is full
- **Result**: Game ends
- **Reward calculation**:
  - Game over → **-100** 💀 (big penalty)
  - **Total reward: -100**

## Why This Works

### Sparse Rewards (Bad)
If we only gave rewards at game end:
- Agent gets **-100** for losing, **+1000** for winning
- But winning might take 100+ pieces
- Agent has no idea which pieces were good/bad
- **Very hard to learn!**

### Shaped Rewards (Good - What We Use)
We give rewards after **every piece**:
- Clear a line? **+100 immediately** ✅
- Create a hole? **-5 immediately** ❌
- Stack getting high? **-2 per unit** ⚠️
- **Agent learns quickly** which moves are good!

## Reward Breakdown

After each piece placement, we calculate:

```python
reward = 0.0

# Positive rewards
+ survival_bonus (if configured)      # Small reward for staying alive
+ line_cleared_reward                 # +100 to +800 depending on lines
+ score_increase * 0.01               # Small reward for score
+ level_up_bonus (if leveled up)      # +50

# Negative rewards (penalties)
- game_over_penalty                   # -100 if game ends
- height_penalty                      # -2 per unit above threshold
- hole_penalty                        # -5 per hole
- variance_penalty                     # -0.5 per unit of height variance
```

## Example: Full Episode

```
Piece 1: Place I-piece, no lines    → reward = +0.1
Piece 2: Place O-piece, clear 1     → reward = +105
Piece 3: Place T-piece, create hole → reward = -8.4
Piece 4: Place L-piece, no lines    → reward = +0.1
Piece 5: Place S-piece, clear 2     → reward = +305
Piece 6: Place J-piece, no lines    → reward = +0.1
Piece 7: Place Z-piece, game over  → reward = -100

Total episode reward: +302.9
```

## Why Timesteps Make Sense

- **1 timestep** = 1 piece placement = 1 reward calculation
- **1,000 timesteps** = 1,000 pieces placed = 1,000 reward signals
- Agent learns from **every single placement**, not just final score

## The Learning Process

1. **Early training**: Agent gets mostly negative rewards (bad placements)
2. **Learning**: Agent starts getting occasional positive rewards (clearing lines)
3. **Improving**: Agent learns to avoid holes, keep stacks low
4. **Mastery**: Agent consistently clears lines, avoids penalties

## From Your Training Output

```
ep_rew_mean | -3.5e+03
ep_len_mean | 18.3
```

This means:
- Average episode reward: **-3,500** (lots of penalties!)
- Average episode length: **18.3 pieces**
- Agent is still learning - getting negative rewards means it's making mistakes
- As training progresses, this should improve to positive rewards

## Key Takeaway

**Rewards are given after EVERY piece placement**, not just at game end. This "shaped reward" approach helps the agent learn much faster by providing immediate feedback on whether each move was good or bad.
