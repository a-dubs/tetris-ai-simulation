# Episode Lifecycle: How Game Over Works

## What Happens When the Game Ends

### Step-by-Step: Game Over Sequence

#### 1. Agent Places Final Piece
```
Agent chooses placement #3
Piece is placed...
```

#### 2. Game Over Detected
```python
# In env.py step() method:
game_over = not temp_engine.valid_location(active_tet)

# This becomes True when:
# - New piece spawns in an invalid position
# - Playfield is full, can't place piece
```

#### 3. Final Reward Calculated
```python
# calculate_reward() is called with done=True
reward = 0.0

# Game over penalty applied
if done:  # True!
    reward += -100.0  # Big penalty! 💀

# Plus any other rewards from this final piece:
# - Lines cleared? Maybe +100 if cleared lines
# - Score increase? +0.1
# - Height penalties? -2, -5, etc.

# Final reward: -100.0 (or -100 + other stuff)
```

#### 4. Episode Ends
```python
# step() returns:
obs = current_state_features
reward = -100.0  # (or whatever was calculated)
done = True      # ← This signals episode is over!
truncated = False
info = {"score": 500, "lines": 3, "level": 1, "game_over": True}
```

#### 5. Training Loop Detects Game Over
```python
# In stable-baselines3 training:
while training:
    obs, reward, done, truncated, info = env.step(action)
    
    if done:  # Game over!
        # Episode complete
        # Agent learns from this episode
        # Start new episode
        obs, info = env.reset()  # ← New game starts!
```

#### 6. New Episode Begins
```python
# env.reset() is called automatically:
def reset(self):
    self.engine = GameEngine()  # Fresh game state
    self.prev_state = self.engine.state
    self.current_placements = []
    
    obs = self._get_obs()  # Fresh observation
    info = self._get_info()
    
    return obs, info  # Ready for new episode!
```

## Complete Example: One Episode

```
Episode Start:
├─ reset() called
├─ Fresh playfield, new pieces spawned
└─ obs = [0, 0, 0, ..., 0] (initial state)

Piece 1:
├─ Agent chooses action=5
├─ Piece placed, no lines cleared
├─ reward = +0.1
└─ done = False → continue

Piece 2:
├─ Agent chooses action=12
├─ Piece placed, clears 1 line!
├─ reward = +105
└─ done = False → continue

Piece 3-17:
├─ ... more pieces ...
└─ done = False → continue

Piece 18 (FINAL):
├─ Agent chooses action=7
├─ Piece placed, but playfield full!
├─ game_over = True detected
├─ reward = -100.0 (game over penalty)
├─ done = True ← Episode ends!
└─ Training loop calls reset()

Episode End:
├─ Episode statistics recorded
├─ Agent learns from this episode
└─ reset() called → New episode starts
```

## Key Points

### 1. Game Over = Episode End
- When `game_over = True`, the episode ends
- The `done` flag signals this to the training algorithm
- Agent learns from the entire episode

### 2. Game Over Penalty
- **-100 reward** for losing (configurable)
- This is a strong negative signal
- Agent learns: "Don't let the game end!"

### 3. Automatic Reset
- After `done=True`, `reset()` is called automatically
- New episode starts immediately
- No manual intervention needed

### 4. Episode Statistics
- Each episode produces:
  - Total reward (sum of all step rewards)
  - Final score
  - Lines cleared
  - Episode length (number of pieces)
  - Whether game ended (always True for Tetris)

## Training Process

During training, the agent runs **thousands of episodes**:

```
Episode 1:  18 pieces, reward = -3,500, score = 500
Episode 2:  15 pieces, reward = -2,800, score = 300
Episode 3:  22 pieces, reward = -1,200, score = 800
Episode 4:  25 pieces, reward = +500, score = 1,200  ← Learning!
Episode 5:  30 pieces, reward = +1,200, score = 2,000 ← Getting better!
...
Episode 1000: 150 pieces, reward = +5,000, score = 15,000 ← Much better!
```

## What the Agent Learns

From game over:
- **Negative reward (-100)** teaches: "Avoid game over!"
- Agent learns to:
  - Keep stacks low
  - Avoid creating holes
  - Clear lines regularly
  - Survive longer

## From Your Training Output

```
ep_len_mean | 18.3     ← Average pieces before game over
ep_rew_mean | -3.5e+03 ← Average total reward per episode
```

This means:
- Agent plays ~18 pieces before game over
- Gets mostly negative rewards (making mistakes)
- As training progresses, episodes get longer and rewards become positive

## The Learning Cycle

```
1. Episode starts (reset)
2. Agent places pieces (steps)
3. Gets rewards after each piece
4. Game over (done=True)
5. Episode ends, agent learns
6. Repeat from step 1
```

The agent learns by experiencing **many episodes**, each ending in game over, gradually improving its strategy to survive longer and score higher.
