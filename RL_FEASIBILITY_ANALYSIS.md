# Reinforcement Learning Feasibility Analysis

## Executive Summary

**RL integration is highly feasible** for this Tetris project. The current architecture provides an excellent foundation with clean separation of concerns, fast simulation capabilities, and a well-defined state/action space. With the performance optimizations already in place, training an RL agent should be practical and efficient.

**Estimated effort:** 2-4 weeks for a basic RL implementation with visualization.

---

## Current Architecture Assessment

### ✅ Strengths for RL Integration

1. **Clean State Representation**
   - `GameState` is immutable and well-structured
   - Contains all necessary information: playfield, active/next tetrimino, score, level
   - Easy to convert to feature vectors for neural networks

2. **Fast Simulation**
   - Headless renderer for training (no GUI overhead)
   - Optimized game engine with efficient state transitions
   - Performance analysis shows 50-100x speedup potential already identified

3. **Clear Action Space**
   - Discrete actions: `["left", "right", "cw", "ccw", "drop"]`
   - Action sequences handled via `get_move_sequence()` interface
   - Can work at placement level (choose placement) or move level (choose individual moves)

4. **Reward Signal Available**
   - Score, lines cleared, level progression
   - Game over condition
   - Can derive intermediate rewards (lines cleared per piece, height penalties, etc.)

5. **Modular Design**
   - `AIPlayer` abstract base class - easy to add RL agent
   - `Simulator` orchestrates everything cleanly
   - Renderer abstraction allows visualization during/after training

---

## RL Implementation Approach

### Option 1: Placement-Level Actions (Recommended)

**Action Space:** Choose from all valid placements for current piece
- Use `find_all_placements()` from `placement.py`
- Action = index into placements list
- Much smaller action space (~10-20 actions per state vs thousands of move sequences)

**Advantages:**
- Smaller action space = faster learning
- More meaningful actions (complete placements vs individual moves)
- Leverages existing placement finding logic

**Implementation:**
```python
class RLAIPlayer(AIPlayer):
    def __init__(self, model, device='cpu'):
        self.model = model  # Neural network
        self.device = device
        
    def get_move_sequence(self, state: GameState) -> List[str]:
        # Find all valid placements
        placements = find_all_placements(state)
        
        # Convert state to feature vector
        features = self._state_to_features(state)
        
        # Get action from model
        action_probs = self.model(features)
        action_idx = action_probs.argmax().item()
        
        # Return move sequence for chosen placement
        return placements[action_idx].path
```

### Option 2: Move-Level Actions

**Action Space:** Choose next move directly
- Actions: `["left", "right", "cw", "ccw", "drop"]`
- Agent decides move-by-move

**Advantages:**
- More granular control
- Can learn intermediate positioning strategies

**Disadvantages:**
- Larger action space
- Longer sequences = more steps per piece
- Harder to learn (sparse rewards)

---

## State Representation for RL

### Feature Vector Design

**Option A: Raw Playfield (Simple)**
```python
def state_to_features(state: GameState) -> np.ndarray:
    # Playfield: 10x24 = 240 features (binary)
    playfield = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in state.playfield
    ]).flatten()
    
    # Active tetrimino: position + orientation = 3 features
    active = [state.active_tetrimino.x, 
              state.active_tetrimino.y,
              orientation_to_int(state.active_tetrimino.orientation)]
    
    # Next tetrimino: shape = 1 feature (one-hot encoded = 7 features)
    next_shape = shape_to_int(state.next_tetrimino.shape)
    
    # Game stats: 3 features
    stats = [state.level, state.lines_cleared, state.score / 1000.0]
    
    return np.concatenate([playfield, active, [next_shape], stats])
    # Total: ~250 features
```

**Option B: Engineered Features (Recommended)**
```python
def state_to_features(state: GameState) -> np.ndarray:
    """Extract meaningful features from game state."""
    features = []
    
    # Playfield statistics
    pf = np.array([[1 if c != ' ' else 0 for c in row] for row in state.playfield])
    
    # Column heights (10 features)
    column_heights = [np.max(np.where(pf[:, i] == 1)[0]) if np.any(pf[:, i]) else 0 
                      for i in range(10)]
    features.extend(column_heights)
    
    # Aggregate statistics
    features.append(np.mean(column_heights))  # Average height
    features.append(np.max(column_heights))   # Max height
    features.append(np.std(column_heights))    # Height variance
    features.append(np.sum(pf))               # Total blocks
    
    # Holes (empty cells with blocks above)
    holes = count_holes(pf)
    features.append(holes)
    
    # Active piece features
    active = state.active_tetrimino
    features.extend([
        active.x,
        active.y,
        orientation_to_int(active.orientation),
        shape_to_int(active.shape)
    ])
    
    # Next piece
    features.append(shape_to_int(state.next_tetrimino.shape))
    
    # Game progress
    features.extend([
        state.level,
        state.lines_cleared,
        state.score / 10000.0  # Normalized
    ])
    
    return np.array(features, dtype=np.float32)
    # Total: ~25-30 features (much more efficient!)
```

---

## Reward Function Design

### Recommended Reward Structure

```python
def calculate_reward(state: GameState, prev_state: GameState, 
                    action: int, done: bool) -> float:
    """Calculate reward for RL agent."""
    reward = 0.0
    
    # Primary rewards
    if done:
        reward -= 100.0  # Large penalty for game over
    
    # Lines cleared (most important!)
    lines_cleared = state.lines_cleared - prev_state.lines_cleared
    if lines_cleared > 0:
        # Reward increases exponentially with lines cleared
        reward += [100, 300, 500, 800][min(lines_cleared - 1, 3)]
    
    # Score increase
    score_delta = state.score - prev_state.score
    reward += score_delta * 0.01  # Small reward for score
    
    # Level progression
    if state.level > prev_state.level:
        reward += 50.0
    
    # Shape penalties (encourage good play)
    pf = state.playfield
    column_heights = [max_height(pf, col) for col in range(10)]
    max_height = max(column_heights)
    
    # Penalize high stacks
    if max_height > 15:
        reward -= (max_height - 15) * 2.0
    
    # Penalize holes
    holes = count_holes(pf)
    reward -= holes * 5.0
    
    # Penalize height variance (encourage flat playfield)
    height_variance = np.std(column_heights)
    reward -= height_variance * 0.5
    
    return reward
```

---

## RL Algorithm Recommendations

### Option 1: PPO (Proximal Policy Optimization) - **Recommended**

**Why:**
- Stable, sample-efficient
- Works well with discrete action spaces
- Good for continuous training
- Standard implementation available (Stable-Baselines3)

**Implementation:**
```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create Tetris environment
env = TetrisEnv()

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10)
model.learn(total_timesteps=1_000_000)
```

### Option 2: DQN (Deep Q-Network)

**Why:**
- Classic RL algorithm
- Good for discrete action spaces
- Well-understood and documented

**Considerations:**
- Requires experience replay buffer
- Can be less stable than PPO
- Good for initial experiments

### Option 3: A3C/A2C (Actor-Critic)

**Why:**
- On-policy, can learn faster
- Good for parallel training
- Moderate complexity

---

## Training Infrastructure

### Environment Wrapper

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TetrisEnv(gym.Env):
    """Gym environment for Tetris RL training."""
    
    def __init__(self):
        super().__init__()
        
        # Action space: choose placement (discrete)
        # Will be set dynamically based on available placements
        self.action_space = spaces.Discrete(20)  # Max ~20 placements
        
        # Observation space: feature vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(30,), dtype=np.float32
        )
        
        # Initialize game
        self.engine = GameEngine()
        self.prev_state = self.engine.state
        
    def reset(self, seed=None):
        """Reset environment to initial state."""
        self.engine = GameEngine()
        self.prev_state = self.engine.state
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute action and return (obs, reward, done, truncated, info)."""
        # Get all placements
        placements = find_all_placements(self.engine.state)
        
        if not placements or action >= len(placements):
            # Invalid action - end episode
            return self._get_obs(), -100.0, True, False, {}
        
        # Execute placement
        placement = placements[action]
        for move in placement.path:
            self.engine.state = self.engine.apply_move(move)
            if move == "drop":
                # Place piece and update playfield
                new_state, lines_cleared = self.engine.update_playfield(
                    self.engine.state.active_tetrimino
                )
                # Spawn next piece, update score, etc.
                self.engine.state = self._spawn_next_piece(new_state, lines_cleared)
        
        # Calculate reward
        reward = self._calculate_reward(self.prev_state, self.engine.state)
        done = self.engine.state.game_over
        
        self.prev_state = self.engine.state
        
        return self._get_obs(), reward, done, False, {
            "score": self.engine.state.score,
            "lines": self.engine.state.lines_cleared,
            "level": self.engine.state.level
        }
    
    def _get_obs(self):
        """Convert game state to observation vector."""
        return state_to_features(self.engine.state)
    
    def _calculate_reward(self, prev_state, current_state):
        """Calculate reward (see reward function above)."""
        # Implementation from reward section
        ...
```

---

## Visualization During Training

### Option 1: Periodic Visualization (Recommended)

**Strategy:** Render games periodically during training

```python
def train_with_visualization(model, total_steps=1_000_000, 
                             render_every=10_000):
    """Train RL agent with periodic visualization."""
    
    for step in range(total_steps):
        # Training step
        model.learn(total_timesteps=1000, reset_num_timesteps=False)
        
        # Visualize every N steps
        if step % render_every == 0:
            visualize_episode(model, num_episodes=3)
            plot_training_metrics(metrics_history)
```

**Visualization Components:**
1. **Live Game Rendering:** Use PygameRenderer to show agent playing
2. **Training Metrics:** Plot score, lines cleared, reward over time
3. **Comparison:** Show agent performance vs baseline (greedy AI)

### Option 2: Real-Time Dashboard

**Tools:**
- **TensorBoard:** Track metrics (reward, score, loss)
- **Wandb (Weights & Biases):** Advanced experiment tracking
- **Matplotlib/Plotly:** Custom dashboards

**Metrics to Track:**
- Episode reward (mean, std, min, max)
- Score per episode
- Lines cleared per episode
- Average episode length
- Training loss
- Epsilon (if using epsilon-greedy)

### Option 3: Video Recording

**Strategy:** Record gameplay videos at checkpoints

```python
def record_episode_video(model, env, filename="episode.mp4"):
    """Record an episode as video."""
    import imageio
    
    frames = []
    obs, _ = env.reset()
    done = False
    
    while not done:
        # Render to image
        frame = env.render(mode='rgb_array')
        frames.append(frame)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
    
    # Save as video
    imageio.mimsave(filename, frames, fps=10)
```

**Use Cases:**
- Compare early vs late training
- Create demo videos
- Debug agent behavior

### Option 4: Interactive Visualization

**Strategy:** Build web dashboard with:
- Real-time training metrics
- Gameplay replay viewer
- Hyperparameter tuning interface
- Model comparison tools

**Tech Stack:**
- **Flask/FastAPI:** Backend API
- **React/Vue:** Frontend dashboard
- **WebSocket:** Real-time updates
- **Plotly/D3.js:** Interactive charts

---

## Implementation Plan

### Phase 1: Basic RL Setup (Week 1)

1. **Create Gym Environment**
   - Implement `TetrisEnv` wrapper
   - Define observation/action spaces
   - Implement reward function

2. **Basic Agent**
   - Integrate Stable-Baselines3
   - Simple PPO agent
   - Basic training loop

3. **Initial Visualization**
   - Render episodes periodically
   - Plot training metrics (reward, score)

**Deliverable:** Working RL agent that can train and play Tetris

### Phase 2: Optimization & Visualization (Week 2)

1. **Performance Optimization**
   - Vectorized environments (parallel training)
   - Efficient feature extraction
   - Batch processing

2. **Enhanced Visualization**
   - TensorBoard integration
   - Video recording
   - Comparison with baseline AIs

3. **Hyperparameter Tuning**
   - Learning rate, network architecture
   - Reward function tuning
   - Action space refinement

**Deliverable:** Optimized training with comprehensive visualization

### Phase 3: Advanced Features (Weeks 3-4)

1. **Advanced RL Techniques**
   - Curriculum learning (start easy, increase difficulty)
   - Reward shaping experiments
   - Multi-step planning

2. **Advanced Visualization**
   - Web dashboard
   - Interactive replay viewer
   - Model comparison tools

3. **Evaluation & Analysis**
   - Benchmark against existing AIs
   - Analyze learned strategies
   - Performance profiling

**Deliverable:** Production-ready RL system with full visualization suite

---

## Dependencies to Add

```toml
# Add to pyproject.toml
dependencies = [
    # ... existing dependencies ...
    "gymnasium>=0.29.0",           # RL environment standard
    "stable-baselines3>=2.0.0",    # RL algorithms
    "numpy>=1.24.0",               # Already likely needed
    "tensorboard>=2.13.0",         # Training visualization
    "wandb>=0.15.0",               # Experiment tracking (optional)
    "imageio>=2.31.0",             # Video recording
    "matplotlib>=3.7.0",           # Plotting
]

[project.optional-dependencies]
rl = [
    "gymnasium>=0.29.0",
    "stable-baselines3>=2.0.0",
    "tensorboard>=2.13.0",
    "wandb>=0.15.0",
    "imageio>=2.31.0",
]
```

---

## Expected Performance

### Training Time Estimates

**With current optimized simulator:**
- **1 episode:** ~1-5 seconds (depending on agent skill)
- **1000 episodes:** ~15-80 minutes
- **100k episodes:** ~1-3 days (for decent performance)
- **1M episodes:** ~1-2 weeks (for strong performance)

**With parallel training (8 workers):**
- **100k episodes:** ~3-6 hours
- **1M episodes:** ~1-2 days

### Performance Targets

- **Beginner (1k episodes):** ~100-500 lines cleared
- **Intermediate (10k episodes):** ~500-2000 lines cleared
- **Advanced (100k episodes):** ~2000-5000 lines cleared
- **Expert (1M episodes):** 5000+ lines cleared, competitive with greedy AI

---

## Challenges & Solutions

### Challenge 1: Sparse Rewards
**Problem:** Most actions don't clear lines, so reward signal is sparse.

**Solutions:**
- Dense reward shaping (height penalties, hole penalties)
- Reward for intermediate progress (piece placement quality)
- Curriculum learning (start with easier scenarios)

### Challenge 2: Large State Space
**Problem:** Playfield has 10×24 = 240 cells, huge state space.

**Solutions:**
- Use engineered features (25-30 features vs 240)
- Convolutional networks for spatial patterns
- Attention mechanisms for important regions

### Challenge 3: Action Space Size
**Problem:** Variable number of placements per state.

**Solutions:**
- Mask invalid actions (set probability to 0)
- Pad action space to fixed size
- Use placement-level actions (smaller space)

### Challenge 4: Training Stability
**Problem:** RL can be unstable, especially early in training.

**Solutions:**
- Use PPO (more stable than DQN)
- Careful reward normalization
- Learning rate scheduling
- Early stopping and checkpointing

---

## Next Steps

1. **Immediate (This Week):**
   - Add RL dependencies to `pyproject.toml`
   - Create `TetrisEnv` gym environment
   - Implement basic PPO agent
   - Test with simple reward function

2. **Short Term (Next 2 Weeks):**
   - Optimize feature extraction
   - Implement comprehensive reward function
   - Add TensorBoard visualization
   - Train initial model (10k episodes)

3. **Medium Term (Next Month):**
   - Hyperparameter tuning
   - Advanced visualization (video recording, dashboard)
   - Compare with baseline AIs
   - Document learned strategies

---

## Conclusion

RL integration is **highly feasible** and should produce strong results. The current architecture provides an excellent foundation, and with the performance optimizations already identified, training should be efficient. The modular design makes it easy to add an RL agent without disrupting existing code.

**Key Success Factors:**
1. ✅ Clean architecture (already done)
2. ✅ Fast simulation (optimizations identified)
3. ✅ Good reward function design (needs implementation)
4. ✅ Effective visualization (needs implementation)
5. ✅ Appropriate RL algorithm (PPO recommended)

**Estimated Timeline:** 2-4 weeks for a production-ready RL system with visualization.
