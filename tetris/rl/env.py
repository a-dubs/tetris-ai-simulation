"""Example RL agent implementation for Tetris.

This is a reference implementation showing how to integrate RL into the existing
Tetris architecture. It demonstrates:
1. Gym environment wrapper
2. Feature extraction
3. Reward calculation
4. Basic training loop with visualization

To use this, install RL dependencies:
    uv pip install gymnasium stable-baselines3 tensorboard

Then train:
    uv run python -m tetris.rl.training train --timesteps 100000

Visualize:
    uv run python -m tetris.rl.training visualize --model model.zip
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None

from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.core.tetrimino import Tetrimino
from tetris.ai.placement import find_all_placements, Placement
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


@dataclass
class TrainingMetrics:
    """Track training metrics over time."""
    episode_rewards: List[float] = None
    episode_scores: List[int] = None
    episode_lines: List[int] = None
    episode_lengths: List[int] = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_scores is None:
            self.episode_scores = []
        if self.episode_lines is None:
            self.episode_lines = []
        if self.episode_lengths is None:
            self.episode_lengths = []
    
    def add_episode(self, reward: float, score: int, lines: int, length: int):
        """Add metrics for one episode."""
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_lines.append(lines)
        self.episode_lengths.append(length)
    
    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """Get statistics for last N episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent = min(n, len(self.episode_rewards))
        return {
            "mean_reward": np.mean(self.episode_rewards[-recent:]),
            "mean_score": np.mean(self.episode_scores[-recent:]),
            "mean_lines": np.mean(self.episode_lines[-recent:]),
            "mean_length": np.mean(self.episode_lengths[-recent:]),
        }


def state_to_features(state: GameState) -> np.ndarray:
    """Convert game state to feature vector for RL.
    
    Uses engineered features for efficiency:
    - Column heights (10 features)
    - Aggregate statistics (4 features)
    - Holes count (1 feature)
    - Active piece info (4 features)
    - Next piece (1 feature)
    - Game progress (3 features)
    
    Total: ~23 features
    
    Args:
        state: Current game state
        
    Returns:
        Feature vector as numpy array
    """
    features = []
    
    # Convert playfield to binary array
    pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in state.playfield
    ], dtype=np.float32)
    
    # Column heights (10 features)
    column_heights = []
    for col in range(PLAYFIELD_WIDTH):
        col_data = pf[:, col]
        filled_rows = np.where(col_data == 1)[0]
        if len(filled_rows) > 0:
            # Height is distance from top (row 0 is bottom)
            height = TOTAL_PLAYFIELD_HEIGHT - np.min(filled_rows)
        else:
            height = 0.0
        column_heights.append(height)
    
    features.extend(column_heights)
    
    # Aggregate statistics
    if column_heights:
        features.append(np.mean(column_heights))  # Average height
        features.append(np.max(column_heights))    # Max height
        features.append(np.std(column_heights))    # Height variance
    else:
        features.extend([0.0, 0.0, 0.0])
    
    features.append(np.sum(pf))  # Total blocks
    
    # Count holes (empty cells with blocks above)
    holes = 0
    for col in range(PLAYFIELD_WIDTH):
        found_block = False
        for row in range(TOTAL_PLAYFIELD_HEIGHT - 1, -1, -1):
            if pf[row, col] == 1:
                found_block = True
            elif found_block and pf[row, col] == 0:
                holes += 1
    
    features.append(float(holes))
    
    # Active piece features
    if state.active_tetrimino:
        active = state.active_tetrimino
        active_shape = _get_shape_from_tetrimino(active)
        features.extend([
            float(active.x),
            float(active.y),
            float(_orientation_to_int(active.orientation)),
            float(_shape_to_int(active_shape)),
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Next piece
    if state.next_tetrimino:
        next_shape = _get_shape_from_tetrimino(state.next_tetrimino)
        features.append(float(_shape_to_int(next_shape)))
    else:
        features.append(0.0)
    
    # Game progress (normalized)
    features.extend([
        float(state.level),
        float(state.lines_cleared),
        float(state.score) / 10000.0,  # Normalize score
    ])
    
    return np.array(features, dtype=np.float32)


def _orientation_to_int(orientation: str) -> int:
    """Convert orientation string to int."""
    mapping = {"N": 0, "E": 1, "S": 2, "W": 3}
    return mapping.get(orientation, 0)


def _get_shape_from_tetrimino(tetrimino) -> str:
    """Extract shape name from tetrimino by matching minos pattern.
    
    Uses the color codes in minos to identify the shape:
    - 'c' = I (cyan)
    - 'y' = O (yellow)
    - 'p' = T (purple)
    - 'g' = S (green)
    - 'r' = Z (red)
    - 'b' = J (blue)
    - 'o' = L (orange)
    
    Args:
        tetrimino: Tetrimino object
        
    Returns:
        Shape name string
    """
    if not tetrimino or not hasattr(tetrimino, 'minos'):
        return "I"
    
    # Find first non-space character in minos
    for row in tetrimino.minos:
        for cell in row:
            if cell != ' ':
                # Map color code to shape
                color_to_shape = {
                    'c': 'I',
                    'y': 'O',
                    'p': 'T',
                    'g': 'S',
                    'r': 'Z',
                    'b': 'J',
                    'o': 'L',
                }
                return color_to_shape.get(cell, 'I')
    
    return "I"  # Default fallback


def _shape_to_int(shape: str) -> int:
    """Convert tetrimino shape to int."""
    shapes = ["I", "O", "T", "S", "Z", "J", "L"]
    return shapes.index(shape) if shape in shapes else 0


def calculate_reward(
    prev_state: GameState,
    current_state: GameState,
    done: bool,
    reward_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Calculate reward for RL agent.
    
    Reward structure is configurable via reward_params. Defaults to standard
    values if not provided.
    
    Args:
        prev_state: Previous game state
        current_state: Current game state
        done: Whether episode is done
        reward_params: Optional reward configuration dict
        
    Returns:
        Reward value
    """
    # Use default reward params if not provided
    if reward_params is None:
        reward_params = {
            "line_cleared": {"single": 100.0, "double": 300.0, "triple": 500.0, "tetris": 800.0},
            "score_multiplier": 0.01,
            "level_up": 50.0,
            "game_over": -100.0,
            "height_threshold": 15,
            "height_penalty": 2.0,
            "hole_penalty": 5.0,
            "variance_penalty": 0.5,
            "survival_bonus": 0.0,
        }
    
    reward = 0.0
    
    # Survival bonus (per step)
    reward += reward_params.get("survival_bonus", 0.0)
    
    # Game over penalty
    if done:
        reward += reward_params.get("game_over", -100.0)
    
    # Lines cleared (most important!)
    lines_cleared = current_state.lines_cleared - prev_state.lines_cleared
    if lines_cleared > 0:
        line_rewards_config = reward_params.get("line_cleared", {})
        line_rewards = [
            line_rewards_config.get("single", 100.0),
            line_rewards_config.get("double", 300.0),
            line_rewards_config.get("triple", 500.0),
            line_rewards_config.get("tetris", 800.0),
        ]
        reward += line_rewards[min(lines_cleared - 1, 3)]
    
    # Score increase
    score_delta = current_state.score - prev_state.score
    reward += score_delta * reward_params.get("score_multiplier", 0.01)
    
    # Level progression
    if current_state.level > prev_state.level:
        reward += reward_params.get("level_up", 50.0)
    
    # Shape penalties (encourage good play)
    pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in current_state.playfield
    ])
    
    # Column heights
    column_heights = []
    for col in range(PLAYFIELD_WIDTH):
        col_data = pf[:, col]
        filled_rows = np.where(col_data == 1)[0]
        if len(filled_rows) > 0:
            height = TOTAL_PLAYFIELD_HEIGHT - np.min(filled_rows)
        else:
            height = 0.0
        column_heights.append(height)
    
    if column_heights:
        max_height = max(column_heights)
        height_variance = np.std(column_heights)
        height_threshold = reward_params.get("height_threshold", 15)
        height_penalty = reward_params.get("height_penalty", 2.0)
        variance_penalty = reward_params.get("variance_penalty", 0.5)
        
        # Penalize high stacks
        if max_height > height_threshold:
            reward -= (max_height - height_threshold) * height_penalty
        
        # Penalize height variance (encourage flat playfield)
        reward -= height_variance * variance_penalty
    
    # Penalize holes
    holes = 0
    for col in range(PLAYFIELD_WIDTH):
        found_block = False
        for row in range(TOTAL_PLAYFIELD_HEIGHT - 1, -1, -1):
            if pf[row, col] == 1:
                found_block = True
            elif found_block and pf[row, col] == 0:
                holes += 1
    
    reward -= holes * reward_params.get("hole_penalty", 5.0)
    
    return reward


if GYMNASIUM_AVAILABLE:
    class TetrisEnv(gym.Env):
        """Gymnasium environment for Tetris RL training.
        
        This wraps the Tetris game engine in a standard RL environment interface.
        Actions are placement choices (discrete), observations are feature vectors.
        """
        
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
        
        def __init__(self, render_mode: Optional[str] = None, reward_params: Optional[Dict[str, Any]] = None):
            """Initialize Tetris environment.
            
            Args:
                render_mode: "human" for GUI, "rgb_array" for video, None for headless
                reward_params: Optional reward configuration dict
            """
            super().__init__()
            
            # Action space: choose placement (discrete, max 20 placements)
            self.action_space = spaces.Discrete(20)
            
            # Observation space: feature vector (~23 features)
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(23,),
                dtype=np.float32,
            )
            
            self.render_mode = render_mode
            self.reward_params = reward_params
            self.engine = None
            self.prev_state = None
            self.current_placements = []
            self.metrics = TrainingMetrics()
            
            # For rendering
            self.renderer = None
            if render_mode == "human":
                from tetris.render.pygame_gui import PygameRenderer
                self.renderer = PygameRenderer("Tetris RL Training")
        
        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            """Reset environment to initial state.
            
            Returns:
                observation, info
            """
            super().reset(seed=seed)
            
            self.engine = GameEngine()
            self.prev_state = self.engine.state
            self.current_placements = []
            
            obs = self._get_obs()
            info = self._get_info()
            
            return obs, info
        
        def step(self, action: int):
            """Execute action and return (obs, reward, done, truncated, info).
            
            Args:
                action: Placement index to choose
                
            Returns:
                observation, reward, done, truncated, info
            """
            # Get all valid placements
            if not self.current_placements:
                self.current_placements = find_all_placements(
                    self.engine.state, self.engine
                )
            
            # Handle invalid action
            if not self.current_placements or action >= len(self.current_placements):
                # End episode with penalty
                obs = self._get_obs()
                reward = -100.0
                done = True
                truncated = False
                info = self._get_info()
                return obs, reward, done, truncated, info
            
            # Execute chosen placement
            placement = self.current_placements[action]
            
            # Apply moves in sequence
            for move in placement.path:
                self.engine.state = self.engine.apply_move(move)
                
                # If dropping, place piece and update playfield
                if move == "drop":
                    new_state, lines_cleared = self.engine.update_playfield(
                        self.engine.state.active_tetrimino
                    )
                    
                    # Update score and level
                    from tetris.core.constants import BAG_SIZE, spawn_y_for_size
                    from tetris.core.tetrimino import Tetrimino
                    
                    score = new_state.score + lines_cleared * 100 * new_state.level
                    new_level = new_state.level
                    clears_needed = new_level * ((5 + new_level * 5) / 2)
                    if new_state.lines_cleared >= clears_needed:
                        new_level += 1
                    
                    # Spawn next piece
                    bag = new_state.bag.copy()
                    if not bag:
                        bag = Tetrimino.make_bag(BAG_SIZE)
                    
                    active_tet = new_state.next_tetrimino
                    active_tet.x = PLAYFIELD_WIDTH // 2 - 1
                    active_tet.y = spawn_y_for_size(active_tet.size)
                    active_tet.lowest_y = active_tet.y
                    next_shape = bag.pop(0)
                    next_tet = Tetrimino(
                        next_shape,
                        x=PLAYFIELD_WIDTH // 2 - 1,
                        y=spawn_y_for_size(len(Tetrimino.get_minos(next_shape))),
                    )
                    
                    # Check if piece has landed
                    temp_engine = GameEngine(initial_state=new_state)
                    if temp_engine.valid_location(active_tet):
                        test_tet = active_tet.__class__.__new__(active_tet.__class__)
                        test_tet.__dict__.update(active_tet.__dict__)
                        test_tet.y -= 1
                        if temp_engine.valid_location(test_tet):
                            active_tet.y -= 1
                    
                    game_over = not temp_engine.valid_location(active_tet)
                    
                    self.engine.state = (
                        new_state.with_score(score)
                        .with_level(new_level)
                        .with_lines_cleared(new_state.lines_cleared + lines_cleared)
                        .with_active_tetrimino(active_tet)
                        .with_next_tetrimino(next_tet)
                        .with_bag(bag)
                        .with_game_over(game_over)
                    )
                    
                    # Clear placements cache for next piece
                    self.current_placements = []
                    break
            
            # Calculate reward
            reward = calculate_reward(
                self.prev_state, 
                self.engine.state, 
                self.engine.state.game_over,
                reward_params=self.reward_params
            )
            
            # Update previous state
            self.prev_state = self.engine.state
            
            # Get observation and info
            obs = self._get_obs()
            done = self.engine.state.game_over
            truncated = False
            info = self._get_info()
            
            # Render if requested
            if self.render_mode == "human" and self.renderer:
                self.renderer.render(self.engine.state)
            
            return obs, reward, done, truncated, info
        
        def _get_obs(self) -> np.ndarray:
            """Convert game state to observation vector."""
            return state_to_features(self.engine.state)
        
        def _get_info(self) -> Dict:
            """Get info dictionary."""
            return {
                "score": self.engine.state.score,
                "lines_cleared": self.engine.state.lines_cleared,
                "level": self.engine.state.level,
                "game_over": self.engine.state.game_over,
            }
        
        def render(self):
            """Render environment (handled in step for human mode)."""
            if self.render_mode == "human":
                pass  # Already rendered in step
            elif self.render_mode == "rgb_array":
                # Return RGB array for video recording
                # Would need to implement this with pygame or similar
                return None
        
        def close(self):
            """Clean up resources."""
            if self.renderer:
                self.renderer.cleanup()


def train_rl_agent(
    total_episodes: int = 10000,
    render_every: int = 100,
    checkpoint_every: int = 1000,
    model_path: str = "tetris_rl_model",
):
    """Train RL agent with periodic visualization.
    
    Args:
        total_episodes: Total episodes to train
        render_every: Render every N episodes
        checkpoint_every: Save model every N episodes
        model_path: Path prefix for saved models
    """
    if not GYMNASIUM_AVAILABLE:
        print("Error: gymnasium not installed. Install with: uv pip install gymnasium stable-baselines3")
        return
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    except ImportError:
        print("Error: stable-baselines3 not installed. Install with: uv pip install stable-baselines3")
        return
    
    # Create environment
    env = TetrisEnv()
    eval_env = TetrisEnv()
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./rl_tensorboard/",
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{model_path}_best/",
        log_path="./rl_logs/",
        eval_freq=checkpoint_every,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_every,
        save_path=f"{model_path}_checkpoints/",
        name_prefix="model",
    )
    
    # Training loop with periodic visualization
    print(f"Starting training for {total_episodes} episodes...")
    print(f"Rendering every {render_every} episodes")
    print(f"Checkpointing every {checkpoint_every} episodes")
    print()
    
    for episode in range(0, total_episodes, checkpoint_every):
        # Train for checkpoint_every episodes
        model.learn(
            total_timesteps=checkpoint_every * 100,  # Approximate timesteps
            callback=[eval_callback, checkpoint_callback],
            reset_num_timesteps=False,
        )
        
        # Visualize periodically
        if episode % render_every == 0:
            print(f"\nVisualizing episode {episode}...")
            visualize_episode(model, env, num_episodes=1)
    
    # Save final model
    model.save(f"{model_path}_final")
    print(f"\nTraining complete! Model saved to {model_path}_final")


def visualize_episode(model, env, num_episodes: int = 1):
    """Visualize agent playing Tetris.
    
    Args:
        model: Trained RL model
        env: Tetris environment
        num_episodes: Number of episodes to visualize
    """
    if not GYMNASIUM_AVAILABLE:
        print("Error: gymnasium not installed")
        return
    
    # Create visualization environment
    viz_env = TetrisEnv(render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = viz_env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = viz_env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1}: Score={info['score']}, "
              f"Lines={info['lines_cleared']}, Reward={total_reward:.2f}")
    
    viz_env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tetris RL Training")
    parser.add_argument("command", choices=["train", "visualize"], 
                       help="Command to run")
    parser.add_argument("--episodes", type=int, default=10000,
                       help="Number of episodes to train")
    parser.add_argument("--checkpoint", type=str, default="tetris_rl_model_final",
                       help="Path to model checkpoint")
    parser.add_argument("--render-every", type=int, default=100,
                       help="Render every N episodes")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_rl_agent(
            total_episodes=args.episodes,
            render_every=args.render_every,
        )
    elif args.command == "visualize":
        if not GYMNASIUM_AVAILABLE:
            print("Error: gymnasium not installed")
            exit(1)
        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("Error: stable-baselines3 not installed")
            exit(1)
        
        env = TetrisEnv(render_mode="human")
        model = PPO.load(args.checkpoint)
        visualize_episode(model, env, num_episodes=3)
