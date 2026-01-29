"""Hybrid RL approach: RL agent that uses FastAI's heuristic structure.

Instead of learning from scratch, this agent:
1. Uses FastAI's placement finding (all possible placements)
2. Uses FastAI's heuristic features (stack height, cliffs, etc.)
3. Learns to weight these features via RL

This combines the best of both worlds:
- FastAI's domain knowledge (what features matter)
- RL's ability to learn optimal weights
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.ai.placement import find_all_placements, Placement
from tetris.ai.fast import FastAIPlayer
from tetris.core.constants import PLAYFIELD_WIDTH, TOTAL_PLAYFIELD_HEIGHT


class HybridRLEnv(gym.Env):
    """RL environment that uses FastAI's heuristic features.
    
    Observation: Board state + FastAI heuristic features
    Action: Select from available placements (discrete)
    Reward: Game performance metrics
    """
    
    def __init__(
        self,
        use_heuristic_features: bool = True,
        max_placements: int = 50,
    ):
        """Initialize hybrid RL environment.
        
        Args:
            use_heuristic_features: Whether to include FastAI features in obs
            max_placements: Maximum placements to consider per step
        """
        super().__init__()
        
        self.use_heuristic_features = use_heuristic_features
        self.max_placements = max_placements
        
        # FastAI for feature extraction
        self.fast_ai = FastAIPlayer()
        
        # Action space: select placement index (discrete)
        self.action_space = spaces.Discrete(max_placements)
        
        # Observation space
        if use_heuristic_features:
            # Board channels + heuristic features
            # Board: 3 channels (locked, active, ghost) = H*W*3
            # Features: stack metrics, cliff metrics, etc. = ~20 features
            board_size = TOTAL_PLAYFIELD_HEIGHT * PLAYFIELD_WIDTH * 3
            feature_size = 20  # Heuristic features
            obs_size = board_size + feature_size
        else:
            # Just board channels
            obs_size = TOTAL_PLAYFIELD_HEIGHT * PLAYFIELD_WIDTH * 3
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )
        
        # Game state
        self.engine = None
        self.current_state = None
        self.placements = []
        self.episode_reward = 0.0
        self.episode_length = 0
        
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.engine = GameEngine()
        self.current_state = self.engine.state
        self.placements = []
        self.episode_reward = 0.0
        self.episode_length = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Take step: select placement and execute."""
        # Get available placements if not cached
        if not self.placements:
            self.placements = find_all_placements(
                self.current_state, self.engine
            )[:self.max_placements]
        
        # Select placement
        if action < len(self.placements):
            placement = self.placements[action]
        else:
            # Invalid action: use first placement or drop
            if self.placements:
                placement = self.placements[0]
            else:
                # No placements: just drop
                placement = None
        
        # Execute placement
        prev_state = self.current_state
        if placement:
            # Execute move sequence
            for move in placement.path:
                self.engine.process_move(move)
            self.current_state = self.engine.state
        else:
            # Just drop
            self.engine.process_move("drop")
            self.current_state = self.engine.state
        
        # Calculate reward
        reward = self._calculate_reward(prev_state, self.current_state)
        self.episode_reward += reward
        self.episode_length += 1
        
        # Check done
        done = self.current_state.game_over
        truncated = False
        
        # Clear placements cache for next step
        self.placements = []
        
        # Info
        info = {
            "score": self.current_state.score,
            "lines_cleared": self.current_state.lines_cleared,
            "level": self.current_state.level,
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        from tetris.rl.env import state_to_board_channels
        
        # Board channels
        board_channels = state_to_board_channels(
            self.current_state, self.engine, include_ghost=True
        )
        board_flat = board_channels.flatten()
        
        if not self.use_heuristic_features:
            return board_flat.astype(np.float32)
        
        # Extract FastAI heuristic features
        features = self._extract_heuristic_features()
        
        # Combine
        obs = np.concatenate([board_flat, features]).astype(np.float32)
        return obs
    
    def _extract_heuristic_features(self) -> np.ndarray:
        """Extract FastAI-style heuristic features from current state."""
        playfield = self.current_state.playfield
        pfw = len(playfield[0])
        pfh = len(playfield)
        
        features = []
        
        # Stack height metrics
        row_empty_counts = [row.count(" ") for row in playfield]
        max_stack_height = sum([int(empty_count < pfw) for empty_count in row_empty_counts])
        features.append(max_stack_height / pfh)  # Normalized
        
        # Average stack height
        column_heights = []
        for col in range(pfw):
            max_height = 0
            for row in range(pfh):
                if playfield[row][col] != " ":
                    max_height = pfh - row
                    break
            column_heights.append(max_height)
        avg_stack = sum(column_heights) / float(pfw) if pfw > 0 else 0
        features.append(avg_stack / pfh)  # Normalized
        
        # Stack variance (height differences)
        if column_heights:
            stack_variance = np.var(column_heights) / (pfh ** 2)  # Normalized
            features.append(stack_variance)
        else:
            features.append(0.0)
        
        # Cliff metrics (simplified)
        cliff_count = 0
        for row in range(1, min(10, pfh)):  # Check top 10 rows
            for col in range(pfw):
                if (playfield[row][col] != " " and 
                    playfield[row - 1][col] == " "):
                    cliff_count += 1
        features.append(cliff_count / (pfw * 10))  # Normalized
        
        # Hole count (simplified)
        hole_count = 0
        for col in range(pfw):
            found_block = False
            for row in range(pfh):
                if playfield[row][col] != " ":
                    found_block = True
                elif found_block and playfield[row][col] == " ":
                    hole_count += 1
        features.append(hole_count / (pfw * pfh))  # Normalized
        
        # Well depth (simplified)
        well_depth = 0
        for col in range(1, pfw - 1):
            center = column_heights[col]
            left = column_heights[col - 1]
            right = column_heights[col + 1]
            if center < left - 2 and center < right - 2:
                well_depth += min(left, right) - center
        features.append(well_depth / pfh)  # Normalized
        
        # Active piece info
        if self.current_state.active_tetrimino:
            features.append(1.0)  # Has active piece
            # Piece type (one-hot-ish)
            piece_types = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
            piece_type_idx = piece_types.index(
                self.current_state.active_tetrimino.type
            ) if self.current_state.active_tetrimino.type in piece_types else 0
            for i in range(len(piece_types)):
                features.append(1.0 if i == piece_type_idx else 0.0)
        else:
            features.append(0.0)  # No active piece
            features.extend([0.0] * 7)  # No piece type
        
        # Next piece info
        if self.current_state.next_tetrimino:
            next_types = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
            next_type_idx = next_types.index(
                self.current_state.next_tetrimino.type
            ) if self.current_state.next_tetrimino.type in next_types else 0
            for i in range(len(next_types)):
                features.append(1.0 if i == next_type_idx else 0.0)
        else:
            features.extend([0.0] * 7)
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0.0)
        
        return np.array(features[:20], dtype=np.float32)
    
    def _calculate_reward(
        self, prev_state: GameState, current_state: GameState
    ) -> float:
        """Calculate reward based on state changes."""
        reward = 0.0
        
        # Line clears (strong reward)
        lines_cleared = current_state.lines_cleared - prev_state.lines_cleared
        if lines_cleared > 0:
            clear_rewards = {1: 100.0, 2: 300.0, 3: 500.0, 4: 800.0}
            reward += clear_rewards.get(lines_cleared, 100.0 * lines_cleared)
            if lines_cleared == 4:
                reward += 300.0  # Tetris bonus
        
        # Score increase
        score_diff = current_state.score - prev_state.score
        reward += score_diff * 0.01
        
        # Level up
        if current_state.level > prev_state.level:
            reward += 50.0
        
        # Game over penalty
        if current_state.game_over:
            reward -= 1000.0
        
        # Survival bonus (small)
        reward += 0.1
        
        # Stack height penalty (small, continuous)
        playfield = current_state.playfield
        pfw = len(playfield[0])
        pfh = len(playfield)
        row_empty_counts = [row.count(" ") for row in playfield]
        max_height = sum([int(empty_count < pfw) for empty_count in row_empty_counts])
        height_penalty = -0.1 * max_height
        reward += height_penalty
        
        return reward


def train_hybrid_rl(
    total_timesteps: int = 500000,
    learning_rate: float = 3e-4,
    use_heuristic_features: bool = True,
    model_name: str = "hybrid_rl",
    log_dir: str = "./rl_logs",
    seed: Optional[int] = None,
):
    """Train hybrid RL agent.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        use_heuristic_features: Whether to use FastAI features
        model_name: Model name for saving
        log_dir: Log directory
        seed: Random seed
    """
    if not RL_AVAILABLE:
        raise ImportError(
            "RL dependencies required. Install with: "
            "uv pip install gymnasium stable-baselines3"
        )
    
    print(f"Training Hybrid RL agent...")
    print(f"  Timesteps: {total_timesteps}")
    print(f"  Using heuristic features: {use_heuristic_features}")
    print()
    
    # Create environment
    env = HybridRLEnv(use_heuristic_features=use_heuristic_features)
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=seed,
        tensorboard_log=f"{log_dir}/tensorboard",
    )
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )
    
    # Save
    model_path = f"{log_dir}/{model_name}_final.zip"
    model.save(model_path)
    print(f"\nSaved model to {model_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hybrid RL agent")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--no-heuristic-features", action="store_true")
    parser.add_argument("--model-name", type=str, default="hybrid_rl")
    parser.add_argument("--log-dir", type=str, default="./rl_logs")
    parser.add_argument("--seed", type=int)
    
    args = parser.parse_args()
    
    train_hybrid_rl(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        use_heuristic_features=not args.no_heuristic_features,
        model_name=args.model_name,
        log_dir=args.log_dir,
        seed=args.seed,
    )
