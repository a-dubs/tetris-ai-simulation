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
import copy
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


def state_to_board_channels(
    state: GameState,
    engine: GameEngine = None,
    include_ghost: bool = True,
) -> np.ndarray:
    """Convert game state to multi-channel board representation.
    
    Creates an image-like representation with separate channels:
    - Channel 0: Locked blocks (0/1)
    - Channel 1: Active piece mask (0/1)
    - Channel 2: Ghost piece landing mask (0/1) [optional]
    
    This is better for CNN-based policies.
    
    Args:
        state: Current game state
        engine: Optional game engine for ghost piece calculation
        include_ghost: Whether to include ghost piece channel
        
    Returns:
        Array of shape (H, W, C) where C is 2 or 3 depending on include_ghost
    """
    if engine is None:
        engine = GameEngine(initial_state=state)
    
    height = TOTAL_PLAYFIELD_HEIGHT
    width = PLAYFIELD_WIDTH
    channels = 3 if include_ghost else 2
    
    # Initialize channels
    board = np.zeros((height, width, channels), dtype=np.float32)
    
    # Channel 0: Locked blocks
    for row in range(height):
        for col in range(width):
            if state.playfield[row][col] != ' ':
                board[row, col, 0] = 1.0
    
    # Channel 1: Active piece
    if state.active_tetrimino:
        active = state.active_tetrimino
        for row in range(active.size):
            for col in range(active.size):
                if active.minos[row][col] != ' ':
                    # Convert tetrimino coordinates to playfield coordinates
                    pf_row = active.y + row - 1  # y is 1-indexed
                    pf_col = active.x + col - 1   # x is 1-indexed
                    if 0 <= pf_row < height and 0 <= pf_col < width:
                        board[pf_row, pf_col, 1] = 1.0
        
        # Channel 2: Ghost piece (landing position)
        if include_ghost:
            ghost_tet = copy.copy(active)
            ghost_tet.minos = [row[:] for row in active.minos]
            # Drop ghost piece to landing position
            start_y = ghost_tet.y
            while engine.valid_location(ghost_tet):
                ghost_tet.y -= 1
            ghost_tet.y += 1
            ghost_tet.y = start_y if ghost_tet.y > start_y else ghost_tet.y
            
            # Only draw ghost if it's different from active piece position
            if ghost_tet.y != active.y:
                for row in range(ghost_tet.size):
                    for col in range(ghost_tet.size):
                        if ghost_tet.minos[row][col] != ' ':
                            pf_row = ghost_tet.y + row - 1
                            pf_col = ghost_tet.x + col - 1
                            if 0 <= pf_row < height and 0 <= pf_col < width:
                                # Only mark if not already marked by active piece
                                if board[pf_row, pf_col, 1] == 0:
                                    board[pf_row, pf_col, 2] = 1.0
    
    return board


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
    
    # Column heights (10 features) - NORMALIZED to [0, 1]
    column_heights = []
    max_possible_height = TOTAL_PLAYFIELD_HEIGHT
    for col in range(PLAYFIELD_WIDTH):
        col_data = pf[:, col]
        filled_rows = np.where(col_data == 1)[0]
        if len(filled_rows) > 0:
            # Height is distance from top (row 0 is bottom)
            height = TOTAL_PLAYFIELD_HEIGHT - np.min(filled_rows)
        else:
            height = 0.0
        # Normalize to [0, 1]
        column_heights.append(height / max_possible_height)
    
    features.extend(column_heights)
    
    # Aggregate statistics (normalized)
    if column_heights:
        features.append(np.mean(column_heights))  # Average height (already normalized)
        features.append(np.max(column_heights))    # Max height (already normalized)
        features.append(np.std(column_heights))    # Height variance (normalized)
    else:
        features.extend([0.0, 0.0, 0.0])
    
    # Total blocks (normalized by max possible blocks)
    max_blocks = PLAYFIELD_WIDTH * TOTAL_PLAYFIELD_HEIGHT
    features.append(np.sum(pf) / max_blocks)
    
    # Count holes (empty cells with blocks above) - NORMALIZED
    holes = 0
    for col in range(PLAYFIELD_WIDTH):
        found_block = False
        for row in range(TOTAL_PLAYFIELD_HEIGHT - 1, -1, -1):
            if pf[row, col] == 1:
                found_block = True
            elif found_block and pf[row, col] == 0:
                holes += 1
    
    # Normalize holes by max possible (rough estimate: width * height / 2)
    max_holes = PLAYFIELD_WIDTH * TOTAL_PLAYFIELD_HEIGHT // 2
    features.append(float(holes) / max_holes if max_holes > 0 else 0.0)
    
    # Active piece features - NORMALIZED
    if state.active_tetrimino:
        active = state.active_tetrimino
        active_shape = _get_shape_from_tetrimino(active)
        # Normalize x to [0, 1] (x is 1-indexed, range is roughly 0-10)
        normalized_x = active.x / PLAYFIELD_WIDTH if PLAYFIELD_WIDTH > 0 else 0.0
        # Normalize y to [0, 1]
        normalized_y = active.y / TOTAL_PLAYFIELD_HEIGHT if TOTAL_PLAYFIELD_HEIGHT > 0 else 0.0
        # Orientation is already 0-3, normalize to [0, 1]
        normalized_orientation = _orientation_to_int(active.orientation) / 4.0
        # Shape is already 0-6, normalize to [0, 1]
        normalized_shape = _shape_to_int(active_shape) / 7.0
        features.extend([
            normalized_x,
            normalized_y,
            normalized_orientation,
            normalized_shape,
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # Next piece - NORMALIZED
    if state.next_tetrimino:
        next_shape = _get_shape_from_tetrimino(state.next_tetrimino)
        normalized_next_shape = _shape_to_int(next_shape) / 7.0
        features.append(normalized_next_shape)
    else:
        features.append(0.0)
    
    # Game progress (normalized)
    # Level: normalize by assuming max level ~30
    normalized_level = min(float(state.level) / 30.0, 1.0)
    # Lines cleared: normalize by assuming max ~1000 lines
    normalized_lines = min(float(state.lines_cleared) / 1000.0, 1.0)
    # Score: already normalized
    normalized_score = float(state.score) / 10000.0
    features.extend([
        normalized_level,
        normalized_lines,
        normalized_score,
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


def calculate_reward_minimal(
    prev_state: GameState,
    current_state: GameState,
    done: bool,
    reward_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Ultra-minimal reward: ONLY stack height and holes.
    
    CRITICAL: This reward is based on CHANGE from prev_state to current_state.
    This ensures the agent gets penalized for making things worse, not for
    the absolute state of the board.
    
    This is the simplest possible reward structure:
    - Penalty for INCREASING stack height (encourages low stacks)
    - Penalty for CREATING holes (encourages clean playfield)
    - Strong penalty for max column height (discourages vertical stacking)
    - Terminal penalty on game over
    - Small survival bonus to provide positive signal
    
    No rewards for lines cleared, no complex shaping.
    The agent must learn that clearing lines is good because
    it reduces stack height and holes.
    
    Args:
        prev_state: Previous game state
        current_state: Current game state
        done: Whether episode is done
        reward_params: Optional reward configuration dict
        
    Returns:
        Reward value (based on CHANGE, not absolute state)
    """
    if reward_params is None:
        reward_params = {
            "height_increase_penalty": -2.0,  # Penalty for increasing aggregate height
            "holes_created_penalty": -10.0,   # Strong penalty for creating new holes
            "max_height_penalty": -20.0,      # VERY strong penalty for max column height
            "survival_bonus": 5.0,            # Larger positive reward for surviving
            "game_over": -1000.0,             # Large terminal penalty
        }
    
    reward = 0.0
    
    # Terminal penalty
    if done:
        reward += reward_params.get("game_over", -1000.0)
        return reward
    
    # Survival bonus (provides positive signal)
    reward += reward_params.get("survival_bonus", 5.0)
    
    # Convert playfields to numpy arrays
    prev_pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in prev_state.playfield
    ], dtype=np.float32)
    curr_pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in current_state.playfield
    ], dtype=np.float32)
    
    # Calculate column heights for both states
    def get_column_heights(pf_array):
        heights = []
        for col in range(PLAYFIELD_WIDTH):
            col_data = pf_array[:, col]
            filled_rows = np.where(col_data == 1)[0]
            if len(filled_rows) > 0:
                height = TOTAL_PLAYFIELD_HEIGHT - np.min(filled_rows)
            else:
                height = 0.0
            heights.append(height)
        return heights
    
    prev_heights = get_column_heights(prev_pf)
    curr_heights = get_column_heights(curr_pf)
    
    prev_aggregate = sum(prev_heights)
    curr_aggregate = sum(curr_heights)
    prev_max = max(prev_heights) if prev_heights else 0
    curr_max = max(curr_heights) if curr_heights else 0
    
    # 1. Penalty for INCREASING aggregate height (not absolute height)
    height_increase = curr_aggregate - prev_aggregate
    if height_increase > 0:
        reward += height_increase * reward_params.get("height_increase_penalty", -2.0)
    
    # 2. STRONG penalty for max column height (discourages vertical stacking)
    # This is the key fix - penalize tall columns heavily
    if curr_max > 0:
        reward += curr_max * reward_params.get("max_height_penalty", -20.0)
    
    # 3. Penalty for CREATING holes (not total holes)
    def count_holes(pf_array):
        holes = 0
        for col in range(PLAYFIELD_WIDTH):
            found_block = False
            for row in range(TOTAL_PLAYFIELD_HEIGHT - 1, -1, -1):
                if pf_array[row, col] == 1:
                    found_block = True
                elif found_block and pf_array[row, col] == 0:
                    holes += 1
        return holes
    
    prev_holes = count_holes(prev_pf)
    curr_holes = count_holes(curr_pf)
    new_holes = max(0, curr_holes - prev_holes)
    if new_holes > 0:
        reward += new_holes * reward_params.get("holes_created_penalty", -10.0)
    
    return reward


def calculate_reward_simple(
    prev_state: GameState,
    current_state: GameState,
    done: bool,
    reward_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Calculate simplified reward for RL agent.
    
    Clean reward structure focusing on:
    - Lines cleared (with Tetris bonus)
    - Holes created (penalty)
    - Aggregate height (penalty)
    - Game over (large penalty)
    
    Args:
        prev_state: Previous game state
        current_state: Current game state
        done: Whether episode is done
        reward_params: Optional reward configuration dict
        
    Returns:
        Reward value
    """
    if reward_params is None:
        reward_params = {
            "lines_cleared": 1.0,      # Base reward per line
            "tetris_bonus": 2.0,       # Extra multiplier for 4-line clear
            "holes_penalty": -0.5,     # Penalty per new hole
            "aggregate_height_penalty": -0.01,  # Penalty per unit of aggregate height
            "game_over": -1000.0,      # Large penalty for game over
        }
    
    reward = 0.0
    
    # Game over penalty
    if done:
        reward += reward_params.get("game_over", -1000.0)
        return reward
    
    # Lines cleared (most important!)
    lines_cleared = current_state.lines_cleared - prev_state.lines_cleared
    if lines_cleared > 0:
        base_reward = lines_cleared * reward_params.get("lines_cleared", 1.0)
        # Extra bonus for Tetris (4-line clear)
        if lines_cleared == 4:
            base_reward *= (1.0 + reward_params.get("tetris_bonus", 2.0))
        reward += base_reward
    
    # Calculate holes created (new holes in current state)
    prev_pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in prev_state.playfield
    ], dtype=np.float32)
    curr_pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in current_state.playfield
    ], dtype=np.float32)
    
    # Count holes (empty cells with blocks above)
    def count_holes(pf_array):
        holes = 0
        for col in range(PLAYFIELD_WIDTH):
            found_block = False
            for row in range(TOTAL_PLAYFIELD_HEIGHT - 1, -1, -1):
                if pf_array[row, col] == 1:
                    found_block = True
                elif found_block and pf_array[row, col] == 0:
                    holes += 1
        return holes
    
    prev_holes = count_holes(prev_pf)
    curr_holes = count_holes(curr_pf)
    new_holes = max(0, curr_holes - prev_holes)
    reward += new_holes * reward_params.get("holes_penalty", -0.5)
    
    # Aggregate height penalty
    column_heights = []
    for col in range(PLAYFIELD_WIDTH):
        col_data = curr_pf[:, col]
        filled_rows = np.where(col_data == 1)[0]
        if len(filled_rows) > 0:
            height = TOTAL_PLAYFIELD_HEIGHT - np.min(filled_rows)
        else:
            height = 0.0
        column_heights.append(height)
    
    if column_heights:
        aggregate_height = sum(column_heights)
        reward += aggregate_height * reward_params.get("aggregate_height_penalty", -0.01)
    
    return reward


def calculate_reward(
    prev_state: GameState,
    current_state: GameState,
    done: bool,
    reward_params: Optional[Dict[str, Any]] = None,
) -> float:
    """Calculate reward for RL agent.
    
    Reward structure is configurable via reward_params. Defaults to improved
    values matching fast.py AI philosophy: strong penalties for bad play,
    strong rewards for clears.
    
    Args:
        prev_state: Previous game state
        current_state: Current game state
        done: Whether episode is done
        reward_params: Optional reward configuration dict
        
    Returns:
        Reward value
    """
    # Use improved default reward params matching fast.py AI
    if reward_params is None:
        reward_params = {
            "line_cleared": {"single": 100.0, "double": 300.0, "triple": 500.0, "tetris": 800.0},
            "tetris_bonus": 300.0,  # Extra bonus for 4-line clear
            "score_multiplier": 0.01,
            "level_up": 50.0,
            "game_over": -1000.0,  # Much stronger penalty (was -100)
            # Stack height penalties (exponential like fast.py)
            "max_stack_weight": -8.0,  # Strong penalty for max height
            "max_stack_exponent": 1.5,
            "avg_stack_weight": -5.0,  # Penalty for average height
            "avg_stack_exponent": 1.2,
            "stack_danger_weight": -4.0,  # Penalty for dangerous heights
            "stack_danger_threshold": 15,
            "stack_danger_exponent": 1.2,
            # Cliff penalties (better than simple hole counting)
            "cliff_horizontal_weight": -6.0,
            "cliff_horizontal_exponent": 1.3,
            "cliff_vertical_weight": -7.0,  # Strong penalty for vertical cliffs (holes)
            "cliff_vertical_exponent": 1.5,
            # Legacy parameters (kept for compatibility)
            "height_threshold": 15,
            "height_penalty": 2.0,
            "hole_penalty": 5.0,
            "variance_penalty": 0.5,
            "survival_bonus": 0.0,
        }
    
    reward = 0.0
    
    # Survival bonus (per step) - usually 0
    reward += reward_params.get("survival_bonus", 0.0)
    
    # Game over penalty (VERY STRONG - matching fast.py)
    if done:
        reward += reward_params.get("game_over", -1000.0)
    
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
        # Extra bonus for Tetris (4-line clear)
        if lines_cleared == 4:
            reward += reward_params.get("tetris_bonus", 300.0)
    
    # Score increase
    score_delta = current_state.score - prev_state.score
    reward += score_delta * reward_params.get("score_multiplier", 0.01)
    
    # Level progression
    if current_state.level > prev_state.level:
        reward += reward_params.get("level_up", 50.0)
    
    # Convert playfield to numpy array for analysis
    pf = np.array([
        [1 if cell != ' ' else 0 for cell in row] 
        for row in current_state.playfield
    ])
    pfw = PLAYFIELD_WIDTH
    pfh = TOTAL_PLAYFIELD_HEIGHT
    
    # Calculate column heights
    column_heights = []
    for col in range(pfw):
        col_data = pf[:, col]
        filled_rows = np.where(col_data == 1)[0]
        if len(filled_rows) > 0:
            height = pfh - np.min(filled_rows)
        else:
            height = 0.0
        column_heights.append(height)
    
    if column_heights:
        max_height = max(column_heights)
        avg_height = sum(column_heights) / len(column_heights)
        
        # Exponential stack height penalties (matching fast.py)
        # Scale down to prevent reward explosion
        max_stack_w = reward_params.get("max_stack_weight", -8.0) * 0.1  # Scale down
        max_stack_e = reward_params.get("max_stack_exponent", 1.5)
        # Normalize height before exponentiation to keep rewards reasonable
        normalized_max_height = max_height / TOTAL_PLAYFIELD_HEIGHT
        reward += (normalized_max_height ** max_stack_e) * max_stack_w * 100
        
        avg_stack_w = reward_params.get("avg_stack_weight", -5.0) * 0.1  # Scale down
        avg_stack_e = reward_params.get("avg_stack_exponent", 1.2)
        normalized_avg_height = avg_height / TOTAL_PLAYFIELD_HEIGHT
        reward += (normalized_avg_height ** avg_stack_e) * avg_stack_w * 100
        
        # Stack danger penalty (for heights above threshold)
        stack_d_thresh = reward_params.get("stack_danger_threshold", 15)
        stack_d_w = reward_params.get("stack_danger_weight", -4.0) * 0.1  # Scale down
        stack_d_e = reward_params.get("stack_danger_exponent", 1.2)
        
        # Count dangerous rows (rows above threshold with blocks)
        row_empty_counts = [row.count(" ") for row in current_state.playfield]
        stack_danger = sum([
            int(row_empty_counts[row] < pfw) * ((row - stack_d_thresh) ** stack_d_e)
            for row in range(stack_d_thresh + 1, pfh)
        ])
        reward += stack_danger * stack_d_w * 10  # Scale down
        
        # Calculate cliff penalties (better hole/cliff detection)
        cliff_h_sum, cliff_l_sum = _calculate_cliff_penalties(
            pf, pfw, pfh, reward_params
        )
        reward += cliff_h_sum + cliff_l_sum
    
    return reward


def _calculate_cliff_penalties(
    pf: np.ndarray,
    pfw: int,
    pfh: int,
    reward_params: Dict[str, Any],
) -> tuple[float, float]:
    """Calculate cliff penalties (horizontal and vertical).
    
    Cliffs are overhangs that create holes. This is better than simple
    hole counting because it detects the structure that creates holes.
    
    Args:
        pf: Playfield as numpy array (1 = block, 0 = empty)
        pfw: Playfield width
        pfh: Playfield height
        reward_params: Reward configuration dict
        
    Returns:
        Tuple of (cliff_heights_sum, cliff_lengths_sum)
    """
    # Scale down cliff penalties to prevent reward explosion
    cliff_h_w = reward_params.get("cliff_vertical_weight", -7.0) * 0.1
    cliff_h_e = reward_params.get("cliff_vertical_exponent", 1.5)
    cliff_l_w = reward_params.get("cliff_horizontal_weight", -6.0) * 0.1
    cliff_l_e = reward_params.get("cliff_horizontal_exponent", 1.3)
    
    # Detect cliffs (horizontal overhangs)
    # A cliff is when a block exists in current row but not in row above
    cliffs = np.zeros((pfh - 1, pfw), dtype=int)
    for row in range(1, pfh):
        for col in range(pfw):
            # Cliff if current row has block but row above doesn't
            if pf[row, col] == 1 and pf[row - 1, col] == 0:
                cliffs[row - 1, col] = 1
    
    # Calculate cliff heights (vertical - depth of holes created)
    cliff_heights_sum = 0.0
    for col in range(pfw):
        prev_row = -1
        for row in range(pfh - 2, -1, -1):
            if cliffs[row, col] == 1 and prev_row == -1:
                prev_row = row
            elif pf[row, col] == 1 and prev_row != -1:
                # Found bottom of cliff, calculate depth
                depth = prev_row - row
                # Normalize depth before exponentiation
                normalized_depth = depth / pfh
                cliff_heights_sum += (normalized_depth ** cliff_h_e) * cliff_h_w * 100
                prev_row = -1
            elif row == 0 and prev_row != -1 and pf[row, col] == 0:
                # Cliff extends to top
                depth = prev_row - row + 1
                normalized_depth = depth / pfh
                cliff_heights_sum += (normalized_depth ** cliff_h_e) * cliff_h_w * 100
    
    # Calculate cliff lengths (horizontal - width of overhangs)
    cliff_lengths_sum = 0.0
    for row in range(pfh - 1):
        prev_i = -1
        for i in range(pfw):
            if cliffs[row, i] > 0 and prev_i == -1:
                prev_i = i
            elif cliffs[row, i] == 0 and prev_i != -1:
                # End of cliff segment
                length = i - prev_i
                # Normalize length before exponentiation
                normalized_length = length / pfw
                cliff_lengths_sum += (normalized_length ** cliff_l_e) * cliff_l_w * 100
                prev_i = -1
            elif i == pfw - 1 and prev_i != -1 and cliffs[row, i] > 0:
                # Cliff extends to edge
                length = i - prev_i + 1
                normalized_length = length / pfw
                cliff_lengths_sum += (normalized_length ** cliff_l_e) * cliff_l_w * 100
    
    return cliff_heights_sum, cliff_lengths_sum


if GYMNASIUM_AVAILABLE:
    class TetrisEnv(gym.Env):
        """Gymnasium environment for Tetris RL training.
        
        This wraps the Tetris game engine in a standard RL environment interface.
        Actions are placement choices (discrete), observations can be feature vectors
        or multi-channel board representations.
        
        Supports:
        - Per-piece placement actions (recommended)
        - Multi-channel board observations (for CNN policies)
        - Next queue support
        - Configurable reward functions
        """
        
        metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
        
        def __init__(
            self,
            render_mode: Optional[str] = None,
            reward_params: Optional[Dict[str, Any]] = None,
            observation_mode: str = "features",  # "features" or "board_channels"
            use_simple_reward: bool = False,
            use_minimal_reward: bool = False,
            next_queue_size: int = 5,
            max_placements: int = 50,
            include_ghost: bool = True,
        ):
            """Initialize Tetris environment.
            
            Args:
                render_mode: "human" for GUI, "rgb_array" for video, None for headless
                reward_params: Optional reward configuration dict
                observation_mode: "features" for handcrafted features, "board_channels" for CNN
                use_simple_reward: If True, use simplified reward function (with line clearing rewards)
                use_minimal_reward: If True, use minimal reward (ONLY stack height + holes, no line rewards)
                next_queue_size: Number of next pieces to include in observation
                max_placements: Maximum number of placements to consider (action space size)
                include_ghost: Whether to include ghost piece in board_channels mode
            """
            super().__init__()
            
            self.observation_mode = observation_mode
            self.use_simple_reward = use_simple_reward
            self.use_minimal_reward = use_minimal_reward
            self.next_queue_size = next_queue_size
            self.max_placements = max_placements
            self.include_ghost = include_ghost
            
            # Action space: choose placement (discrete, dynamic based on available placements)
            # We'll use max_placements as the upper bound, but actual valid actions vary
            self.action_space = spaces.Discrete(max_placements)
            
            # Observation space depends on mode
            if observation_mode == "board_channels":
                channels = 3 if include_ghost else 2
                # Add channels for current piece one-hot, next queue, etc.
                # Board: (H, W, C)
                # Current piece: one-hot(7)
                # Next queue: next_queue_size × one-hot(7)
                # Total: H×W×C + 7 + next_queue_size×7
                board_size = TOTAL_PLAYFIELD_HEIGHT * PLAYFIELD_WIDTH * channels
                piece_info_size = 7 + next_queue_size * 7
                obs_size = board_size + piece_info_size
                
                self.observation_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(obs_size,),
                    dtype=np.float32,
                )
            else:  # features mode
                # Original feature vector + next queue
                base_features = 23
                # In features mode, we use normalized shape indices (1 value per piece), not one-hot
                next_queue_features = next_queue_size * 1  # normalized shape index per piece
                obs_size = base_features + next_queue_features
                
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_size,),
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
            
            # Handle invalid action - wrap to valid range instead of ending episode
            if not self.current_placements:
                # No placements available (shouldn't happen, but handle gracefully)
                obs = self._get_obs()
                reward = -10.0  # Small penalty, don't end episode
                done = self.engine.state.game_over
                truncated = False
                info = self._get_info()
                return obs, reward, done, truncated, info
            
            # Handle invalid actions - use modulo but also add penalty
            # This prevents episodes from ending prematurely but discourages invalid actions
            if action >= len(self.current_placements):
                # Wrap to valid range
                action = action % len(self.current_placements)
                # Small penalty for invalid action (helps agent learn valid action space)
                # Note: This penalty is added to the reward calculation below
            
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
            if self.use_minimal_reward:
                reward = calculate_reward_minimal(
                    self.prev_state,
                    self.engine.state,
                    self.engine.state.game_over,
                    reward_params=self.reward_params
                )
            elif self.use_simple_reward:
                reward = calculate_reward_simple(
                    self.prev_state,
                    self.engine.state,
                    self.engine.state.game_over,
                    reward_params=self.reward_params
                )
            else:
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
            if self.observation_mode == "board_channels":
                return self._get_board_channels_obs()
            else:
                return self._get_features_obs()
        
        def _get_board_channels_obs(self) -> np.ndarray:
            """Get multi-channel board observation."""
            # Get board channels
            board_channels = state_to_board_channels(
                self.engine.state,
                engine=self.engine,
                include_ghost=self.include_ghost
            )
            # Flatten board: (H, W, C) -> (H*W*C,)
            board_flat = board_channels.flatten()
            
            # Current piece one-hot
            if self.engine.state.active_tetrimino:
                active_shape = _get_shape_from_tetrimino(self.engine.state.active_tetrimino)
                piece_idx = _shape_to_int(active_shape)
                piece_onehot = np.zeros(7, dtype=np.float32)
                piece_onehot[piece_idx] = 1.0
            else:
                piece_onehot = np.zeros(7, dtype=np.float32)
            
            # Next queue one-hot encoding
            next_queue_onehot = []
            # Build queue from bag and next_tetrimino
            queue_shapes = []
            if self.engine.state.next_tetrimino:
                next_shape = _get_shape_from_tetrimino(self.engine.state.next_tetrimino)
                queue_shapes.append(next_shape)
            # Add shapes from bag
            for shape_name in self.engine.state.bag[:self.next_queue_size - 1]:
                queue_shapes.append(shape_name)
            
            # Pad queue to next_queue_size
            while len(queue_shapes) < self.next_queue_size:
                # Use last piece or default to "I"
                if queue_shapes:
                    queue_shapes.append(queue_shapes[-1])
                else:
                    queue_shapes.append("I")
            
            # Encode each piece in queue
            for shape in queue_shapes[:self.next_queue_size]:
                piece_idx = _shape_to_int(shape)
                piece_oh = np.zeros(7, dtype=np.float32)
                piece_oh[piece_idx] = 1.0
                next_queue_onehot.extend(piece_oh)
            
            # Combine all parts
            obs = np.concatenate([board_flat, piece_onehot, np.array(next_queue_onehot)])
            return obs.astype(np.float32)
        
        def _get_features_obs(self) -> np.ndarray:
            """Get handcrafted feature observation."""
            # Base features
            base_features = state_to_features(self.engine.state)
            
            # Next queue features
            next_queue_features = []
            # Build queue from bag and next_tetrimino
            queue_shapes = []
            if self.engine.state.next_tetrimino:
                next_shape = _get_shape_from_tetrimino(self.engine.state.next_tetrimino)
                queue_shapes.append(next_shape)
            # Add shapes from bag
            for shape_name in self.engine.state.bag[:self.next_queue_size - 1]:
                queue_shapes.append(shape_name)
            
            # Pad queue to next_queue_size
            while len(queue_shapes) < self.next_queue_size:
                # Use last piece or default to "I"
                if queue_shapes:
                    queue_shapes.append(queue_shapes[-1])
                else:
                    queue_shapes.append("I")
            
            # Encode each piece in queue as normalized shape index
            for shape in queue_shapes[:self.next_queue_size]:
                normalized_shape = _shape_to_int(shape) / 7.0
                next_queue_features.append(normalized_shape)
            
            # Combine
            obs = np.concatenate([base_features, np.array(next_queue_features)])
            return obs.astype(np.float32)
        
        def _get_info(self) -> Dict:
            """Get info dictionary."""
            info = {
                "score": self.engine.state.score,
                "lines_cleared": self.engine.state.lines_cleared,
                "level": self.engine.state.level,
                "game_over": self.engine.state.game_over,
            }
            # Add valid actions count for action masking (if needed)
            if hasattr(self, 'current_placements'):
                info["valid_actions"] = len(self.current_placements)
            return info
        
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
