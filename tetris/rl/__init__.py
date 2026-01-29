"""Reinforcement Learning module for Tetris.

This module provides RL training capabilities for Tetris, including:
- Gymnasium environment wrapper
- Feature extraction and reward functions
- Training utilities
- Visualization tools
- Benchmarking
"""

try:
    from tetris.rl.env import TetrisEnv, state_to_features, calculate_reward, TrainingMetrics
    __all__ = ["TetrisEnv", "state_to_features", "calculate_reward", "TrainingMetrics"]
except ImportError:
    # RL dependencies not installed
    __all__ = []
