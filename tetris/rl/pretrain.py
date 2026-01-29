"""Pretrain RL agent using Fast AI decisions (behavioral cloning).

This allows us to:
1. Initialize the RL agent with good weights from the fast AI
2. Sanity check the RL harness (if agent starts at fast AI performance)
3. Provide a good starting point for RL learning
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from collections import deque

try:
    import torch
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from tetris.rl.env import TetrisEnv
from tetris.ai.fast import FastAIPlayer
from tetris.ai.placement import find_all_placements
from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState


def generate_expert_dataset(
    env,
    num_episodes: int = 100,
    max_steps_per_episode: int = 1000,
) -> Tuple[List[np.ndarray], List[int]]:
    """Generate dataset of (observation, action) pairs from Fast AI.
    
    Args:
        env: Tetris environment (can be wrapped in DummyVecEnv)
        num_episodes: Number of episodes to collect
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        Tuple of (observations, actions) lists
    """
    fast_ai = FastAIPlayer()
    observations = []
    actions = []
    
    print(f"Generating expert dataset from Fast AI ({num_episodes} episodes)...")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps_per_episode:
            # Get Fast AI's decision
            state = env.engine.state
            placements = find_all_placements(state, env.engine)
            
            if not placements:
                break
            
            # Fast AI evaluates and picks best placement
            best_placement = None
            best_score = float("-inf")
            
            for placement in placements:
                # Use Fast AI's evaluation
                test_engine = GameEngine(initial_state=state)
                new_state, lines_cleared = test_engine.update_playfield(placement.tetrimino)
                score = fast_ai._improved_heuristic_evaluation(new_state.playfield, lines_cleared)
                
                if score > best_score:
                    best_score = score
                    best_placement = placement
            
            if best_placement is None:
                break
            
            # Find the index of the best placement
            best_action = placements.index(best_placement)
            
            # Store observation and action
            observations.append(obs.copy())
            actions.append(best_action)
            
            # Step environment (but we already know the action)
            obs, reward, done, truncated, info = env.step(best_action)
            steps += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Collected {episode + 1}/{num_episodes} episodes ({len(observations)} samples)")
    
    print(f"Generated {len(observations)} expert samples")
    return observations, actions


def pretrain_with_behavioral_cloning(
    env: TetrisEnv,
    model: PPO,
    observations: List[np.ndarray],
    actions: List[int],
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> PPO:
    """Pretrain PPO policy network using behavioral cloning.
    
    Uses stable-baselines3's built-in pretraining via learn() with expert dataset.
    This is simpler and more reliable than manually training the network.
    
    Args:
        env: Tetris environment
        model: PPO model to pretrain
        observations: List of expert observations
        actions: List of expert actions
        epochs: Number of training epochs (not used, kept for compatibility)
        batch_size: Batch size for training
        learning_rate: Learning rate for pretraining
        
    Returns:
        Pretrained model
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for pretraining")
    
    print(f"Pretraining model with {len(observations)} expert samples...")
    
    # Convert to numpy arrays
    obs_array = np.array(observations)
    actions_array = np.array(actions)
    
    # Use stable-baselines3's expert dataset pretraining
    # We'll create a simple expert dataset and use learn() with expert actions
    # Actually, simpler approach: use the expert actions directly via learn()
    
    # Create a custom callback that uses expert actions
    from stable_baselines3.common.callbacks import BaseCallback
    
    class ExpertActionCallback(BaseCallback):
        def __init__(self, expert_obs, expert_actions, verbose=0):
            super().__init__(verbose)
            self.expert_obs = expert_obs
            self.expert_actions = expert_actions
            self.current_idx = 0
        
        def _on_step(self) -> bool:
            # This callback doesn't modify behavior, just tracks
            return True
    
    # Simpler approach: train the model using expert demonstrations
    # We'll use a small learning rate and train for a few steps
    print("  Training policy network to match expert actions...")
    
    # Get the policy network
    policy = model.policy
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Convert observations and actions to tensors
    obs_tensor = torch.as_tensor(np.array(observations), dtype=torch.float32, device=policy.device)
    actions_tensor = torch.as_tensor(np.array(actions), dtype=torch.long, device=policy.device)
    
    # Training loop
    num_batches = len(observations) // batch_size
    if num_batches == 0:
        num_batches = 1
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Shuffle
        indices = torch.randperm(len(observations), device=policy.device)
        obs_shuffled = obs_tensor[indices]
        actions_shuffled = actions_tensor[indices]
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(observations))
            
            batch_obs = obs_shuffled[start_idx:end_idx]
            batch_actions = actions_shuffled[start_idx:end_idx]
            
            # Forward pass through policy network
            # Extract features
            features = policy.extract_features(batch_obs)
            
            # Get latent representation for actor
            latent_pi = policy.mlp_extractor.forward_actor(features)
            
            # Get action logits (action distribution)
            action_logits = policy.action_net(latent_pi)
            
            # Compute cross-entropy loss
            loss = nn.functional.cross_entropy(action_logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: avg_loss = {avg_loss:.4f}")
    
    print("Pretraining complete!")
    return model


def pretrain_agent_from_fast_ai(
    env,
    model: PPO,
    num_expert_episodes: int = 100,
    pretrain_epochs: int = 10,
) -> PPO:
    """Complete pretraining pipeline: generate dataset and train model.
    
    Args:
        env: Tetris environment
        model: PPO model to pretrain
        num_expert_episodes: Number of episodes to collect from Fast AI
        pretrain_epochs: Number of epochs for behavioral cloning
        
    Returns:
        Pretrained model
    """
    # Generate expert dataset
    observations, actions = generate_expert_dataset(env, num_expert_episodes)
    
    if len(observations) == 0:
        print("Warning: No expert samples collected!")
        return model
    
    # Pretrain model
    model = pretrain_with_behavioral_cloning(
        env, model, observations, actions, epochs=pretrain_epochs
    )
    
    return model


def create_pretrained_model(
    env,
    model_kwargs: dict,
    pretrain: bool = True,
    num_expert_episodes: int = 100,
    pretrain_epochs: int = 10,
    save_path: Optional[str] = None,
) -> PPO:
    """Create and optionally pretrain a PPO model.
    
    Args:
        env: Tetris environment
        model_kwargs: Arguments for PPO constructor
        pretrain: Whether to pretrain with Fast AI
        num_expert_episodes: Number of expert episodes for pretraining
        pretrain_epochs: Number of epochs for behavioral cloning
        save_path: Optional path to save pretrained model
        
    Returns:
        PPO model (pretrained if requested)
    """
    from stable_baselines3 import PPO
    
    # Extract policy (positional arg)
    policy = model_kwargs.pop("policy", "MlpPolicy")
    
    # Create model
    print("Creating PPO model...")
    model = PPO(policy, env, **model_kwargs)
    
    if pretrain:
        print("\nPretraining model with Fast AI decisions...")
        model = pretrain_agent_from_fast_ai(
            env, model, num_expert_episodes, pretrain_epochs
        )
        
        if save_path:
            print(f"Saving pretrained model to {save_path}...")
            model.save(save_path)
    
    return model
