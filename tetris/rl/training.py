"""Main training script for RL Tetris agent.

This script provides a complete training pipeline with:
- Environment setup
- Model training with PPO
- Periodic evaluation
- Metrics tracking
- Visualization
- Checkpointing

Usage:
    uv run python -m tetris.rl.training train --timesteps 100000
    uv run python -m tetris.rl.training evaluate --model model.zip
    uv run python -m tetris.rl.training visualize --model model.zip
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Check for RL dependencies
try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        CallbackList,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError as e:
    RL_AVAILABLE = False
    print(f"RL dependencies not available: {e}")
    print("Install with: uv pip install gymnasium stable-baselines3")

from tetris.rl.env import TetrisEnv, TrainingMetrics
from tetris.rl.visualization import TrainingVisualizer
from tetris.rl.config import RLConfig, load_config


class TrainingCallback:
    """Custom callback to track metrics during training."""
    
    def __init__(self, visualizer: TrainingVisualizer):
        """Initialize callback.
        
        Args:
            visualizer: TrainingVisualizer instance
        """
        self.visualizer = visualizer
        self.episode_count = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.current_episode_info = {}
    
    def on_step(self) -> bool:
        """Called on each step. Track episode metrics."""
        # This would be called by stable-baselines3 during training
        # For now, we'll track metrics manually in the training loop
        return True
    
    def add_episode_metrics(self, reward: float, info: dict, length: int):
        """Add metrics for completed episode.
        
        Args:
            reward: Total episode reward
            info: Episode info dict
            length: Episode length in steps
        """
        self.episode_count += 1
        self.visualizer.add_episode(
            episode=self.episode_count,
            reward=reward,
            score=info.get('score', 0),
            lines_cleared=info.get('lines_cleared', 0),
            level=info.get('level', 1),
            length=length,
        )


def train_agent(
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_steps: int = 2048,
    n_epochs: int = 10,
    gamma: float = 0.99,
    model_name: str = "tetris_rl",
    log_dir: str = "./rl_logs",
    checkpoint_freq: int = 50_000,
    eval_freq: int = 10_000,
    render_every: int = 100,
    verbose: int = 1,
    config: Optional[RLConfig] = None,
    agent_config: Optional[str] = None,
    scenario_config: Optional[str] = None,
):
    """Train RL agent with comprehensive logging and visualization.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for PPO
        batch_size: Batch size for training
        n_steps: Steps per update
        n_epochs: Epochs per update
        gamma: Discount factor
        model_name: Base name for saved models
        log_dir: Directory for logs and checkpoints
        checkpoint_freq: Frequency of model checkpoints
        eval_freq: Frequency of evaluation
        render_every: Render every N episodes (0 to disable)
        verbose: Verbosity level
    """
    if not RL_AVAILABLE:
        print("Error: RL dependencies not available")
        return
    
    # Load config if provided
    if config is None:
        if agent_config or scenario_config:
            config = load_config(agent_config=agent_config, scenario_config=scenario_config)
        else:
            # Use defaults
            config = load_config()
    
    # Override config with explicit parameters if provided
    if total_timesteps != 1_000_000:
        config.total_timesteps = total_timesteps
    if model_name != "tetris_rl":
        config.model_name = model_name
    if log_dir != "./rl_logs":
        config.log_dir = log_dir
    if checkpoint_freq != 50_000:
        config.checkpoint_freq = checkpoint_freq
    if eval_freq != 10_000:
        config.scenario.training["eval_freq"] = eval_freq
    if render_every != 100:
        config.render_every = render_every
    
    # Override agent config with explicit params if provided
    if learning_rate != 3e-4:
        config.agent.learning_rate = learning_rate
    if batch_size != 64:
        config.agent.batch_size = batch_size
    if n_steps != 2048:
        config.agent.n_steps = n_steps
    if n_epochs != 10:
        config.agent.n_epochs = n_epochs
    if gamma != 0.99:
        config.agent.gamma = gamma
    if verbose != 1:
        config.agent.verbose = verbose
    
    # Create directories
    log_path = Path(config.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = log_path / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)
    tensorboard_path = log_path / "tensorboard"
    tensorboard_path.mkdir(exist_ok=True)
    
    # Create environments with scenario config
    print("Creating environments...")
    reward_params = config.scenario.get_reward_params()
    env_params = config.scenario.get_env_params()
    
    train_env = TetrisEnv(
        render_mode=env_params.get("render_mode"),
        reward_params=reward_params
    )
    train_env = Monitor(train_env, str(log_path / "train_monitor"))
    train_env = DummyVecEnv([lambda: train_env])
    
    eval_env = TetrisEnv(
        render_mode=env_params.get("render_mode"),
        reward_params=reward_params
    )
    eval_env = Monitor(eval_env, str(log_path / "eval_monitor"))
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create model with agent config
    print("Initializing PPO model...")
    model_kwargs = config.agent.to_dict()
    model_kwargs["tensorboard_log"] = str(tensorboard_path)
    # Extract policy from kwargs (it's a positional arg to PPO)
    policy = model_kwargs.pop("policy")
    model = PPO(policy, train_env, **model_kwargs)
    
    # Setup callbacks
    visualizer = TrainingVisualizer(log_dir=str(log_path))
    training_callback = TrainingCallback(visualizer)
    
    callbacks = []
    
    # Evaluation callback
    training_params = config.scenario.get_training_params()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(checkpoint_path / "best"),
        log_path=str(log_path / "eval"),
        eval_freq=training_params.get("eval_freq", eval_freq),
        deterministic=True,
        render=False,
        verbose=config.agent.verbose,
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(checkpoint_path),
        name_prefix=config.model_name,
        verbose=config.agent.verbose,
    )
    callbacks.append(checkpoint_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Training loop
    print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
    print(f"Agent config: {config.agent.algorithm} ({config.agent.policy})")
    print(f"Scenario: reward-based training")
    print(f"Checkpoint frequency: {config.checkpoint_freq:,} timesteps")
    print(f"Evaluation frequency: {training_params.get('eval_freq', eval_freq):,} timesteps")
    if config.render_every > 0:
        print(f"Rendering every {config.render_every} episodes")
    print()
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_model_path = log_path / f"{config.model_name}_final"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to {final_model_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    visualizer.print_statistics(recent_n=100)
    
    # Generate plots
    plot_path = log_path / "training_progress.png"
    visualizer.plot_training_progress(window_size=100, save_path=str(plot_path))
    print(f"\nTraining plot saved to {plot_path}")
    
    # Cleanup
    train_env.close()
    eval_env.close()


def evaluate_agent(
    model_path: str,
    num_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True,
):
    """Evaluate trained agent.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        deterministic: Use deterministic policy
    """
    if not RL_AVAILABLE:
        print("Error: RL dependencies not available")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment
    env = TetrisEnv(render_mode="human" if render else None)
    
    # Run episodes
    results = {
        "rewards": [],
        "scores": [],
        "lines_cleared": [],
        "levels": [],
        "lengths": [],
    }
    
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
        
        results["rewards"].append(total_reward)
        results["scores"].append(info["score"])
        results["lines_cleared"].append(info["lines_cleared"])
        results["levels"].append(info["level"])
        results["lengths"].append(steps)
        
        print(f"Episode {episode + 1:3d}: "
              f"Score={info['score']:6,}, "
              f"Lines={info['lines_cleared']:3d}, "
              f"Level={info['level']:2d}, "
              f"Reward={total_reward:7.2f}, "
              f"Steps={steps:4d}")
    
    env.close()
    
    # Print statistics
    print("-" * 60)
    print("\nEvaluation Results:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Mean Score: {sum(results['scores'])/len(results['scores']):,.0f}")
    print(f"  Mean Lines: {sum(results['lines_cleared'])/len(results['lines_cleared']):.1f}")
    print(f"  Mean Reward: {sum(results['rewards'])/len(results['rewards']):.2f}")
    print(f"  Mean Length: {sum(results['lengths'])/len(results['lengths']):.1f} steps")
    print(f"  Best Score: {max(results['scores']):,}")
    print(f"  Best Lines: {max(results['lines_cleared'])}")


def visualize_agent(
    model_path: str,
    num_episodes: int = 3,
    deterministic: bool = True,
):
    """Visualize agent playing Tetris.
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to visualize
        deterministic: Use deterministic policy
    """
    if not RL_AVAILABLE:
        print("Error: RL dependencies not available")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = TetrisEnv(render_mode="human")
    
    print(f"\nVisualizing {num_episodes} episodes...")
    print("Close window or press ESC to skip episode")
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            if not env.renderer or not env.renderer.is_running():
                print("Window closed, skipping episode")
                break
            
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        if env.renderer and env.renderer.is_running():
            print(f"Episode complete: Score={info['score']:,}, "
                  f"Lines={info['lines_cleared']}, Reward={total_reward:.2f}")
    
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL Tetris agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train RL agent")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000,
                             help="Total training timesteps")
    train_parser.add_argument("--agent-config", type=str, default=None,
                             help="Agent config name or path (e.g., 'default', 'large_network')")
    train_parser.add_argument("--scenario-config", type=str, default=None,
                             help="Scenario config name or path (e.g., 'default', 'aggressive_rewards')")
    train_parser.add_argument("--lr", type=float, default=None,
                             help="Learning rate (overrides config)")
    train_parser.add_argument("--batch-size", type=int, default=None,
                             help="Batch size (overrides config)")
    train_parser.add_argument("--model-name", type=str, default="tetris_rl",
                             help="Model name prefix")
    train_parser.add_argument("--log-dir", type=str, default="./rl_logs",
                             help="Log directory")
    train_parser.add_argument("--checkpoint-freq", type=int, default=50_000,
                             help="Checkpoint frequency")
    train_parser.add_argument("--eval-freq", type=int, default=10_000,
                             help="Evaluation frequency")
    train_parser.add_argument("--render-every", type=int, default=0,
                             help="Render every N episodes (0 to disable)")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained agent")
    eval_parser.add_argument("--model", type=str, required=True,
                            help="Path to model file")
    eval_parser.add_argument("--episodes", type=int, default=10,
                            help="Number of episodes")
    eval_parser.add_argument("--render", action="store_true",
                            help="Render episodes")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize agent playing")
    viz_parser.add_argument("--model", type=str, required=True,
                            help="Path to model file")
    viz_parser.add_argument("--episodes", type=int, default=3,
                            help="Number of episodes")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "train":
        train_kwargs = {
            "total_timesteps": args.timesteps,
            "agent_config": args.agent_config,
            "scenario_config": args.scenario_config,
            "model_name": args.model_name,
            "log_dir": args.log_dir,
            "checkpoint_freq": args.checkpoint_freq,
            "eval_freq": args.eval_freq,
            "render_every": args.render_every,
        }
        # Only add override params if explicitly provided
        if args.lr is not None:
            train_kwargs["learning_rate"] = args.lr
        if args.batch_size is not None:
            train_kwargs["batch_size"] = args.batch_size
        
        train_agent(**train_kwargs)
    elif args.command == "evaluate":
        evaluate_agent(
            model_path=args.model,
            num_episodes=args.episodes,
            render=args.render,
        )
    elif args.command == "visualize":
        visualize_agent(
            model_path=args.model,
            num_episodes=args.episodes,
        )


if __name__ == "__main__":
    main()
