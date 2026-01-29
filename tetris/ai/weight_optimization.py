"""Weight optimization for FastAI heuristic parameters.

Provides multiple approaches to iteratively improve FastAI weights:
1. Evolutionary Strategy (CMA-ES) - gradient-free optimization
2. Bayesian Optimization - sample-efficient black-box optimization
3. Meta-RL - RL agent that learns to set weights
4. Hybrid RL - RL agent that uses FastAI as action space

Usage:
    # Evolutionary optimization
    python -m tetris.ai.weight_optimization evolve --runs 50 --games-per-run 20
    
    # Bayesian optimization
    python -m tetris.ai.weight_optimization bayesian --iterations 30 --games-per-iter 20
    
    # Meta-RL (learns weights via RL)
    python -m tetris.ai.weight_optimization meta_rl --timesteps 100000
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import random

from tetris.ai.fast import FastAIPlayer
from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.simulation.factory import create_renderer


@dataclass
class WeightBounds:
    """Bounds for each weight parameter."""
    # Stack height penalties
    m_stack_w: Tuple[float, float] = (-15.0, -1.0)
    m_stack_e: Tuple[float, float] = (0.5, 3.0)
    a_stack_w: Tuple[float, float] = (-10.0, -1.0)
    a_stack_e: Tuple[float, float] = (0.5, 3.0)
    
    # Cliff penalties
    cliff_l_w: Tuple[float, float] = (-10.0, -1.0)
    cliff_l_e: Tuple[float, float] = (0.5, 3.0)
    cliff_h_w: Tuple[float, float] = (-12.0, -1.0)
    cliff_h_e: Tuple[float, float] = (0.5, 3.0)
    
    # Stack danger
    stack_d_w: Tuple[float, float] = (-8.0, -1.0)
    stack_d_e: Tuple[float, float] = (0.5, 3.0)
    stack_d_thresh: Tuple[int, int] = (10, 20)
    
    # Score rewards
    score_w: Tuple[float, float] = (50.0, 200.0)
    tetris_bonus: Tuple[float, float] = (100.0, 500.0)
    
    # Game over penalty
    go_w: Tuple[float, float] = (-2000.0, -100.0)
    
    def get_param_names(self) -> List[str]:
        """Get list of parameter names."""
        return [
            "m_stack_w", "m_stack_e", "a_stack_w", "a_stack_e",
            "cliff_l_w", "cliff_l_e", "cliff_h_w", "cliff_h_e",
            "stack_d_w", "stack_d_e", "stack_d_thresh",
            "score_w", "tetris_bonus", "go_w",
        ]
    
    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as numpy array for optimization."""
        bounds = []
        for name in self.get_param_names():
            low, high = getattr(self, name)
            bounds.append([low, high])
        return np.array(bounds)
    
    def clip_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clip parameters to valid bounds."""
        clipped = {}
        for name in self.get_param_names():
            value = params[name]
            low, high = getattr(self, name)
            if isinstance(value, float):
                clipped[name] = np.clip(value, low, high)
            else:  # int (stack_d_thresh)
                clipped[name] = int(np.clip(value, low, high))
        return clipped


@dataclass
class EvaluationResult:
    """Result from evaluating a set of weights."""
    params: Dict[str, Any]
    avg_score: float
    avg_lines: float
    avg_moves: float
    std_score: float
    std_lines: float
    num_games: int
    seed: Optional[int] = None
    
    def fitness(self) -> float:
        """Calculate fitness score (higher is better)."""
        # Weighted combination: prioritize score and lines
        # Normalize by typical values to balance
        score_component = self.avg_score / 10000.0  # ~10k is good
        lines_component = self.avg_lines / 50.0     # ~50 lines is good
        moves_component = self.avg_moves / 1000.0  # ~1000 moves is good
        
        # Weighted sum (prioritize score and lines)
        return 0.5 * score_component + 0.4 * lines_component + 0.1 * moves_component


def evaluate_weights(
    params: Dict[str, Any],
    num_games: int = 20,
    seed: Optional[int] = None,
    headless: bool = True,
) -> EvaluationResult:
    """Evaluate a set of weights by running full games.
    
    Plays complete games from start to game over with fixed weights.
    No weight adjustments occur during gameplay - weights are constant
    for the entire duration of each game.
    
    Args:
        params: Weight parameters for FastAI (fixed for entire game)
        num_games: Number of full games to run
        seed: Random seed for reproducibility
        headless: Whether to run headless
        
    Returns:
        EvaluationResult with statistics from completed games
    """
    scores = []
    lines = []
    moves = []
    
    # Clip params to valid bounds
    bounds = WeightBounds()
    params = bounds.clip_params(params)
    
    for game_idx in range(num_games):
        game_seed = seed + game_idx if seed is not None else None
        if game_seed is not None:
            random.seed(game_seed)
            np.random.seed(game_seed)
        
        # Create AI with these weights
        ai = FastAIPlayer(params=params)
        
        # Run simulation
        config = SimulationConfig(
            ai_type="fast",  # Will be overridden
            headless=headless,
            max_moves=10000,
            max_time=None,
            initial_level=1,
            moves_per_second=4,
        )
        
        engine = GameEngine()
        renderer = create_renderer(headless)
        sim = Simulator(engine, ai, renderer, config)
        
        result = sim.run()
        
        scores.append(result.final_state.score)
        lines.append(result.final_state.lines_cleared)
        moves.append(result.moves_executed)
    
    return EvaluationResult(
        params=params,
        avg_score=statistics.mean(scores),
        avg_lines=statistics.mean(lines),
        avg_moves=statistics.mean(moves),
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        std_lines=statistics.stdev(lines) if len(lines) > 1 else 0.0,
        num_games=num_games,
        seed=seed,
    )


def params_to_vector(params: Dict[str, Any], bounds: WeightBounds) -> np.ndarray:
    """Convert params dict to normalized vector [0, 1]."""
    vector = []
    for name in bounds.get_param_names():
        value = params[name]
        low, high = getattr(bounds, name)
        # Normalize to [0, 1]
        normalized = (value - low) / (high - low)
        vector.append(normalized)
    return np.array(vector)


def vector_to_params(vector: np.ndarray, bounds: WeightBounds) -> Dict[str, Any]:
    """Convert normalized vector [0, 1] to params dict."""
    params = {}
    param_names = bounds.get_param_names()
    for i, name in enumerate(param_names):
        normalized = np.clip(vector[i], 0.0, 1.0)
        low, high = getattr(bounds, name)
        value = normalized * (high - low) + low
        if name == "stack_d_thresh":
            params[name] = int(round(value))
        else:
            params[name] = float(value)
    return params


# ============================================================================
# Evolutionary Strategy (CMA-ES)
# ============================================================================

def optimize_evolutionary(
    initial_params: Optional[Dict[str, Any]] = None,
    num_runs: int = 50,
    games_per_run: int = 20,
    population_size: int = 10,
    seed: Optional[int] = 42,
    output_file: Optional[str] = None,
) -> EvaluationResult:
    """Optimize weights using CMA-ES evolutionary strategy.
    
    Process: For each candidate weight set:
    1. Play full game(s) with fixed weights
    2. Get final score/performance
    3. Adjust weights based on performance
    4. Repeat
    
    No weight adjustments occur during gameplay.
    
    Args:
        initial_params: Starting parameters (uses defaults if None)
        num_runs: Number of optimization iterations
        games_per_run: Full games to play per weight evaluation
        population_size: CMA-ES population size
        seed: Random seed
        output_file: Path to save best params JSON
        
    Returns:
        Best EvaluationResult found
    """
    try:
        import cma
    except ImportError:
        raise ImportError(
            "CMA-ES requires 'cma' package. Install with: uv pip install cma"
        )
    
    bounds = WeightBounds()
    
    # Start from initial params or defaults
    if initial_params is None:
        initial_params = FastAIPlayer._get_improved_default_params()
    
    # Convert to normalized vector
    x0 = params_to_vector(initial_params, bounds)
    
    # CMA-ES options
    sigma0 = 0.3  # Initial standard deviation (30% of range)
    options = {
        "maxiter": num_runs,
        "popsize": population_size,
        "seed": seed,
        "verb_disp": 1,
        "verb_log": 1,
    }
    
    best_result = None
    best_fitness = float("-inf")
    history = []
    
    def objective(x: np.ndarray) -> float:
        """Objective function: negative fitness (CMA-ES minimizes)."""
        nonlocal best_result, best_fitness
        
        params = vector_to_params(x, bounds)
        result = evaluate_weights(params, num_games=games_per_run, seed=seed)
        fitness = result.fitness()
        
        history.append({
            "iteration": len(history),
            "fitness": fitness,
            "avg_score": result.avg_score,
            "avg_lines": result.avg_lines,
            "params": params,
        })
        
        # Track best
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = result
        
        return -fitness  # Minimize negative fitness
    
    print(f"Starting CMA-ES optimization...")
    print(f"  Runs: {num_runs}")
    print(f"  Games per run: {games_per_run}")
    print(f"  Population size: {population_size}")
    print(f"  Initial params: {initial_params}")
    print()
    
    # Run CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma0, options)
    es.optimize(objective)
    
    print(f"\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if best_result:
        print(f"Best avg score: {best_result.avg_score:.1f}")
        print(f"Best avg lines: {best_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_result.params.items():
            print(f"  {key}: {value}")
    
    # Save best params
    if output_file and best_result:
        with open(output_file, 'w') as f:
            json.dump({
                "params": best_result.params,
                "fitness": best_fitness,
                "avg_score": best_result.avg_score,
                "avg_lines": best_result.avg_lines,
                "avg_moves": best_result.avg_moves,
                "num_games": best_result.num_games,
            }, f, indent=2)
        print(f"\nSaved best params to {output_file}")
    
    return best_result


# ============================================================================
# Bayesian Optimization
# ============================================================================

def optimize_bayesian(
    initial_params: Optional[Dict[str, Any]] = None,
    num_iterations: int = 30,
    games_per_iter: int = 20,
    seed: Optional[int] = 42,
    output_file: Optional[str] = None,
) -> EvaluationResult:
    """Optimize weights using Bayesian Optimization.
    
    Process: For each candidate weight set:
    1. Play full game(s) with fixed weights
    2. Get final score/performance
    3. Use Bayesian optimization to select next weights
    4. Repeat
    
    No weight adjustments occur during gameplay.
    
    Args:
        initial_params: Starting parameters
        num_iterations: Number of BO iterations
        games_per_iter: Full games to play per weight evaluation
        seed: Random seed
        output_file: Path to save best params JSON
        
    Returns:
        Best EvaluationResult found
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
        from skopt.utils import use_named_args
    except ImportError:
        raise ImportError(
            "Bayesian Optimization requires 'scikit-optimize' package. "
            "Install with: uv pip install scikit-optimize"
        )
    
    bounds = WeightBounds()
    
    # Define search space
    dimensions = []
    param_names = bounds.get_param_names()
    
    for name in param_names:
        low, high = getattr(bounds, name)
        if name == "stack_d_thresh":
            dimensions.append(Integer(low, high, name=name))
        else:
            dimensions.append(Real(low, high, name=name))
    
    # Initial params
    if initial_params is None:
        initial_params = FastAIPlayer._get_improved_default_params()
    
    x0 = [[initial_params[name] for name in param_names]]
    
    best_result = None
    best_fitness = float("-inf")
    history = []
    
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        """Objective function: negative fitness."""
        nonlocal best_result, best_fitness
        
        result = evaluate_weights(params, num_games=games_per_iter, seed=seed)
        fitness = result.fitness()
        
        history.append({
            "iteration": len(history),
            "fitness": fitness,
            "avg_score": result.avg_score,
            "avg_lines": result.avg_lines,
            "params": params,
        })
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_result = result
        
        return -fitness  # Minimize negative fitness
    
    print(f"Starting Bayesian Optimization...")
    print(f"  Iterations: {num_iterations}")
    print(f"  Games per iteration: {games_per_iter}")
    print(f"  Initial params: {initial_params}")
    print()
    
    # Run BO
    result_bo = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=num_iterations,
        x0=x0,
        random_state=seed,
        n_initial_points=5,
        acq_func="EI",  # Expected Improvement
    )
    
    print(f"\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if best_result:
        print(f"Best avg score: {best_result.avg_score:.1f}")
        print(f"Best avg lines: {best_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_result.params.items():
            print(f"  {key}: {value}")
    
    # Save best params
    if output_file and best_result:
        with open(output_file, 'w') as f:
            json.dump({
                "params": best_result.params,
                "fitness": best_fitness,
                "avg_score": best_result.avg_score,
                "avg_lines": best_result.avg_lines,
                "avg_moves": best_result.avg_moves,
                "num_games": best_result.num_games,
            }, f, indent=2)
        print(f"\nSaved best params to {output_file}")
    
    return best_result


# ============================================================================
# Meta-RL: RL agent that learns to set weights
# ============================================================================

def optimize_meta_rl(
    timesteps: int = 100000,
    games_per_eval: int = 5,
    seed: Optional[int] = 42,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Use RL to learn optimal weights (meta-learning).
    
    Episodic RL where each episode:
    1. Agent sets weights (action)
    2. Play full game(s) with those fixed weights
    3. Get final score/performance
    4. Reward based on performance
    5. Episode ends, next episode starts
    
    No weight adjustments occur during gameplay - weights are fixed
    for the entire duration of each game.
    
    Creates an RL environment where:
    - State: Previous weights + previous performance
    - Action: New weight values to try
    - Reward: Performance from playing full game(s) with those weights
    
    Args:
        timesteps: Total training timesteps
        games_per_eval: Full games to play per weight evaluation
        seed: Random seed
        output_file: Path to save best params JSON
        
    Returns:
        Best parameters found
    """
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        RL_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "Meta-RL requires RL dependencies. "
            "Install with: uv pip install gymnasium stable-baselines3"
        )
    
    bounds = WeightBounds()
    param_names = bounds.get_param_names()
    n_params = len(param_names)
    
    class WeightOptimizationEnv(gym.Env):
        """RL environment for weight optimization.
        
        Episodic: Each episode is:
        1. Set weights (from action)
        2. Play full game(s) with those weights
        3. Get final score/performance
        4. Reward based on performance
        5. Episode done, reset for next episode
        """
        
        def __init__(self):
            super().__init__()
            
            # Action: weight values (normalized [0, 1])
            # Agent directly sets weights, not adjustments
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_params,), dtype=np.float32
            )
            
            # State: previous weights + previous performance (for context)
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(n_params + 3,), dtype=np.float32
            )
            
            # Previous episode's weights and performance
            self.prev_params = FastAIPlayer._get_improved_default_params()
            self.prev_vector = params_to_vector(self.prev_params, bounds)
            self.prev_score = 0.0
            self.prev_lines = 0.0
            self.prev_moves = 0.0
        
        def reset(self, seed=None, options=None):
            """Reset environment for new episode."""
            super().reset(seed=seed)
            # Keep previous episode's info for observation
            # (weights will be set by action in first step)
            return self._get_obs(), {}
        
        def step(self, action):
            """Take step: set weights, play full game(s), get score.
            
            Action sets the weights directly. Then we play full game(s)
            with those weights and return the final performance as reward.
            """
            # Action directly sets weights (normalized [0, 1])
            new_vector = np.clip(action, 0.0, 1.0)
            new_params = vector_to_params(new_vector, bounds)
            
            # Play full game(s) with these weights
            # No weight changes during gameplay - weights are fixed for entire game(s)
            result = evaluate_weights(
                new_params, num_games=games_per_eval, seed=seed, headless=True
            )
            
            # Calculate reward from final game performance
            fitness = result.fitness()
            reward = fitness * 10.0  # Scale up for RL
            
            # Update previous episode info for next observation
            self.prev_params = new_params
            self.prev_vector = new_vector
            self.prev_score = result.avg_score
            self.prev_lines = result.avg_lines
            self.prev_moves = result.avg_moves
            
            # Episode is done after each evaluation
            # Next episode will start with reset(), then new weights from action
            done = True
            
            return self._get_obs(), reward, done, False, {
                "fitness": fitness,
                "avg_score": result.avg_score,
                "avg_lines": result.avg_lines,
                "avg_moves": result.avg_moves,
                "params": new_params,
            }
        
        def _get_obs(self):
            """Get current observation: previous weights + previous performance."""
            # Normalize performance metrics
            score_norm = np.clip(self.prev_score / 20000.0, 0.0, 1.0)
            lines_norm = np.clip(self.prev_lines / 100.0, 0.0, 1.0)
            moves_norm = np.clip(self.prev_moves / 2000.0, 0.0, 1.0)
            
            obs = np.concatenate([
                self.prev_vector,
                [score_norm, lines_norm, moves_norm]
            ]).astype(np.float32)
            
            return obs
    
    print(f"Starting Meta-RL optimization...")
    print(f"  Timesteps: {timesteps}")
    print(f"  Games per eval: {games_per_eval}")
    print()
    
    # Create environment
    env = WeightOptimizationEnv()
    
    # Create PPO agent
    # Note: Since episodes are very short (one evaluation each),
    # we use smaller n_steps to update more frequently
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=32,  # Smaller since episodes are single evaluations
        batch_size=32,
        n_epochs=10,
        gamma=0.99,  # Still use discounting for multi-step learning
        verbose=1,
        seed=seed,
    )
    
    # Track best during training
    best_params = None
    best_fitness = float("-inf")
    
    def track_best_callback(locals_, globals_):
        """Callback to track best weights during training."""
        nonlocal best_params, best_fitness
        
        if "infos" in locals_:
            for info in locals_["infos"]:
                if info and "params" in info:
                    fitness = info.get("fitness", 0.0)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = info["params"].copy()
        return True
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=track_best_callback,
        progress_bar=True,
    )
    
    # Final evaluation
    if best_params:
        final_result = evaluate_weights(
            best_params, num_games=games_per_eval * 4, seed=seed
        )
        print(f"\nTraining complete!")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best avg score: {final_result.avg_score:.1f}")
        print(f"Best avg lines: {final_result.avg_lines:.1f}")
        print(f"Best params:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        # Save
        if output_file:
            with open(output_file, 'w') as f:
                json.dump({
                    "params": best_params,
                    "fitness": best_fitness,
                    "avg_score": final_result.avg_score,
                    "avg_lines": final_result.avg_lines,
                    "avg_moves": final_result.avg_moves,
                }, f, indent=2)
            print(f"\nSaved best params to {output_file}")
        
        return best_params
    
    return env.current_params


# ============================================================================
# Command-line interface
# ============================================================================

def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize FastAI heuristic weights"
    )
    subparsers = parser.add_subparsers(dest="method", help="Optimization method")
    
    # Evolutionary
    evolve_parser = subparsers.add_parser("evolve", help="CMA-ES evolutionary optimization")
    evolve_parser.add_argument("--runs", type=int, default=50, help="Number of iterations")
    evolve_parser.add_argument("--games-per-run", type=int, default=20, help="Games per evaluation")
    evolve_parser.add_argument("--population-size", type=int, default=10, help="CMA-ES population size")
    evolve_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    evolve_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Bayesian
    bayesian_parser = subparsers.add_parser("bayesian", help="Bayesian optimization")
    bayesian_parser.add_argument("--iterations", type=int, default=30, help="Number of iterations")
    bayesian_parser.add_argument("--games-per-iter", type=int, default=20, help="Games per iteration")
    bayesian_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    bayesian_parser.add_argument("--output", type=str, help="Output JSON file")
    
    # Meta-RL
    meta_rl_parser = subparsers.add_parser("meta_rl", help="Meta-RL optimization")
    meta_rl_parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    meta_rl_parser.add_argument("--games-per-eval", type=int, default=5, help="Games per evaluation")
    meta_rl_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    meta_rl_parser.add_argument("--output", type=str, help="Output JSON file")
    
    args = parser.parse_args()
    
    if args.method == "evolve":
        optimize_evolutionary(
            num_runs=args.runs,
            games_per_run=args.games_per_run,
            population_size=args.population_size,
            seed=args.seed,
            output_file=args.output,
        )
    elif args.method == "bayesian":
        optimize_bayesian(
            num_iterations=args.iterations,
            games_per_iter=args.games_per_iter,
            seed=args.seed,
            output_file=args.output,
        )
    elif args.method == "meta_rl":
        optimize_meta_rl(
            timesteps=args.timesteps,
            games_per_eval=args.games_per_eval,
            seed=args.seed,
            output_file=args.output,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
