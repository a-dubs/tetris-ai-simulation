"""Benchmark RL agent against baseline AIs.

Compares RL agent performance with existing AI implementations
(greedy, fast, random) to evaluate training progress.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import statistics

try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

from tetris.core.game_engine import GameEngine
from tetris.core.game_state import GameState
from tetris.simulation.simulator import Simulator
from tetris.simulation.config import SimulationConfig
from tetris.simulation.factory import create_ai, create_renderer
from tetris.rl.env import TetrisEnv


def run_baseline_ai(ai_type: str, num_games: int = 10) -> Dict[str, List[float]]:
    """Run baseline AI and collect statistics.
    
    Args:
        ai_type: Type of AI ("greedy", "fast", "random")
        num_games: Number of games to run
        
    Returns:
        Dictionary with statistics lists
    """
    results = {
        "scores": [],
        "lines": [],
        "levels": [],
        "moves": [],
    }
    
    print(f"\nRunning {ai_type} AI for {num_games} games...")
    
    for game in range(num_games):
        config = SimulationConfig(
            ai_type=ai_type,
            headless=True,
            max_moves=10000,
            max_time=None,
            initial_level=1,
            moves_per_second=4,
        )
        
        engine = GameEngine()
        ai = create_ai(config.ai_type, config)
        renderer = create_renderer(config.headless)
        sim = Simulator(engine, ai, renderer, config)
        
        result = sim.run()
        
        results["scores"].append(result.final_state.score)
        results["lines"].append(result.final_state.lines_cleared)
        results["levels"].append(result.final_state.level)
        results["moves"].append(result.moves_executed)
        
        if (game + 1) % 5 == 0:
            print(f"  Completed {game + 1}/{num_games} games...")
    
    return results


def run_rl_agent(model_path: str, num_games: int = 10) -> Dict[str, List[float]]:
    """Run RL agent and collect statistics.
    
    Args:
        model_path: Path to trained model
        num_games: Number of games to run
        
    Returns:
        Dictionary with statistics lists
    """
    if not RL_AVAILABLE:
        print("Error: RL dependencies not available")
        return {}
    
    results = {
        "scores": [],
        "lines": [],
        "levels": [],
        "moves": [],
    }
    
    print(f"\nRunning RL agent for {num_games} games...")
    print(f"Loading model from {model_path}...")
    
    model = PPO.load(model_path)
    env = TetrisEnv()
    
    for game in range(num_games):
        obs, info = env.reset()
        done = False
        moves = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            moves += 1
            
            if moves >= 10000:  # Safety limit
                break
        
        results["scores"].append(info["score"])
        results["lines"].append(info["lines_cleared"])
        results["levels"].append(info["level"])
        results["moves"].append(moves)
        
        if (game + 1) % 5 == 0:
            print(f"  Completed {game + 1}/{num_games} games...")
    
    env.close()
    return results


def print_statistics(name: str, results: Dict[str, List[float]]):
    """Print statistics for an AI.
    
    Args:
        name: AI name
        results: Results dictionary
    """
    if not results or not results["scores"]:
        print(f"\n{name}: No results")
        return
    
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Games: {len(results['scores'])}")
    print()
    print("Score:")
    print(f"  Mean: {statistics.mean(results['scores']):,.0f}")
    print(f"  Std:  {statistics.stdev(results['scores']):,.0f}")
    print(f"  Min:  {min(results['scores']):,.0f}")
    print(f"  Max:  {max(results['scores']):,.0f}")
    print()
    print("Lines Cleared:")
    print(f"  Mean: {statistics.mean(results['lines']):.1f}")
    print(f"  Std:  {statistics.stdev(results['lines']):.1f}")
    print(f"  Min:  {min(results['lines'])}")
    print(f"  Max:  {max(results['lines'])}")
    print()
    print("Level:")
    print(f"  Mean: {statistics.mean(results['levels']):.1f}")
    print(f"  Std:  {statistics.stdev(results['levels']):.1f}")
    print(f"  Max:  {max(results['levels'])}")
    print()
    print("Moves:")
    print(f"  Mean: {statistics.mean(results['moves']):.1f}")
    print(f"  Std:  {statistics.stdev(results['moves']):.1f}")


def compare_all(model_path: str, num_games: int = 10):
    """Compare RL agent with all baseline AIs.
    
    Args:
        model_path: Path to RL model
        num_games: Number of games per AI
    """
    print("="*60)
    print("Tetris AI Benchmark")
    print("="*60)
    print(f"Running {num_games} games per AI...")
    
    # Run baselines
    baselines = {}
    for ai_type in ["random", "fast", "greedy"]:
        try:
            baselines[ai_type] = run_baseline_ai(ai_type, num_games)
            print_statistics(f"{ai_type.capitalize()} AI", baselines[ai_type])
        except Exception as e:
            print(f"Error running {ai_type} AI: {e}")
            baselines[ai_type] = {}
    
    # Run RL agent
    rl_results = run_rl_agent(model_path, num_games)
    print_statistics("RL Agent", rl_results)
    
    # Comparison summary
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    
    if rl_results and rl_results["scores"]:
        rl_score = statistics.mean(rl_results["scores"])
        rl_lines = statistics.mean(rl_results["lines"])
        
        print(f"\nRL Agent:")
        print(f"  Average Score: {rl_score:,.0f}")
        print(f"  Average Lines: {rl_lines:.1f}")
        
        for ai_type, results in baselines.items():
            if results and results["scores"]:
                baseline_score = statistics.mean(results["scores"])
                baseline_lines = statistics.mean(results["lines"])
                
                score_ratio = rl_score / baseline_score if baseline_score > 0 else 0
                lines_ratio = rl_lines / baseline_lines if baseline_lines > 0 else 0
                
                print(f"\n{ai_type.capitalize()} AI:")
                print(f"  Average Score: {baseline_score:,.0f}")
                print(f"  Average Lines: {baseline_lines:.1f}")
                print(f"  RL vs {ai_type}:")
                print(f"    Score ratio: {score_ratio:.2f}x")
                print(f"    Lines ratio: {lines_ratio:.2f}x")
    
    print("\n" + "="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark RL agent against baseline AIs"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Path to RL model file")
    parser.add_argument("--games", type=int, default=10,
                       help="Number of games per AI")
    parser.add_argument("--baseline", type=str, choices=["random", "fast", "greedy"],
                       help="Compare only with specific baseline")
    
    args = parser.parse_args()
    
    if args.baseline:
        # Compare with single baseline
        baseline_results = run_baseline_ai(args.baseline, args.games)
        print_statistics(f"{args.baseline.capitalize()} AI", baseline_results)
        
        rl_results = run_rl_agent(args.model, args.games)
        print_statistics("RL Agent", rl_results)
    else:
        # Compare with all baselines
        compare_all(args.model, args.games)


if __name__ == "__main__":
    main()
