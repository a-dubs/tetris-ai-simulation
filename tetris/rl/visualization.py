"""Visualization tools for RL training progress.

Provides utilities for:
- Plotting training metrics over time
- Comparing agent performance vs baselines
- Creating training dashboards
- Recording and replaying episodes
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode: int
    reward: float
    score: int
    lines_cleared: int
    level: int
    length: int  # Number of steps
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TrainingVisualizer:
    """Visualize RL training progress and metrics."""
    
    def __init__(self, log_dir: str = "./rl_logs"):
        """Initialize visualizer.
        
        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "metrics.json"
        self.metrics: List[EpisodeMetrics] = []
        self._load_metrics()
    
    def _load_metrics(self):
        """Load metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = [EpisodeMetrics(**m) for m in data]
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
                self.metrics = []
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, 'w') as f:
            json.dump([asdict(m) for m in self.metrics], f, indent=2)
    
    def add_episode(self, episode: int, reward: float, score: int, 
                   lines_cleared: int, level: int, length: int):
        """Add metrics for an episode.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            score: Final score
            lines_cleared: Lines cleared
            level: Final level
            length: Episode length in steps
        """
        metric = EpisodeMetrics(
            episode=episode,
            reward=reward,
            score=score,
            lines_cleared=lines_cleared,
            level=level,
            length=length,
        )
        self.metrics.append(metric)
        self._save_metrics()
    
    def plot_training_progress(self, window_size: int = 100, 
                              save_path: Optional[str] = None):
        """Plot training progress over time.
        
        Args:
            window_size: Size of moving average window
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib")
            return
        
        if len(self.metrics) == 0:
            print("No metrics to plot")
            return
        
        episodes = [m.episode for m in self.metrics]
        rewards = [m.reward for m in self.metrics]
        scores = [m.score for m in self.metrics]
        lines = [m.lines_cleared for m in self.metrics]
        
        # Calculate moving averages
        def moving_average(data, window):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=16)
        
        # Episode reward
        ax = axes[0, 0]
        ax.plot(episodes, rewards, alpha=0.3, label='Raw', color='lightblue')
        if len(rewards) >= window_size:
            ma_rewards = moving_average(rewards, window_size)
            ma_episodes = episodes[window_size-1:]
            ax.plot(ma_episodes, ma_rewards, label=f'MA({window_size})', 
                   color='blue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Score
        ax = axes[0, 1]
        ax.plot(episodes, scores, alpha=0.3, label='Raw', color='lightgreen')
        if len(scores) >= window_size:
            ma_scores = moving_average(scores, window_size)
            ma_episodes = episodes[window_size-1:]
            ax.plot(ma_episodes, ma_scores, label=f'MA({window_size})', 
                   color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Episode Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Lines cleared
        ax = axes[1, 0]
        ax.plot(episodes, lines, alpha=0.3, label='Raw', color='lightcoral')
        if len(lines) >= window_size:
            ma_lines = moving_average(lines, window_size)
            ma_episodes = episodes[window_size-1:]
            ax.plot(ma_episodes, ma_lines, label=f'MA({window_size})', 
                   color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Lines Cleared')
        ax.set_title('Lines Cleared per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Episode length
        lengths = [m.length for m in self.metrics]
        ax = axes[1, 1]
        ax.plot(episodes, lengths, alpha=0.3, label='Raw', color='lightyellow')
        if len(lengths) >= window_size:
            ma_lengths = moving_average(lengths, window_size)
            ma_episodes = episodes[window_size-1:]
            ax.plot(ma_episodes, ma_lengths, label=f'MA({window_size})', 
                   color='orange', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def plot_comparison(self, baseline_metrics: Dict[str, float],
                       save_path: Optional[str] = None):
        """Compare RL agent performance with baseline AIs.
        
        Args:
            baseline_metrics: Dict with baseline performance metrics
                e.g., {"greedy": {"score": 5000, "lines": 200}, ...}
            save_path: Optional path to save figure
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available")
            return
        
        if len(self.metrics) == 0:
            print("No metrics to plot")
            return
        
        # Calculate recent RL performance
        recent = min(100, len(self.metrics))
        recent_metrics = self.metrics[-recent:]
        rl_score = np.mean([m.score for m in recent_metrics])
        rl_lines = np.mean([m.lines_cleared for m in recent_metrics])
        
        # Prepare data
        agents = ["RL Agent"] + list(baseline_metrics.keys())
        scores = [rl_score] + [baseline_metrics[a]["score"] 
                               for a in baseline_metrics.keys()]
        lines = [rl_lines] + [baseline_metrics[a]["lines"] 
                              for a in baseline_metrics.keys()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Score comparison
        bars1 = ax1.bar(agents, scores, color=['blue'] + ['gray'] * len(baseline_metrics))
        ax1.set_ylabel('Average Score')
        ax1.set_title('Score Comparison')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, (bar, score) in enumerate(zip(bars1, scores)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(score):,}',
                    ha='center', va='bottom')
        
        # Lines comparison
        bars2 = ax2.bar(agents, lines, color=['blue'] + ['gray'] * len(baseline_metrics))
        ax2.set_ylabel('Average Lines Cleared')
        ax2.set_title('Lines Cleared Comparison')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, (bar, line) in enumerate(zip(bars2, lines)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(line):,}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved comparison plot to {save_path}")
        else:
            plt.show()
    
    def get_statistics(self, recent_n: int = 100) -> Dict[str, float]:
        """Get statistics for recent episodes.
        
        Args:
            recent_n: Number of recent episodes to analyze
            
        Returns:
            Dictionary with statistics
        """
        if len(self.metrics) == 0:
            return {}
        
        recent = min(recent_n, len(self.metrics))
        recent_metrics = self.metrics[-recent:]
        
        return {
            "episodes": len(self.metrics),
            "recent_episodes": recent,
            "mean_reward": np.mean([m.reward for m in recent_metrics]),
            "std_reward": np.std([m.reward for m in recent_metrics]),
            "mean_score": np.mean([m.score for m in recent_metrics]),
            "std_score": np.std([m.score for m in recent_metrics]),
            "mean_lines": np.mean([m.lines_cleared for m in recent_metrics]),
            "std_lines": np.std([m.lines_cleared for m in recent_metrics]),
            "mean_length": np.mean([m.length for m in recent_metrics]),
            "best_score": max([m.score for m in recent_metrics]),
            "best_lines": max([m.lines_cleared for m in recent_metrics]),
        }
    
    def print_statistics(self, recent_n: int = 100):
        """Print statistics to console."""
        stats = self.get_statistics(recent_n)
        if not stats:
            print("No metrics available")
            return
        
        print("\n" + "="*60)
        print("Training Statistics")
        print("="*60)
        print(f"Total Episodes: {stats['episodes']}")
        print(f"Analyzing Last: {stats['recent_episodes']} episodes")
        print()
        print("Reward:")
        print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print()
        print("Score:")
        print(f"  Mean: {stats['mean_score']:,.0f} ± {stats['std_score']:,.0f}")
        print(f"  Best: {stats['best_score']:,}")
        print()
        print("Lines Cleared:")
        print(f"  Mean: {stats['mean_lines']:.1f} ± {stats['std_lines']:.1f}")
        print(f"  Best: {stats['best_lines']}")
        print()
        print("Episode Length:")
        print(f"  Mean: {stats['mean_length']:.1f} steps")
        print("="*60 + "\n")


def load_tensorboard_logs(log_dir: str) -> Dict:
    """Load metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard event files
        
    Returns:
        Dictionary with scalar metrics
    """
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available. Install with: pip install tensorboard")
        return {}
    
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        
        metrics = {}
        for tag in ea.Tags()['scalars']:
            scalar_events = ea.Scalars(tag)
            metrics[tag] = {
                'steps': [e.step for e in scalar_events],
                'values': [e.value for e in scalar_events],
                'wall_times': [e.wall_time for e in scalar_events],
            }
        
        return metrics
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")
        return {}


def create_training_dashboard(metrics_file: str, output_path: str = "training_dashboard.html"):
    """Create an HTML dashboard for training visualization.
    
    Args:
        metrics_file: Path to metrics JSON file
        output_path: Output HTML file path
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        PLOTLY_AVAILABLE = True
    except ImportError:
        PLOTLY_AVAILABLE = False
        print("plotly not available. Install with: pip install plotly")
        return
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)
    
    episodes = [m['episode'] for m in metrics_data]
    rewards = [m['reward'] for m in metrics_data]
    scores = [m['score'] for m in metrics_data]
    lines = [m['lines_cleared'] for m in metrics_data]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Reward', 'Score', 'Lines Cleared', 'Episode Length'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Reward
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, mode='lines+markers', 
                  name='Reward', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    
    # Score
    fig.add_trace(
        go.Scatter(x=episodes, y=scores, mode='lines+markers', 
                  name='Score', line=dict(color='green', width=1)),
        row=1, col=2
    )
    
    # Lines
    fig.add_trace(
        go.Scatter(x=episodes, y=lines, mode='lines+markers', 
                  name='Lines', line=dict(color='red', width=1)),
        row=2, col=1
    )
    
    # Length
    lengths = [m['length'] for m in metrics_data]
    fig.add_trace(
        go.Scatter(x=episodes, y=lengths, mode='lines+markers', 
                  name='Length', line=dict(color='orange', width=1)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text="RL Training Dashboard",
        showlegend=False
    )
    
    # Save HTML
    fig.write_html(output_path)
    print(f"Dashboard saved to {output_path}")
    print(f"Open in browser: file://{Path(output_path).absolute()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Training Visualization")
    parser.add_argument("command", choices=["plot", "stats", "dashboard", "compare"],
                       help="Command to run")
    parser.add_argument("--metrics-file", type=str, default="./rl_logs/metrics.json",
                       help="Path to metrics JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--window", type=int, default=100,
                       help="Moving average window size")
    parser.add_argument("--recent", type=int, default=100,
                       help="Number of recent episodes for statistics")
    
    args = parser.parse_args()
    
    if args.command == "plot":
        viz = TrainingVisualizer()
        viz.plot_training_progress(window_size=args.window, save_path=args.output)
    elif args.command == "stats":
        viz = TrainingVisualizer()
        viz.print_statistics(recent_n=args.recent)
    elif args.command == "dashboard":
        create_training_dashboard(args.metrics_file, args.output or "dashboard.html")
    elif args.command == "compare":
        # Example comparison - would need baseline metrics
        print("Comparison feature - provide baseline metrics")
