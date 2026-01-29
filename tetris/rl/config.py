"""Configuration loading and management for RL training.

Supports loading agent configs (hyperparameters) and scenario configs
(environment/reward settings) from YAML files, with ability to mix and match.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


# Default config paths
DEFAULT_CONFIG_DIR = Path(__file__).parent / "configs"
DEFAULT_AGENT_CONFIG = DEFAULT_CONFIG_DIR / "agents" / "default.yaml"
DEFAULT_SCENARIO_CONFIG = DEFAULT_CONFIG_DIR / "scenarios" / "default.yaml"


@dataclass
class AgentConfig:
    """Agent/hyperparameter configuration."""
    
    algorithm: str = "PPO"
    policy: str = "MlpPolicy"
    learning_rate: float = 3.0e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "net_arch": [64, 64],
        "activation_fn": "tanh"
    })
    verbose: int = 1
    tensorboard_log: str = "./rl_logs/tensorboard"
    device: str = "auto"
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "AgentConfig":
        """Load agent config from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            AgentConfig instance
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle policy_kwargs specially
        policy_kwargs = data.pop('policy_kwargs', {})
        if 'activation_fn' in policy_kwargs:
            # Convert string to actual function if needed
            if policy_kwargs['activation_fn'] == "tanh":
                import torch.nn as nn
                policy_kwargs['activation_fn'] = nn.Tanh
            elif policy_kwargs['activation_fn'] == "relu":
                import torch.nn as nn
                policy_kwargs['activation_fn'] = nn.ReLU
        
        return cls(**data, policy_kwargs=policy_kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for passing to model constructor."""
        return {
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": self.policy_kwargs,
            "verbose": self.verbose,
            "tensorboard_log": self.tensorboard_log,
            "device": self.device,
        }


@dataclass
class ScenarioConfig:
    """Scenario/environment configuration."""
    
    # Reward function parameters
    rewards: Dict[str, Any] = field(default_factory=lambda: {
        "line_cleared": {
            "single": 100.0,
            "double": 300.0,
            "triple": 500.0,
            "tetris": 800.0,
        },
        "score_multiplier": 0.01,
        "level_up": 50.0,
        "game_over": -100.0,
        "height_threshold": 15,
        "height_penalty": 2.0,
        "hole_penalty": 5.0,
        "variance_penalty": 0.5,
        "survival_bonus": 0.0,
    })
    
    # Environment settings
    environment: Dict[str, Any] = field(default_factory=lambda: {
        "max_placements": 50,
        "observation_mode": "features",  # "features" or "board_channels"
        "use_simple_reward": False,
        "use_minimal_reward": False,
        "next_queue_size": 5,
        "include_ghost": True,
        "feature_normalization": {
            "score_divisor": 10000.0,
        },
        "render_mode": None,
        "render_fps": 10,
    })
    
    # Training scenario settings
    training: Dict[str, Any] = field(default_factory=lambda: {
        "max_steps_per_episode": 10000,
        "eval_episodes": 10,
        "eval_freq": 10000,
    })
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "ScenarioConfig":
        """Load scenario config from YAML file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            ScenarioConfig instance
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    def get_reward_params(self) -> Dict[str, Any]:
        """Get reward function parameters."""
        return self.rewards
    
    def get_env_params(self) -> Dict[str, Any]:
        """Get environment parameters."""
        return self.environment
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self.training


@dataclass
class RLConfig:
    """Complete RL configuration combining agent and scenario."""
    
    agent: AgentConfig
    scenario: ScenarioConfig
    
    # Training settings
    total_timesteps: int = 1_000_000
    checkpoint_freq: int = 50_000
    render_every: int = 0  # 0 = never render during training
    model_name: str = "tetris_rl"
    log_dir: str = "./rl_logs"
    
    @classmethod
    def from_files(
        cls,
        agent_config_path: Optional[Path] = None,
        scenario_config_path: Optional[Path] = None,
        **kwargs
    ) -> "RLConfig":
        """Load complete config from agent and scenario files.
        
        Args:
            agent_config_path: Path to agent config YAML (defaults to default.yaml)
            scenario_config_path: Path to scenario config YAML (defaults to default.yaml)
            **kwargs: Additional training settings to override
            
        Returns:
            RLConfig instance
        """
        if agent_config_path is None:
            agent_config_path = DEFAULT_AGENT_CONFIG
        if scenario_config_path is None:
            scenario_config_path = DEFAULT_SCENARIO_CONFIG
        
        agent = AgentConfig.from_yaml(agent_config_path)
        scenario = ScenarioConfig.from_yaml(scenario_config_path)
        
        return cls(agent=agent, scenario=scenario, **kwargs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RLConfig":
        """Create config from dictionary (for programmatic use).
        
        Args:
            config_dict: Dictionary with 'agent' and 'scenario' keys
            
        Returns:
            RLConfig instance
        """
        agent = AgentConfig(**config_dict.get('agent', {}))
        scenario = ScenarioConfig(**config_dict.get('scenario', {}))
        
        training_kwargs = {
            k: v for k, v in config_dict.items()
            if k not in ('agent', 'scenario')
        }
        
        return cls(agent=agent, scenario=scenario, **training_kwargs)


def load_config(
    agent_config: Optional[str] = None,
    scenario_config: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> RLConfig:
    """Convenience function to load RL config.
    
    Args:
        agent_config: Name of agent config file (e.g., "default", "large_network")
                     or full path. Looks in configs/agents/ if relative.
        scenario_config: Name of scenario config file (e.g., "default", "aggressive_rewards")
                        or full path. Looks in configs/scenarios/ if relative.
        config_dir: Base directory for configs (defaults to tetris/rl/configs)
        
    Returns:
        RLConfig instance
        
    Examples:
        # Use defaults
        config = load_config()
        
        # Mix and match
        config = load_config(
            agent_config="large_network",
            scenario_config="aggressive_rewards"
        )
        
        # Use absolute paths
        config = load_config(
            agent_config="/path/to/custom_agent.yaml",
            scenario_config="/path/to/custom_scenario.yaml"
        )
    """
    if config_dir is None:
        config_dir = DEFAULT_CONFIG_DIR
    
    # Resolve agent config path
    if agent_config is None:
        agent_path = DEFAULT_AGENT_CONFIG
    elif Path(agent_config).is_absolute():
        agent_path = Path(agent_config)
    else:
        # Assume it's a name in agents/ directory
        agent_path = config_dir / "agents" / f"{agent_config}.yaml"
        if not agent_path.exists():
            # Try without .yaml extension
            agent_path = config_dir / "agents" / agent_config
    
    # Resolve scenario config path
    if scenario_config is None:
        scenario_path = DEFAULT_SCENARIO_CONFIG
    elif Path(scenario_config).is_absolute():
        scenario_path = Path(scenario_config)
    else:
        # Assume it's a name in scenarios/ directory
        scenario_path = config_dir / "scenarios" / f"{scenario_config}.yaml"
        if not scenario_path.exists():
            # Try without .yaml extension
            scenario_path = config_dir / "scenarios" / scenario_config
    
    return RLConfig.from_files(agent_path, scenario_path)
