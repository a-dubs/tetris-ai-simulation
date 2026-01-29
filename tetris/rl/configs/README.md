# RL Configuration System

The RL module uses a flexible configuration system that separates **agent settings** (hyperparameters, model architecture) from **scenario settings** (reward functions, environment parameters). This allows you to mix and match different configurations.

## Directory Structure

```
tetris/rl/configs/
├── agents/          # Agent/hyperparameter configs
│   ├── default.yaml
│   ├── large_network.yaml
│   └── fast_training.yaml
└── scenarios/       # Scenario/reward configs
    ├── default.yaml
    ├── aggressive_rewards.yaml
    ├── conservative_rewards.yaml
    └── survival_focused.yaml
```

## Usage

### Basic Usage

Use default configs:
```bash
python -m tetris.rl.training train --timesteps 100000
```

### Mix and Match

Combine different agent and scenario configs:
```bash
# Large network with aggressive rewards
python -m tetris.rl.training train \
    --agent-config large_network \
    --scenario-config aggressive_rewards \
    --timesteps 1000000

# Fast training with survival-focused rewards
python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config survival_focused \
    --timesteps 500000
```

### Override Specific Parameters

Override config values with command-line arguments:
```bash
python -m tetris.rl.training train \
    --agent-config default \
    --scenario-config default \
    --lr 5e-4 \
    --batch-size 128 \
    --timesteps 1000000
```

### Using Custom Config Files

Use absolute paths for custom configs:
```bash
python -m tetris.rl.training train \
    --agent-config /path/to/my_agent.yaml \
    --scenario-config /path/to/my_scenario.yaml
```

## Agent Configs

Agent configs define hyperparameters and model architecture:

### Available Agent Configs

- **`default.yaml`**: Standard PPO configuration (64x64 network)
- **`large_network.yaml`**: Larger network (128x128x64) for complex learning
- **`fast_training.yaml`**: Smaller network, higher learning rate for quick iteration

### Agent Config Structure

```yaml
algorithm: "PPO"
policy: "MlpPolicy"
learning_rate: 3.0e-4
n_steps: 2048
batch_size: 64
n_epochs: 10
gamma: 0.99
policy_kwargs:
  net_arch: [64, 64]
  activation_fn: "tanh"
```

## Scenario Configs

Scenario configs define reward functions and environment settings:

### Available Scenario Configs

- **`default.yaml`**: Balanced rewards (standard Tetris scoring)
- **`aggressive_rewards.yaml`**: Higher rewards, stronger penalties
- **`conservative_rewards.yaml`**: Lower rewards, gentler penalties
- **`survival_focused.yaml`**: Emphasizes staying alive over scoring

### Scenario Config Structure

```yaml
rewards:
  line_cleared:
    single: 100.0
    double: 300.0
    triple: 500.0
    tetris: 800.0
  score_multiplier: 0.01
  level_up: 50.0
  game_over: -100.0
  height_threshold: 15
  height_penalty: 2.0
  hole_penalty: 5.0
  variance_penalty: 0.5
```

## Creating Custom Configs

### Custom Agent Config

Create `tetris/rl/configs/agents/my_agent.yaml`:

```yaml
algorithm: "PPO"
policy: "MlpPolicy"
learning_rate: 1.0e-4
n_steps: 4096
batch_size: 128
n_epochs: 20
gamma: 0.995
policy_kwargs:
  net_arch: [256, 256, 128]
  activation_fn: "tanh"
```

### Custom Scenario Config

Create `tetris/rl/configs/scenarios/my_scenario.yaml`:

```yaml
rewards:
  line_cleared:
    single: 150.0
    double: 400.0
    triple: 700.0
    tetris: 1200.0
  score_multiplier: 0.015
  level_up: 75.0
  game_over: -150.0
  height_threshold: 13
  height_penalty: 3.0
  hole_penalty: 8.0
  variance_penalty: 0.75
```

## Programmatic Usage

Load configs in Python:

```python
from tetris.rl.config import load_config

# Load default configs
config = load_config()

# Mix and match
config = load_config(
    agent_config="large_network",
    scenario_config="aggressive_rewards"
)

# Use config
from tetris.rl.training import train_agent
train_agent(config=config, total_timesteps=1000000)
```

## Best Practices

1. **Start with defaults**: Use `default` configs initially
2. **Experiment systematically**: Change one config at a time
3. **Document custom configs**: Add comments explaining your choices
4. **Version control**: Commit configs to track experiments
5. **Name descriptively**: Use clear names like `high_exploration.yaml`

## Examples

### Example 1: Quick Experimentation

```bash
# Fast iteration with aggressive rewards
python -m tetris.rl.training train \
    --agent-config fast_training \
    --scenario-config aggressive_rewards \
    --timesteps 50000
```

### Example 2: Production Training

```bash
# Large network with balanced rewards
python -m tetris.rl.training train \
    --agent-config large_network \
    --scenario-config default \
    --timesteps 5000000
```

### Example 3: Survival Challenge

```bash
# Focus on staying alive
python -m tetris.rl.training train \
    --agent-config default \
    --scenario-config survival_focused \
    --timesteps 1000000
```

## Config Validation

Configs are validated when loaded. Common issues:

- **Missing required fields**: Will use defaults
- **Invalid values**: Will raise errors
- **Type mismatches**: Will raise errors

Check configs before training:
```python
from tetris.rl.config import load_config
config = load_config("my_agent", "my_scenario")
print(config.agent.learning_rate)
print(config.scenario.rewards)
```
