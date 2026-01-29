# Legacy Source Code

⚠️ **DEPRECATED**: This directory contains the original source code from 2020.

The codebase has been refactored into a modern modular architecture. See the main `README.md` for information about the new structure.

## Old Scripts

These scripts are kept for reference but are **deprecated**:

- `tetris_sim.py` - Original simulation with GUI
- `tetris_ai.py` - Original AI implementation
- `tetris_ai_fast.py` - Fast AI variant
- `run_game.py` - Headless game runner
- `run_game_fast_ai.py` - Fast AI runner
- `run_game_random.py` - Random AI runner
- `test_performance.py` - Performance benchmarks
- `profile_game.py` - Profiling script

## Migration Guide

### Old Way
```python
from source.tetris_ai import TetrisAI
from source.tetrimino import Tetrimino

ai = TetrisAI(playfield, active_tet, next_tet, 4, 1)
moves = ai.get_best_moves()
```

### New Way
```python
from tetris.core.game_engine import GameEngine
from tetris.ai.greedy import GreedyAIPlayer
from tetris.simulation.simulator import Simulator

engine = GameEngine()
ai = GreedyAIPlayer(mps=4, level=1)
# Use Simulator to run games
```

See main `README.md` for full documentation.
