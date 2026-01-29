"""Test script to verify RL environment works correctly.

Run this before training to ensure everything is set up properly:
    python -m tetris.rl.test_env
"""

import sys

def test_imports():
    """Test that all required dependencies are available."""
    print("Testing imports...")
    
    try:
        import gymnasium as gym
        print("✓ gymnasium")
    except ImportError:
        print("✗ gymnasium not found. Install with: uv pip install gymnasium")
        return False
    
    try:
        from stable_baselines3 import PPO
        print("✓ stable-baselines3")
    except ImportError:
        print("✗ stable-baselines3 not found. Install with: uv pip install stable-baselines3")
        return False
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError:
        print("✗ numpy not found. Install with: uv pip install numpy")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib")
    except ImportError:
        print("⚠ matplotlib not found (optional, for visualization)")
    
    return True


def test_environment():
    """Test that TetrisEnv works correctly."""
    print("\nTesting TetrisEnv...")
    
    try:
        from tetris.rl.env import TetrisEnv, state_to_features
        print("✓ TetrisEnv imported")
    except ImportError as e:
        print(f"✗ Failed to import TetrisEnv: {e}")
        return False
    
    try:
        env = TetrisEnv()
        print("✓ Environment created")
    except Exception as e:
        print(f"✗ Failed to create environment: {e}")
        return False
    
    try:
        obs, info = env.reset()
        print(f"✓ Environment reset - observation shape: {obs.shape}")
        assert obs.shape == (23,), f"Expected shape (23,), got {obs.shape}"
    except Exception as e:
        print(f"✗ Failed to reset environment: {e}")
        return False
    
    try:
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Step {i+1}: action={action}, reward={reward:.2f}, done={done}")
            if done:
                obs, info = env.reset()
    except Exception as e:
        print(f"✗ Failed to step environment: {e}")
        return False
    
    try:
        env.close()
        print("✓ Environment closed")
    except Exception as e:
        print(f"⚠ Failed to close environment: {e}")
    
    return True


def test_feature_extraction():
    """Test feature extraction."""
    print("\nTesting feature extraction...")
    
    try:
        from tetris.core.game_engine import GameEngine
        from tetris.rl.env import state_to_features
        
        engine = GameEngine()
        features = state_to_features(engine.state)
        
        assert features.shape == (23,), f"Expected 23 features, got {features.shape[0]}"
        assert not np.isnan(features).any(), "Features contain NaN"
        assert not np.isinf(features).any(), "Features contain Inf"
        
        print(f"✓ Feature extraction works - shape: {features.shape}")
        print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
        return True
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_calculation():
    """Test reward calculation."""
    print("\nTesting reward calculation...")
    
    try:
        from tetris.core.game_engine import GameEngine
        from tetris.rl.env import calculate_reward
        
        engine = GameEngine()
        prev_state = engine.state
        
        # Make a move
        engine.state = engine.apply_move("right")
        
        reward = calculate_reward(prev_state, engine.state, False)
        
        assert isinstance(reward, (int, float)), "Reward should be numeric"
        assert not np.isnan(reward), "Reward should not be NaN"
        
        print(f"✓ Reward calculation works - reward: {reward:.2f}")
        return True
    except Exception as e:
        print(f"✗ Reward calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_setup():
    """Test that training setup works."""
    print("\nTesting training setup...")
    
    try:
        from tetris.rl.env import TetrisEnv
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        env = TetrisEnv()
        env = DummyVecEnv([lambda: env])
        
        model = PPO("MlpPolicy", env, verbose=0)
        print("✓ PPO model created")
        
        # Test a few training steps
        model.learn(total_timesteps=100, progress_bar=False)
        print("✓ Training step completed")
        
        return True
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("RL Environment Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Feature Extraction", test_feature_extraction),
        ("Reward Calculation", test_reward_calculation),
        ("Training Setup", test_training_setup),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready for training.")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
