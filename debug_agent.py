#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug script to see what's happening with the agent.
"""

import numpy as np
import torch
import backgammon
import agent_td_lambda_Andri as agent

def test_feature_extraction():
    """Test if feature extraction works correctly"""
    print("Testing feature extraction...")
    
    board = backgammon.init_board()
    
    # Test with player +1
    features = agent.one_hot_encoding(board, nSecondRoll=False)
    print(f"  Features for player +1: shape={features.shape}, sum={np.sum(features)}")
    
    # Test with flipped board
    import flipped_agent
    flipped_board = flipped_agent.flip_board(board)
    features2 = agent.one_hot_encoding(flipped_board, nSecondRoll=False)
    print(f"  Features for flipped: shape={features2.shape}, sum={np.sum(features2)}")
    
    # Check if features are different
    if np.array_equal(features, features2):
        print("  WARNING: Features are identical after flipping!")
    else:
        print("  ✓ Features change with perspective")
    
    return True

def test_value_prediction():
    """Test if the critic network produces values"""
    print("\nTesting value prediction...")
    
    agent.set_eval_mode(True)  # Ensure eval mode
    agent.episode_start()  # Reset traces
    
    board = backgammon.init_board()
    features = agent.one_hot_encoding(board, nSecondRoll=False)
    
    # Convert to tensor and get value
    x = torch.tensor(features, dtype=torch.float).view(-1, 1)  # (nx, 1)
    
    # Manually compute forward pass (copy from agent)
    nx = 24 * 2 * 6 + 4 + 1
    H = nx // 2
    
    # Use the agent's parameters
    h = torch.mm(agent.w1, x) + agent.b1
    h_tanh = torch.tanh(h)
    y = torch.mm(agent.w2, h_tanh) + agent.b2
    value = torch.sigmoid(y).item()
    
    print(f"  Initial board value: {value:.4f}")
    
    # Test random vs trained weights
    print(f"  w1 norm: {torch.norm(agent.w1.data):.4f}")
    print(f"  w2 norm: {torch.norm(agent.w2.data):.4f}")
    
    return True

def test_training_update():
    """Test if training updates actually change weights"""
    print("\nTesting training updates...")
    
    agent.set_eval_mode(False)
    agent.episode_start()
    
    # Save initial weights
    w1_before = agent.w1.data.clone()
    w2_before = agent.w2.data.clone()
    
    # Play a few moves to trigger updates
    board = backgammon.init_board()
    player = 1
    
    for turn in range(5):
        dice = backgammon.roll_dice()
        for r in range(1 + int(dice[0] == dice[1])):
            move = agent.action(board.copy(), dice, player, i=r, train=True)
            if len(move) > 0:
                board = backgammon.update_board(board, move, player)
        player = -player
    
    # Check if weights changed
    w1_changed = not torch.allclose(agent.w1.data, w1_before, rtol=1e-5)
    w2_changed = not torch.allclose(agent.w2.data, w2_before, rtol=1e-5)
    
    print(f"  w1 changed: {w1_changed}")
    print(f"  w2 changed: {w2_changed}")
    
    if not (w1_changed or w2_changed):
        print("  CRITICAL: Weights are NOT updating during training!")
    
    return w1_changed or w2_changed

def test_epsilon_decay():
    """Test if epsilon decay is working"""
    print("\nTesting epsilon decay...")
    
    print(f"  Initial epsilon: {agent.current_epsilon:.4f}")
    
    # Simulate some games
    for game in range(5):
        agent.end_episode(1, None, 1)  # This should trigger epsilon decay
    
    print(f"  Epsilon after 5 games: {agent.current_epsilon:.4f}")
    
    if agent.current_epsilon < 0.3:
        print("  ✓ Epsilon is decaying")
    else:
        print("  ✗ Epsilon NOT decaying")
    
    return agent.current_epsilon < 0.3

def test_move_generation():
    """Test if legal moves are being generated correctly"""
    print("\nTesting move generation...")
    
    board = backgammon.init_board()
    dice = (1, 2)
    
    # Test for player +1
    possible_moves, possible_boards = backgammon.legal_moves(board, dice, 1)
    print(f"  Legal moves for +1: {len(possible_moves)}")
    
    # Test for player -1
    possible_moves2, possible_boards2 = backgammon.legal_moves(board, dice, -1)
    print(f"  Legal moves for -1: {len(possible_moves2)}")
    
    # Test agent's action selection
    move = agent.action(board.copy(), dice, 1, 0, train=False)
    print(f"  Agent's move length: {len(move) if move is not None else 0}")
    
    return len(possible_moves) > 0

if __name__ == "__main__":
    print("=" * 60)
    print("DEBUGGING AGENT_TD_LAMBDA_ANDRI.PY")
    print("=" * 60)
    
    results = []
    results.append(test_feature_extraction())
    results.append(test_value_prediction())
    results.append(test_training_update())
    results.append(test_epsilon_decay())
    results.append(test_move_generation())
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    
    # Additional debugging
    print("\nAdditional checks:")
    print(f"  Device: {agent.device}")
    print(f"  nx (feature size): {agent.nx}")
    print(f"  Move number counter: {agent.moveNumber}")