#!/usr/bin/env python3
"""
Test script for the double tag system in the chess game.
This tests that when a player wins by capturing the queen:
1. The winning player gets +10 reward and terminal_final_pass is set to True
2. Game switches to losing player who gets -10 punishment 
3. Game then terminates with terminal=True
"""

import sys
sys.path.append('.')
from backend_game_chess import ChessGame

def test_double_tag_system():
    """Test the double tag system."""
    print("Testing Double Tag System")
    print("=" * 40)
    
    # Create a simple game with just queens
    game = ChessGame(["queen"], ["queen"])
    
    # Manually set up a board where white can capture black queen
    game.reset()
    
    # Clear board and place pieces for testing
    game.board = [[None for _ in range(8)] for _ in range(8)]
    game.board[0][3] = {"color": "white", "type": "queen"}  # White queen
    game.board[1][3] = {"color": "black", "type": "queen"}  # Black queen adjacent
    
    print("Initial board setup:")
    print("White queen at (0,3), Black queen at (1,3)")
    print("Current player:", game.current_player)
    
    # White captures black queen
    print("\nStep 1: White captures black queen")
    move = (0, 3, 1, 3)  # White queen captures black queen
    action_idx = game.move_to_action(move)
    
    state1, reward1, terminal1 = game.step(action_idx)
    
    print(f"Reward for white: {reward1}")
    print(f"Terminal: {terminal1}")
    print(f"Terminal final pass: {game.terminal_final_pass}")
    print(f"Current player after move: {game.current_player}")
    print(f"Winner: {game.winner}")
    
    # Check that white got win bonus and terminal_final_pass is True
    assert reward1 > 10, f"Expected reward > 10, got {reward1}"
    assert game.terminal_final_pass == True, "terminal_final_pass should be True"
    assert terminal1 == False, "Game should not be terminal yet"
    assert game.current_player == "black", "Current player should be black"
    assert game.winner == "white", "White should be the winner"
    
    print("\n✓ Step 1 passed: Winner got reward, terminal_final_pass set, game not terminal")
    
    # Now the losing player (black) gets their turn
    print("\nStep 2: Black gets final pass (punishment)")
    
    # Any action from black should trigger the final punishment
    dummy_action = 0  # Any action will work since we handle terminal_final_pass at start of step()
    state2, reward2, terminal2 = game.step(dummy_action)
    
    print(f"Reward for black: {reward2}")
    print(f"Terminal: {terminal2}")
    print(f"Terminal final pass: {game.terminal_final_pass}")
    
    # Check that black got punishment and game is now terminal
    assert reward2 == -10, f"Expected reward -10, got {reward2}"
    assert terminal2 == True, "Game should be terminal now"
    
    print("\n✓ Step 2 passed: Losing player got punishment, game is terminal")
    print("\n🎉 Double tag system working correctly!")

if __name__ == "__main__":
    test_double_tag_system()