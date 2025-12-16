#!/usr/bin/env python3
"""
Test script for edge case where losing player makes the losing move themselves.
In this case, they should get immediate punishment and game should end.
"""

import sys
sys.path.append('.')
from backend_game_chess import ChessGame

def test_self_loss():
    """Test when losing player loses their own queen."""
    print("Testing Self-Loss Case")
    print("=" * 30)
    
    # Create a simple game
    game = ChessGame(["queen"], ["queen"])
    
    # Set up board for direct queen vs queen capture
    game.reset()
    game.board = [[None for _ in range(8)] for _ in range(8)]
    game.board[3][3] = {"color": "white", "type": "queen"}  # White queen at center
    game.board[5][5] = {"color": "black", "type": "queen"}  # Black queen diagonal
    
    # Set current player to white
    game.current_player = "white"
    
    print("Board: White queen at (3,3), Black queen at (5,5)")
    print("Current player:", game.current_player)
    
    # White captures black queen directly
    print("\nWhite captures black queen")
    capture_move = (3, 3, 5, 5)  # White queen captures black queen diagonally
    capture_action = game.move_to_action(capture_move)
    print(f"Attempting capture: {capture_move}, action_idx: {capture_action}")
    
    state2, reward2, terminal2 = game.step(capture_action)
    
    print(f"White reward: {reward2}")
    print(f"Terminal: {terminal2}")
    print(f"Terminal final pass: {game.terminal_final_pass}")
    print(f"Current player: {game.current_player}")
    print(f"Winner: {game.winner}")
    
    # White should get win bonus, set terminal_final_pass, game should continue for black's punishment
    assert reward2 > 10, f"Expected reward > 10 for white, got {reward2}"
    assert game.terminal_final_pass == True, "terminal_final_pass should be True"
    assert terminal2 == False, "Game should not be terminal yet"
    assert game.current_player == "black", "Should be black's turn for punishment"
    
    print("\n✓ White won and set final pass flag")
    
    # Black gets final pass
    print("\nBlack gets final punishment pass")
    dummy_action = 0
    state3, reward3, terminal3 = game.step(dummy_action)
    
    print(f"Black punishment reward: {reward3}")
    print(f"Terminal: {terminal3}")
    
    assert reward3 == -10, f"Expected -10 punishment for black, got {reward3}"
    assert terminal3 == True, "Game should be terminal now"
    
    print("\n✓ Black received punishment and game ended")
    print("\n🎉 Self-loss case working correctly!")

if __name__ == "__main__":
    test_self_loss()