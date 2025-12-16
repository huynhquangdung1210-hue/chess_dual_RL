#!/usr/bin/env python3
"""
Test script for the stalemate case where a player has no valid moves.
"""

import sys
sys.path.append('.')
from backend_game_chess import ChessGame

def test_stalemate():
    """Test when a player has no valid moves (stalemate)."""
    print("Testing Stalemate Case")
    print("=" * 25)
    
    # Create a simple game
    game = ChessGame(["queen"], ["queen"])
    
    # Manually create a scenario where one player has no valid moves
    game.reset()
    game.board = [[None for _ in range(8)] for _ in range(8)]
    # Only place white queen, no black pieces
    game.board[3][3] = {"color": "white", "type": "queen"}  # White queen
    
    # Set black as current player (they have no pieces = no valid moves)
    game.current_player = "black"
    
    print("Board: White queen at (3,3), no black pieces")
    print("Current player:", game.current_player)
    print("Valid moves for black:", len(game.get_valid_moves("black")))
    
    # Black tries to move but has no valid moves
    print("\nBlack attempts action but has no valid moves")
    action_idx = 4096  # "No valid move" action
    
    state, reward, terminal = game.step(action_idx)
    
    print(f"Black reward: {reward}")
    print(f"Terminal: {terminal}")
    print(f"Winner: {game.winner}")
    
    # Check that black got stalemate penalty and game ended
    assert reward == -10.0, f"Expected -10.0 stalemate penalty, got {reward}"
    assert terminal == True, "Game should be terminal"
    assert "stalemate" in game.winner, f"Expected stalemate winner, got {game.winner}"
    
    print("\n✓ Stalemate case working correctly!")
    print(f"✓ Player with no moves gets penalty and game ends")

if __name__ == "__main__":
    test_stalemate()