#!/usr/bin/env python3
"""
Debug test to see what's happening with the double tag system.
"""

import sys
sys.path.append('.')
from backend_game_chess import ChessGame

def debug_double_tag():
    """Debug the double tag system."""
    print("Debug Double Tag System")
    print("=" * 30)
    
    game = ChessGame(["queen"], ["queen"])
    game.reset()
    
    # Simple setup
    game.board = [[None for _ in range(8)] for _ in range(8)]
    game.board[0][0] = {"color": "white", "type": "queen"}
    game.board[1][1] = {"color": "black", "type": "queen"}
    
    print("Initial state:")
    print(f"Current player: {game.current_player}")
    print(f"Terminal: {game.terminal}")
    print(f"Terminal final pass: {game.terminal_final_pass}")
    
    # White captures black queen
    capture_move = (0, 0, 1, 1)
    action_idx = game.move_to_action(capture_move)
    
    print(f"\nWhite captures black queen with action {action_idx}")
    state1, reward1, terminal1 = game.step(action_idx)
    
    print(f"After white's move:")
    print(f"  Reward: {reward1}")
    print(f"  Terminal: {terminal1}")
    print(f"  Terminal final pass: {game.terminal_final_pass}")
    print(f"  Current player: {game.current_player}")
    print(f"  Winner: {game.winner}")
    
    # Black gets final punishment
    print(f"\nBlack's final pass:")
    print(f"  Terminal before step: {game.terminal}")
    print(f"  Terminal final pass before step: {game.terminal_final_pass}")
    
    state2, reward2, terminal2 = game.step(0)  # Any action triggers punishment
    
    print(f"After black's step:")
    print(f"  Reward: {reward2}")
    print(f"  Terminal: {terminal2}")
    print(f"  Terminal final pass: {game.terminal_final_pass}")

if __name__ == "__main__":
    debug_double_tag()