"""Launcher script for automated chess training.

This script reads the piece configuration from RL_PPO_chess_dual.py
and automatically starts the chess backend with the correct piece types.

Usage:
  1. Configure piece types in RL_PPO_chess_dual.py (WHITE_PIECE_TYPE, BLACK_PIECE_TYPE)
  2. Run: python src/RL_PPO_chess_dual.py (in one terminal)
  3. Run: python src/launch_chess_training.py (in another terminal)
"""

import subprocess
import sys
import re
from pathlib import Path

def extract_piece_config():
    """Extract piece configuration from RL_PPO_chess_dual.py"""
    dual_trainer_path = Path("RL_PPO_chess_dual.py")
    
    if not dual_trainer_path.exists():
        print("Error: RL_PPO_chess_dual.py not found!")
        return None, None

    white_pieces = ["pawn", "king"]  # defaults with king
    black_pieces = ["pawn", "king"]
    
    with open(dual_trainer_path, 'r') as f:
        content = f.read()
        
        # Extract WHITE_PIECE_TYPES (support both single and list format)
        white_match = re.search(r'WHITE_PIECE_TYPE\w*\s*=\s*([^\n]+)', content)
        if white_match:
            white_config = white_match.group(1).strip()
            if white_config.startswith('['):
                # List format: ["pawn", "rook", "queen"]
                pieces = re.findall(r'["\']([^"\']*)["\']*', white_config)
                if pieces:
                    white_pieces = pieces
            else:
                # Single string format: "pawn"
                single_match = re.search(r'["\']([^"\']*)["\']*', white_config)
                if single_match:
                    white_pieces = [single_match.group(1), "queen"]  # Always add queen
        
        # Extract BLACK_PIECE_TYPES (support both single and list format) 
        black_match = re.search(r'BLACK_PIECE_TYPE\w*\s*=\s*([^\n]+)', content)
        if black_match:
            black_config = black_match.group(1).strip()
            if black_config.startswith('['):
                # List format: ["pawn", "rook", "queen"]
                pieces = re.findall(r'["\']([^"\']*)["\']*', black_config)
                if pieces:
                    black_pieces = pieces
            else:
                # Single string format: "pawn"
                single_match = re.search(r'["\']([^"\']*)["\']*', black_config)
                if single_match:
                    black_pieces = [single_match.group(1), "king"]  # Always add king
    
    # Ensure king is always included
    if "king" not in white_pieces:
        white_pieces.append("king")
    if "king" not in black_pieces:
        black_pieces.append("king")
    
    return white_pieces, black_pieces

def main():
    """Launch chess training with configured piece types."""
    print("Chess Training Launcher")
    print("=" * 40)
    
    # Extract piece configuration
    white_pieces, black_pieces = extract_piece_config()
    
    if not white_pieces or not black_pieces:
        print("Could not extract piece configuration!")
        return
    
    print(f"Launching chess training:")
    print(f"  White: {', '.join(white_pieces)}")
    print(f"  Black: {', '.join(black_pieces)}")
    print("  Win condition: Capture opponent's king")
    print("  Make sure RL_PPO_chess_dual.py is running!")
    print()
    
    # Get episode/game number from user
    try:
        game_input = input("Enter starting game number (press Enter for default 1): ").strip()
        game_num = int(game_input) if game_input else 1
    except ValueError:
        print("Invalid game number, using default (1)")
        game_num = 1
    
    print(f"Starting from game: {game_num}")
    print()
    
    # Launch backend with piece types (comma-separated) and game number as command line arguments
    try:
        white_pieces_str = ','.join(white_pieces)
        black_pieces_str = ','.join(black_pieces)
        cmd = [sys.executable, "backend_game_chess_2.py", white_pieces_str, black_pieces_str, str(game_num)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()