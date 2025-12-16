"""Chess-based backend game for RL training.

A simplified chess environment where:
- Two sides play with chosen piece types (e.g., pawns, rooks, etc.)
- Standard chess movement rules apply
- Pawn promotion: pawns reaching the opposite end move like queens
- Win condition: capture opponent's queen
- Game ends on win, stalemate (no valid moves), or max moves reached

Usage:
  python backend_game_chess_2.py [white_pieces] [black_pieces] [starting_game_num]
  Example: python backend_game_chess_2.py pawn,queen pawn,queen 1

Requirements:
  pip install websockets numpy
"""

import asyncio
import json
import math
import numpy as np
import websockets

# ================================================
# CONFIGURATION
# ================================================

WS_URI = "ws://localhost:8765"
MAX_MOVES_PER_SIDE = 50
BOARD_SIZE = 8

# Action encoding: (from_square, to_square) pairs
# 64 * 64 = 4096 possible moves + 1 for "no valid moves"
MAX_ACTIONS = BOARD_SIZE * BOARD_SIZE * BOARD_SIZE * BOARD_SIZE + 1

# Piece configurations
PIECE_CONFIG = {
    "pawn": {"count": 8, "white_row": 1, "black_row": 6},
    "rook": {"white_pos": [(0, 0), (7, 0)], "black_pos": [(0, 7), (7, 7)]},
    "knight": {"white_pos": [(1, 0), (6, 0)], "black_pos": [(1, 7), (6, 7)]},
    "bishop": {"white_pos": [(2, 0), (5, 0)], "black_pos": [(2, 7), (5, 7)]},
    "queen": {"white_pos": [(3, 0)], "black_pos": [(3, 7)]},
    "king": {"white_pos": [(4, 0)], "black_pos": [(4, 7)]},
}


# ================================================
# CHESS GAME
# ================================================

class ChessGame:
    """Chess game environment for RL training."""
    
    def __init__(self, white_pieces=None, black_pieces=None):
        self.white_pieces = white_pieces or ["pawn", "king"]
        self.black_pieces = black_pieces or ["pawn", "king"]
        
        # Ensure king is always included (win condition)
        if "king" not in self.white_pieces:
            self.white_pieces.append("king")
        if "king" not in self.black_pieces:
            self.black_pieces.append("king")
        
        self.reset()
    
    def reset(self):
        """Reset game to initial state."""
        self.board = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self._place_pieces("white", self.white_pieces)
        self._place_pieces("black", self.black_pieces)
        
        self.current_player = "white"
        self.move_count = 0
        self.terminal = False
        self.winner = None
        self.awaiting_loser_response = False
    
    def _place_pieces(self, color, piece_types):
        """Place pieces on the board for given color."""
        back_row = 0 if color == "white" else 7
        pawn_row = 1 if color == "white" else 6
        used_positions = set()
        
        for piece_type in piece_types:
            if piece_type not in PIECE_CONFIG:
                continue
            
            config = PIECE_CONFIG[piece_type]
            
            if piece_type == "pawn":
                # Place pawns across their row
                for col in range(BOARD_SIZE):
                    self.board[pawn_row][col] = {
                        "color": color,
                        "type": "pawn",
                        "promoted": False
                    }
            else:
                # Place piece at predefined positions
                pos_key = "white_pos" if color == "white" else "black_pos"
                if pos_key in config:
                    for col, row in config[pos_key]:
                        if (col, row) not in used_positions and self.board[row][col] is None:
                            self.board[row][col] = {"color": color, "type": piece_type}
                            used_positions.add((col, row))
    
    def get_state(self):
        """Get flattened board state for RL.
        
        Returns:
            List of floats: board (64) + metadata (3)
            Piece encoding: 0=empty, 1=white, -1=black, 2=promoted white pawn, -2=promoted black pawn
        """
        state = []
        
        for row in self.board:
            for cell in row:
                if cell is None:
                    state.append(0.0)
                elif cell["color"] == "white":
                    if cell["type"] == "pawn" and cell.get("promoted"):
                        state.append(2.0)
                    else:
                        state.append(1.0)
                else:  # black
                    if cell["type"] == "pawn" and cell.get("promoted"):
                        state.append(-2.0)
                    else:
                        state.append(-1.0)
        
        # Add metadata
        state.append(1.0 if self.current_player == "white" else -1.0)
        state.append(float(self.move_count))
        state.append(float(MAX_MOVES_PER_SIDE * 2 - self.move_count))
        
        return state
    
    def get_valid_moves(self, color=None):
        """Get all valid moves for a player.
        
        Returns:
            List of (from_row, from_col, to_row, to_col) tuples
        """
        color = color or self.current_player
        moves = []
        
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = self.board[row][col]
                if piece and piece["color"] == color:
                    piece_moves = self._get_piece_moves(row, col, piece)
                    moves.extend(piece_moves)
        
        return moves
    
    def _get_piece_moves(self, row, col, piece):
        """Get valid moves for a specific piece."""
        piece_type = piece["type"]
        
        if piece_type == "pawn":
            if piece.get("promoted"):
                destinations = self._get_queen_destinations(row, col, piece["color"])
            else:
                destinations = self._get_pawn_destinations(row, col, piece["color"])
        elif piece_type == "rook":
            destinations = self._get_rook_destinations(row, col, piece["color"])
        elif piece_type == "knight":
            destinations = self._get_knight_destinations(row, col, piece["color"])
        elif piece_type == "bishop":
            destinations = self._get_bishop_destinations(row, col, piece["color"])
        elif piece_type == "queen":
            destinations = self._get_queen_destinations(row, col, piece["color"])
        elif piece_type == "king":
            destinations = self._get_king_destinations(row, col, piece["color"])
        else:
            destinations = []
        
        return [(row, col, to_row, to_col) for to_row, to_col in destinations]
    
    def _get_pawn_destinations(self, row, col, color):
        """Get pawn move destinations."""
        moves = []
        direction = 1 if color == "white" else -1
        start_row = 1 if color == "white" else 6
        
        # Forward move
        new_row = row + direction
        if 0 <= new_row < BOARD_SIZE and self.board[new_row][col] is None:
            moves.append((new_row, col))
            
            # Double move from start
            if row == start_row:
                new_row2 = row + 2 * direction
                if self.board[new_row2][col] is None:
                    moves.append((new_row2, col))
        
        # Diagonal captures
        for dc in [-1, 1]:
            new_row = row + direction
            new_col = col + dc
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                target = self.board[new_row][new_col]
                if target and target["color"] != color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_rook_destinations(self, row, col, color):
        """Get rook move destinations (orthogonal lines)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        return self._get_line_destinations(row, col, color, directions)
    
    def _get_bishop_destinations(self, row, col, color):
        """Get bishop move destinations (diagonal lines)."""
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        return self._get_line_destinations(row, col, color, directions)
    
    def _get_queen_destinations(self, row, col, color):
        """Get queen move destinations (all 8 directions)."""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        return self._get_line_destinations(row, col, color, directions)
    
    def _get_line_destinations(self, row, col, color, directions):
        """Get destinations along lines (for rook, bishop, queen)."""
        moves = []
        
        for dr, dc in directions:
            for dist in range(1, BOARD_SIZE):
                new_row = row + dist * dr
                new_col = col + dist * dc
                
                if not (0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE):
                    break
                
                target = self.board[new_row][new_col]
                if target is None:
                    moves.append((new_row, new_col))
                elif target["color"] != color:
                    moves.append((new_row, new_col))  # Capture
                    break
                else:
                    break  # Blocked by own piece
        
        return moves
    
    def _get_knight_destinations(self, row, col, color):
        """Get knight move destinations (L-shapes)."""
        moves = []
        offsets = [(2, 1), (2, -1), (-2, 1), (-2, -1),
                   (1, 2), (1, -2), (-1, 2), (-1, -2)]
        
        for dr, dc in offsets:
            new_row = row + dr
            new_col = col + dc
            
            if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                target = self.board[new_row][new_col]
                if target is None or target["color"] != color:
                    moves.append((new_row, new_col))
        
        return moves
    
    def _get_king_destinations(self, row, col, color):
        """Get king move destinations (1 square any direction)."""
        moves = []
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                new_row = row + dr
                new_col = col + dc
                
                if 0 <= new_row < BOARD_SIZE and 0 <= new_col < BOARD_SIZE:
                    target = self.board[new_row][new_col]
                    if target is None or target["color"] != color:
                        moves.append((new_row, new_col))
        
        return moves
    
    def get_action_mask(self, color=None):
        """Get action mask for valid moves.
        
        Returns:
            numpy array of shape [MAX_ACTIONS] with 1.0 for valid, 0.0 for invalid
        """
        color = color or self.current_player
        mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        
        valid_moves = self.get_valid_moves(color)
        
        if not valid_moves:
            mask[4096] = 1.0  # Special "no valid moves" action
            return mask
        
        for move in valid_moves:
            action_idx = self.move_to_action(move)
            if 0 <= action_idx < MAX_ACTIONS - 1:
                mask[action_idx] = 1.0
        
        return mask
    
    def move_to_action(self, move):
        """Convert move tuple to action index."""
        from_row, from_col, to_row, to_col = move
        from_square = from_row * BOARD_SIZE + from_col
        to_square = to_row * BOARD_SIZE + to_col
        return from_square * (BOARD_SIZE * BOARD_SIZE) + to_square
    
    def action_to_move(self, action_idx):
        """Convert action index to move tuple."""
        if action_idx == 4096:
            return None  # No valid move action
        if action_idx < 0 or action_idx >= MAX_ACTIONS:
            return None
        
        from_square = action_idx // (BOARD_SIZE * BOARD_SIZE)
        to_square = action_idx % (BOARD_SIZE * BOARD_SIZE)
        
        from_row = from_square // BOARD_SIZE
        from_col = from_square % BOARD_SIZE
        to_row = to_square // BOARD_SIZE
        to_col = to_square % BOARD_SIZE
        
        return (from_row, from_col, to_row, to_col)
    
    def step(self, action_idx):
        """Execute a move and return (new_state, reward, terminal).
        
        Rewards:
            -0.01: per move (time penalty)
            +1.0: capture a piece
            +15.0: capture king
            +10.0: win the game
            -10.0: lose the game or stalemate
            +2.0: pawn promotion
        """
        if self.terminal:
            return self.get_state(), 0.0, True
        
        # Handle loser's final response
        if self.awaiting_loser_response:
            self.terminal = True
            print(f"[-10.0R] {self.current_player} receives losing penalty (final pass)")
            return self.get_state(), -10.0, True
        
        move = self.action_to_move(action_idx)
        
        # No valid moves = stalemate
        if move is None or action_idx == 4096:
            self.terminal = True
            self.winner = f"stalemate_{self.current_player}"
            print(f"[-10.0R] {self.current_player} has no valid moves - stalemate!")
            return self.get_state(), -10.0, True
        
        # Validate move
        valid_moves = self.get_valid_moves()
        if move not in valid_moves:
            from_row, from_col, to_row, to_col = move
            piece_at_from = self.board[from_row][from_col]
            piece_at_to = self.board[to_row][to_col]
            print(f"[-0.05R] {self.current_player} made invalid move {move}")
            print(f"  Piece at ({from_row},{from_col}): {piece_at_from}")
            print(f"  Piece at ({to_row},{to_col}): {piece_at_to}")
            if piece_at_from and piece_at_from['color'] == self.current_player:
                piece_valid = self._get_piece_moves(from_row, from_col, piece_at_from)
                print(f"  Valid moves for this piece: {piece_valid[:5]}{'...' if len(piece_valid) > 5 else ''}")
            print(f"  Total valid moves for {self.current_player}: {len(valid_moves)}")
            return self.get_state(), -0.05, False  # Invalid move penalty
        
        from_row, from_col, to_row, to_col = move
        piece = self.board[from_row][from_col]
        captured = self.board[to_row][to_col]
        
        # Execute move
        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        
        # Calculate reward
        reward = -0.01  # Time penalty
        
        # Pawn promotion
        if piece["type"] == "pawn" and not piece.get("promoted"):
            promotion_row = 7 if piece["color"] == "white" else 0
            if to_row == promotion_row:
                piece["promoted"] = True
                reward += 2.0
                print(f"[+2.0R] {self.current_player} pawn promoted at ({to_row},{to_col})!")
        
        # Capture reward
        if captured:
            reward += 1.0
            print(f"[+1.0R] {self.current_player} captured {captured['color']} {captured['type']}")
            if captured["type"] == "king":
                reward += 15.0
                print(f"[+15.0R] {self.current_player} captured the {captured['color']} king!")
        
        # Check win condition (king captured)
        white_has_king = self._has_king("white")
        black_has_king = self._has_king("black")
        
        if not white_has_king:
            self.winner = "black"
            if self.current_player == "black":
                reward += 10.0
                print(f"[+10.0R] {self.current_player} wins!")
                self.awaiting_loser_response = True
            else:
                reward -= 10.0
                print(f"[-10.0R] {self.current_player} loses!")
                self.terminal = True
        elif not black_has_king:
            self.winner = "white"
            if self.current_player == "white":
                reward += 10.0
                print(f"[+10.0R] {self.current_player} wins!")
                self.awaiting_loser_response = True
            else:
                reward -= 10.0
                print(f"[-10.0R] {self.current_player} loses!")
                self.terminal = True
        
        # Switch player and increment move count
        self.current_player = "black" if self.current_player == "white" else "white"
        self.move_count += 1
        
        # Check draw (max moves)
        if self.move_count >= MAX_MOVES_PER_SIDE * 2:
            self.terminal = True
            self.winner = "draw"
        
        return self.get_state(), reward, self.terminal
    
    def _has_king(self, color):
        """Check if player has their king."""
        for row in self.board:
            for cell in row:
                if cell and cell["color"] == color and cell["type"] == "king":
                    return True
        return False


# ================================================
# GAME HANDLER
# ================================================

class ChessGameHandler:
    """WebSocket handler for chess game with statistics tracking."""
    
    def __init__(self, white_pieces, black_pieces):
        self.white_piece_types = white_pieces
        self.black_piece_types = black_pieces
        self.game = ChessGame(white_pieces, black_pieces)
        self.episode_count = 0
        
        # Game statistics
        self.game_stats = {
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'stalemates': 0,
            'white_win_lengths': [],
            'black_win_lengths': [],
            'draw_lengths': [],
            'stalemate_lengths': [],
            'total_games': 0
        }
        
        # Move tracking: count of each action index used
        self.move_tracking = np.zeros(MAX_ACTIONS, dtype=np.int32)
    
    async def run_episode(self, ws):
        """Run a single game episode."""
        self.episode_count += 1
        self.game.reset()
        
        print(f"\n{'='*50}")
        print(f"Episode {self.episode_count}")
        print(f"White: {self.white_piece_types} vs Black: {self.black_piece_types}")
        print(f"{'='*50}")
        
        white_reward = 0.0
        black_reward = 0.0
        last_action = None
        msg_seq = 0  # Message sequence number for debugging
        
        # Track pending rewards for each player (reward earned from their last action)
        pending_reward = {'white': 0.0, 'black': 0.0}
        
        while not self.game.terminal:
            agent_id = f"chess-{self.game.current_player}"
            current_player = self.game.current_player
            state = self.game.get_state()
            mask = self.game.get_action_mask()
            msg_seq += 1
            
            # Store a hash of the mask for verification
            mask_hash = hash(mask.tobytes())
            valid_count = int(np.sum(mask))
            
            # Send the reward this player earned from their LAST action
            reward_to_send = pending_reward[current_player]
            pending_reward[current_player] = 0.0  # Reset after sending
            
            # Request action from RL server
            msg = {
                "type": "step",
                "preyId": agent_id,
                "state": state,
                "action_mask": mask.tolist(),
                "reward": reward_to_send,  # Reward from this player's previous action
                "prev_state": None,
                "prev_action": None,
                "terminal": False,
                "msg_seq": msg_seq,  # Add sequence number
                "mask_hash": mask_hash,  # Add mask hash for verification
                "valid_count": valid_count
            }
            
            await ws.send(json.dumps(msg))
            response = await ws.recv()
            action_data = json.loads(response)
            action_idx = action_data.get("action_idx", 0)
            resp_seq = action_data.get("msg_seq", -1)
            resp_mask_hash = action_data.get("mask_hash", -1)
            last_action = action_idx
            
            # Verify action was in the mask we sent
            if mask[action_idx] != 1.0:
                print(f"[ERROR] RL returned action {action_idx} but mask[{action_idx}]={mask[action_idx]}")
                print(f"  Msg seq: {msg_seq}, Response seq: {resp_seq}")
                print(f"  Mask hash sent: {mask_hash}, Response mask hash: {resp_mask_hash}")
                print(f"  Valid actions in mask: {valid_count}")
                print(f"  Action decoded: {self.game.action_to_move(action_idx)}")
                # Find what actions were actually valid
                valid_indices = np.where(mask == 1.0)[0]
                print(f"  First 10 valid action indices: {valid_indices[:10]}")
            
            # Track the move
            if 0 <= action_idx < MAX_ACTIONS:
                self.move_tracking[action_idx] += 1
            
            # Execute move - note: current_player was already captured above before step()
            # The player who just acted is in 'current_player' variable
            new_state, reward, terminal = self.game.step(action_idx)
            
            # Store reward for this player - will be sent on their NEXT turn
            # or in game_end message if terminal
            pending_reward[current_player] += reward
            
            # Track total rewards for statistics
            if current_player == "white":
                white_reward += reward
            else:
                black_reward += reward
            
            # Handle double-tag for winner/loser final messages
            if self.game.awaiting_loser_response and not terminal:
                # Winner already got their reward from the step() call
                # Now handle the loser's final pass
                
                # Send loser's state to get their "final" action (which will be ignored)
                loser_id = f"chess-{self.game.current_player}"
                loser_state = self.game.get_state()
                loser_mask = self.game.get_action_mask()
                msg_seq += 1
                loser_mask_hash = hash(loser_mask.tobytes())
                loser_valid_count = int(np.sum(loser_mask))
                
                loser_msg = {
                    "type": "step",
                    "preyId": loser_id,
                    "state": loser_state,
                    "action_mask": loser_mask.tolist(),
                    "reward": 0.0,
                    "prev_state": None,
                    "prev_action": None,
                    "terminal": False,
                    "msg_seq": msg_seq,
                    "mask_hash": loser_mask_hash,
                    "valid_count": loser_valid_count
                }
                await ws.send(json.dumps(loser_msg))
                loser_response = await ws.recv()
                loser_action_data = json.loads(loser_response)
                
                # Execute loser's final step (action doesn't matter, just triggers penalty)
                loser_action = loser_action_data.get("action_idx", 4096)
                loser_player = self.game.current_player  # This is the loser
                final_state, final_reward, _ = self.game.step(loser_action)
                
                # Add pending reward for loser plus final reward
                total_loser_reward = pending_reward[loser_player] + final_reward
                pending_reward[loser_player] = 0.0
                
                if loser_player == "white":
                    white_reward += final_reward
                else:
                    black_reward += final_reward
                
                # Send loser's terminal (no response needed)
                terminal_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                final_msg = {
                    "type": "game_end",
                    "preyId": loser_id,
                    "state": final_state,
                    "action_mask": terminal_mask.tolist(),
                    "reward": total_loser_reward,  # Include all pending rewards
                    "prev_state": loser_state,
                    "prev_action": loser_action,
                    "terminal": True,
                    "winner": self.game.winner,
                    "move_count": self.game.move_count,
                    "white_piece_type": ','.join(self.white_piece_types),
                    "black_piece_type": ','.join(self.black_piece_types),
                    "move_tracking": self.move_tracking.tolist()
                }
                await ws.send(json.dumps(final_msg))
                
            elif not terminal:
                # Normal game continuation - no transition message needed
                # The next iteration will send a new action request
                pass
            else:
                # Terminal without double-tag - include pending reward
                total_reward = pending_reward[current_player] + reward
                pending_reward[current_player] = 0.0
                
                terminal_mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                terminal_msg = {
                    "type": "game_end",
                    "preyId": agent_id,
                    "state": new_state,
                    "action_mask": terminal_mask.tolist(),
                    "reward": total_reward,  # Include all pending rewards
                    "prev_state": None,
                    "prev_action": action_idx,
                    "terminal": True,
                    "winner": self.game.winner,
                    "move_count": self.game.move_count,
                    "white_piece_type": ','.join(self.white_piece_types),
                    "black_piece_type": ','.join(self.black_piece_types),
                    "move_tracking": self.move_tracking.tolist()
                }
                await ws.send(json.dumps(terminal_msg))
        
        # Update statistics
        self._update_stats()
        
        # Don't send final_message separately - game_end was already sent above
        # await self._send_final_message(ws, last_action)
        
        # Print episode summary
        self._print_summary(white_reward, black_reward)
        
        return white_reward, black_reward
    
    def _update_stats(self):
        """Update game statistics after episode."""
        game_length = self.game.move_count
        self.game_stats['total_games'] += 1
        
        winner = self.game.winner
        is_stalemate = winner and winner.startswith('stalemate_')
        
        if is_stalemate:
            self.game_stats['stalemates'] += 1
            self.game_stats['stalemate_lengths'].append(game_length)
            if len(self.game_stats['stalemate_lengths']) > 50:
                self.game_stats['stalemate_lengths'] = self.game_stats['stalemate_lengths'][-50:]
        elif winner == 'white':
            self.game_stats['white_wins'] += 1
            self.game_stats['white_win_lengths'].append(game_length)
            if len(self.game_stats['white_win_lengths']) > 50:
                self.game_stats['white_win_lengths'] = self.game_stats['white_win_lengths'][-50:]
        elif winner == 'black':
            self.game_stats['black_wins'] += 1
            self.game_stats['black_win_lengths'].append(game_length)
            if len(self.game_stats['black_win_lengths']) > 50:
                self.game_stats['black_win_lengths'] = self.game_stats['black_win_lengths'][-50:]
        else:  # draw
            self.game_stats['draws'] += 1
            self.game_stats['draw_lengths'].append(game_length)
            if len(self.game_stats['draw_lengths']) > 50:
                self.game_stats['draw_lengths'] = self.game_stats['draw_lengths'][-50:]
    
    def _calc_avg(self, lengths):
        """Calculate average from list."""
        return sum(lengths) / len(lengths) if lengths else 0.0
    
    async def _send_final_message(self, ws, last_action):
        """Send final game summary message."""
        stats = self.game_stats
        
        final_msg = {
            "type": "game_end",
            "preyId": f"chess-{self.game.current_player}",
            "state": None,
            "reward": 0.0,
            "prev_state": None,
            "prev_action": last_action,
            "terminal": True,
            "winner": self.game.winner,
            "white_piece_type": ','.join(self.white_piece_types),
            "black_piece_type": ','.join(self.black_piece_types),
            "game_stats": {
                'white_wins': stats['white_wins'],
                'black_wins': stats['black_wins'],
                'draws': stats['draws'],
                'stalemates': stats['stalemates'],
                'total_games': stats['total_games']
            },
            "avg_game_lengths": {
                'white_wins': self._calc_avg(stats['white_win_lengths']),
                'black_wins': self._calc_avg(stats['black_win_lengths']),
                'draws': self._calc_avg(stats['draw_lengths']),
                'stalemates': self._calc_avg(stats['stalemate_lengths'])
            },
            "move_tracking": self.move_tracking.tolist()
        }
        
        try:
            await ws.send(json.dumps(final_msg))
        except Exception as e:
            print(f"[WARNING] Failed to send final message: {e}")
    
    def _print_summary(self, white_reward, black_reward):
        """Print episode summary."""
        stats = self.game_stats
        
        print(f"\nEpisode {self.episode_count} Complete:")
        print(f"  Winner: {self.game.winner}")
        print(f"  Game Length: {self.game.move_count} moves")
        print(f"  White Reward: {white_reward:.2f}")
        print(f"  Black Reward: {black_reward:.2f}")
        
        if stats['white_wins'] > 0:
            avg = self._calc_avg(stats['white_win_lengths'])
            print(f"  White wins avg: {avg:.1f} moves ({stats['white_wins']} games)")
        
        if stats['black_wins'] > 0:
            avg = self._calc_avg(stats['black_win_lengths'])
            print(f"  Black wins avg: {avg:.1f} moves ({stats['black_wins']} games)")
        
        if stats['draws'] > 0:
            avg = self._calc_avg(stats['draw_lengths'])
            print(f"  Draws avg: {avg:.1f} moves ({stats['draws']} games)")
        
        if stats['stalemates'] > 0:
            avg = self._calc_avg(stats['stalemate_lengths'])
            print(f"  Stalemates avg: {avg:.1f} moves ({stats['stalemates']} games)")


# ================================================
# MAIN
# ================================================

async def main():
    """Main entry point."""
    import sys
    
    print("Chess Backend Game for RL Training")
    print("=" * 50)
    
    # Parse command line args
    starting_game = 1
    
    if len(sys.argv) >= 3:
        white_pieces = sys.argv[1].split(',') if ',' in sys.argv[1] else [sys.argv[1], "king"]
        black_pieces = sys.argv[2].split(',') if ',' in sys.argv[2] else [sys.argv[2], "king"]
        if len(sys.argv) >= 4:
            try:
                starting_game = int(sys.argv[3])
            except ValueError:
                pass
    else:
        white_pieces = ["pawn", "king"]
        black_pieces = ["pawn", "king"]
    
    # Ensure king is included
    if "king" not in white_pieces:
        white_pieces.append("king")
    if "king" not in black_pieces:
        black_pieces.append("king")
    
    print(f"White: {white_pieces}")
    print(f"Black: {black_pieces}")
    print(f"Starting from game: {starting_game}")
    print(f"Connecting to {WS_URI}...")
    
    handler = ChessGameHandler(white_pieces, black_pieces)
    handler.episode_count = starting_game - 1
    
    try:
        async with websockets.connect(WS_URI) as ws:
            print("Connected!")
            
            for episode in range(starting_game, 1000000):
                await handler.run_episode(ws)
                await asyncio.sleep(0.1)
                
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Make sure RL server is running on localhost:8765")


if __name__ == "__main__":
    asyncio.run(main())
