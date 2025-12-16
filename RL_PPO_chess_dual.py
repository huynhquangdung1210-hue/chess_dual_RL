"""Dual Player Chess RL Training Script

This script trains both white and black chess players simultaneously using
separate PPO agents. Based on RL_PPO_fresh.py but modified for chess dual training.

Usage:
  python src/RL_PPO_chess_dual.py
  
Then run: python src/backend_game_chess.py

Requirements:
  pip install websockets torch numpy
"""

import asyncio
import json
import math
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import websockets

# ------------------ Config ------------------
WS_HOST = "localhost"
WS_PORT = 8765

# Chess-specific configuration
CHESS_ACTIONS = 4097  # 64x64 possible moves (from square to square)

# ================================================
# PIECE TYPE CONFIGURATION
# ================================================
# Configure which pieces each side will use
# Change these to set piece types for training
WHITE_PIECE_TYPE = ["rook","pawn","knight","bishop", "queen", "king"]    # Options: pawn, rook, knight, bishop, queen, king
BLACK_PIECE_TYPE = ["rook","pawn","knight","bishop", "queen", "king"]    # Options: pawn, rook, knight, bishop, queen, king

print(f"[Chess Config] Training: White {WHITE_PIECE_TYPE}s vs Black {BLACK_PIECE_TYPE}s")
# ================================================

# State for chess: 64 board squares + 3 metadata = 67 dimensions
STATE_DIM = 67  # Flattened 8x8 board + current_player + move_count + moves_remaining

# PPO Hyperparameters - Conservative values for stable learning
LEARNING_RATE = 1e-4   # More conservative for chess complexity
CRITIC_LR = 5e-4     # Same as actor for stable learning
GAMMA = 0.99             # Standard discount
GAE_LAMBDA = 0.95        # Standard GAE
PPO_EPOCHS = 6           # Fewer epochs to prevent overfitting
PPO_CLIP = 0.2           # Standard clipping
VALUE_CLIP = 0.1         # Value clipping
BATCH_SIZE = 64          # Match working fresh version
ENTROPY_COEF = 0.15     # Entropy coefficient
VALUE_COEF = 0.5         # Standard value coefficient
MAX_GRAD_NORM = 0.1      # Match working fresh version
TARGET_KL = 0.01         # Early stopping KL

# Value bounds
MAX_VALUE = 50.0         # Clip critic outputs
MAX_RETURN = 100.0       # Clip returns
MAX_ADVANTAGE = 20.0     # Clip advantages

# Training Configuration
BUFFER_SIZE = 2048       # Match working fresh version
UPDATE_INTERVAL = BUFFER_SIZE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Opponent Sampling Configuration (Self-Play Stabilization)
OPPONENT_POOL_SIZE = 10          # Keep last N checkpoints for each player
OPPONENT_SAMPLE_PROB = 0.8       # Probability of using historical opponent vs current
OPPONENT_UPDATE_FREQ = 50        # Save opponent checkpoint every N games

# Logging
LOG_DIR = Path(f"logs_w{WHITE_PIECE_TYPE}_b{BLACK_PIECE_TYPE}_dual")
LOG_DIR.mkdir(parents=True, exist_ok=True)

print(f"[Chess PPO Dual] Starting on {DEVICE}")
print(f"[Chess PPO Dual] STATE_DIM = {STATE_DIM}, ACTIONS = {CHESS_ACTIONS}")

# ------------------ Networks ------------------

class ChessActor(nn.Module):
    """Actor network for chess moves."""
    
    def __init__(self, state_dim, num_actions, hidden=128):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        
        # Action head for chess moves
        self.action_head = nn.Linear(hidden, num_actions)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, gain=0.5)
            torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        features = self.backbone(x)
        action_logits = self.action_head(features)
        return action_logits
    
    def get_action(self, x, action_mask=None):
        """Sample action and return log probs with action masking."""
        action_logits = self.forward(x)
        
        # Apply action masking
        if action_mask is not None:
            # Ensure mask has same shape as logits for proper broadcasting
            if action_mask.dim() == 1 and action_logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)
            
            # Ensure we have valid actions to choose from
            valid_actions = torch.sum(action_mask > 0.5)
            if valid_actions == 0:
                print(f"[ERROR] No valid actions in mask! Creating emergency random mask.")
                # Emergency fallback - allow first action
                action_mask = torch.zeros_like(action_mask)
                action_mask[..., 0] = 1.0  # Use ellipsis to handle any number of dimensions
            
            # Set invalid actions to very negative logits
            masked_logits = torch.where(
                action_mask > 0.5,  # Valid actions
                action_logits,
                torch.full_like(action_logits, -1e9)  # Invalid actions get -infinity
            )
            
            # Debug: Check if masking worked
            # Flatten for easier checking
            flat_mask = action_mask.view(-1) if action_mask.dim() > 1 else action_mask
            flat_logits = masked_logits.view(-1) if masked_logits.dim() > 1 else masked_logits
            invalid_indices = flat_mask <= 0.5
            if torch.any(invalid_indices):
                max_invalid_logit = torch.max(flat_logits[invalid_indices])
                if max_invalid_logit > -1e8:
                    print(f"[WARNING] Masking may have failed! Max invalid logit: {max_invalid_logit}")
                
        else:
            masked_logits = action_logits
        
        # Sample action from masked distribution
        action_dist = torch.distributions.Categorical(logits=masked_logits)
        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        # Final validation
        if action_mask is not None:
            action_idx = action.item() if action.dim() == 0 else action[0].item()
            # Flatten mask for validation
            flat_mask = action_mask.view(-1) if action_mask.dim() > 1 else action_mask
            mask_val = flat_mask[action_idx].item()
            if mask_val <= 0.5:
                print(f"[CRITICAL ERROR] Selected invalid action {action_idx} with mask value {mask_val}!")
                print(f"Valid actions: {torch.nonzero(flat_mask > 0.5).flatten().tolist()[:10]}...")
        
        return action, logp, entropy
    
    def get_action_logprobs(self, x, actions, action_mask=None):
        """Get log probabilities for given actions with masking."""
        action_logits = self.forward(x)
        
        # Apply action masking
        if action_mask is not None:
            # Ensure mask has same shape as logits for proper broadcasting
            if action_mask.dim() == 1 and action_logits.dim() == 2:
                action_mask = action_mask.unsqueeze(0)
            elif action_mask.dim() == 2 and action_logits.dim() == 2:
                # Ensure batch dimensions match
                if action_mask.shape[0] != action_logits.shape[0]:
                    action_mask = action_mask.expand(action_logits.shape[0], -1)
            
            masked_logits = torch.where(
                action_mask > 0.5,
                action_logits,
                torch.full_like(action_logits, -1e9)
            )
        else:
            masked_logits = action_logits
        
        action_dist = torch.distributions.Categorical(logits=masked_logits)
        logprobs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return logprobs, entropy

class ChessCritic(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        # Use orthogonal initialization for hidden layers
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        # Output layer with smaller gain but not too small
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc3.bias, 0.0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result = self.fc3(x).squeeze(-1)
        
        if torch.any(torch.isnan(result)) or torch.any(torch.isinf(result)):
            print(f"[CRITIC WARNING] NaN or Inf detected!")
            result = torch.zeros_like(result)
        
        return result

# Create networks for both players
white_actor = ChessActor(STATE_DIM, CHESS_ACTIONS).to(DEVICE)
white_critic = ChessCritic(STATE_DIM).to(DEVICE)
black_actor = ChessActor(STATE_DIM, CHESS_ACTIONS).to(DEVICE)
black_critic = ChessCritic(STATE_DIM).to(DEVICE)

# Optimizers
white_actor_optimizer = optim.Adam(white_actor.parameters(), lr=LEARNING_RATE)
white_critic_optimizer = optim.Adam(white_critic.parameters(), lr=CRITIC_LR)
black_actor_optimizer = optim.Adam(black_actor.parameters(), lr=LEARNING_RATE)
black_critic_optimizer = optim.Adam(black_critic.parameters(), lr=CRITIC_LR)

print(f"[Chess PPO] White Actor params: {sum(p.numel() for p in white_actor.parameters()):,}")
print(f"[Chess PPO] White Critic params: {sum(p.numel() for p in white_critic.parameters()):,}")
print(f"[Chess PPO] Black Actor params: {sum(p.numel() for p in black_actor.parameters()):,}")
print(f"[Chess PPO] Black Critic params: {sum(p.numel() for p in black_critic.parameters()):,}")

# ------------------ Opponent Pool (Self-Play Stabilization) ------------------

class OpponentPool:
    """Manages a pool of historical opponents for self-play stabilization.
    
    Instead of always playing against the latest version of the opponent,
    we randomly sample from recent checkpoints. This reduces non-stationarity
    and stabilizes training.
    """
    
    def __init__(self, pool_size, state_dim, num_actions):
        self.pool_size = pool_size
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # Store state dicts (not full models to save memory)
        self.white_actors = deque(maxlen=pool_size)
        self.black_actors = deque(maxlen=pool_size)
        
        # Current opponent models (lazily loaded)
        self._current_white_opponent = None
        self._current_black_opponent = None
        
        # Track which opponent is currently loaded
        self._white_opponent_idx = -1
        self._black_opponent_idx = -1
        
    def add_white_checkpoint(self, actor_state_dict):
        """Add a white actor checkpoint to the pool."""
        # Deep copy the state dict
        self.white_actors.append({k: v.clone() for k, v in actor_state_dict.items()})
        print(f"[OpponentPool] Added white checkpoint ({len(self.white_actors)}/{self.pool_size})")
        
    def add_black_checkpoint(self, actor_state_dict):
        """Add a black actor checkpoint to the pool."""
        self.black_actors.append({k: v.clone() for k, v in actor_state_dict.items()})
        print(f"[OpponentPool] Added black checkpoint ({len(self.black_actors)}/{self.pool_size})")
    
    def get_white_opponent(self, use_current_prob=0.2):
        """Get a white opponent actor (for black to play against).
        
        Args:
            use_current_prob: Probability of returning None (use current model)
        
        Returns:
            Actor model or None (if should use current model)
        """
        if len(self.white_actors) == 0 or np.random.random() < use_current_prob:
            return None  # Use current model
        
        # Sample random historical opponent
        idx = np.random.randint(0, len(self.white_actors))
        
        # Only reload if different from current
        if idx != self._white_opponent_idx or self._current_white_opponent is None:
            self._current_white_opponent = ChessActor(self.state_dim, self.num_actions).to(DEVICE)
            self._current_white_opponent.load_state_dict(self.white_actors[idx])
            self._current_white_opponent.eval()  # Set to eval mode
            self._white_opponent_idx = idx
            
        return self._current_white_opponent
    
    def get_black_opponent(self, use_current_prob=0.2):
        """Get a black opponent actor (for white to play against).
        
        Args:
            use_current_prob: Probability of returning None (use current model)
        
        Returns:
            Actor model or None (if should use current model)
        """
        if len(self.black_actors) == 0 or np.random.random() < use_current_prob:
            return None  # Use current model
        
        # Sample random historical opponent
        idx = np.random.randint(0, len(self.black_actors))
        
        # Only reload if different from current
        if idx != self._black_opponent_idx or self._current_black_opponent is None:
            self._current_black_opponent = ChessActor(self.state_dim, self.num_actions).to(DEVICE)
            self._current_black_opponent.load_state_dict(self.black_actors[idx])
            self._current_black_opponent.eval()  # Set to eval mode
            self._black_opponent_idx = idx
            
        return self._current_black_opponent
    
    def sample_new_opponents(self):
        """Sample new opponents for both players for the next game.
        
        Returns:
            (white_opponent, black_opponent) - either Actor or None for each
        """
        use_current = 1.0 - OPPONENT_SAMPLE_PROB
        white_opp = self.get_white_opponent(use_current_prob=use_current)
        black_opp = self.get_black_opponent(use_current_prob=use_current)
        return white_opp, black_opp
    
    def get_pool_status(self):
        """Get status string for logging."""
        return f"White pool: {len(self.white_actors)}/{self.pool_size}, Black pool: {len(self.black_actors)}/{self.pool_size}"

# Create opponent pool
opponent_pool = OpponentPool(OPPONENT_POOL_SIZE, STATE_DIM, CHESS_ACTIONS)
print(f"[Chess PPO] Opponent pool initialized (size={OPPONENT_POOL_SIZE}, sample_prob={OPPONENT_SAMPLE_PROB})")

# ------------------ Rollout Buffer ------------------

class ChessRolloutBuffer:
    """Rollout buffer for chess training with action masking."""
    
    def __init__(self, size, state_dim, action_dim):
        self.size = size
        self.ptr = 0
        self.full = False
        
        # Buffers
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.action_masks = np.zeros((size, action_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.logprobs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        
        # GAE
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
    
    def add(self, state, action_mask, action, logprob, value, reward, done):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.action_masks[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def get_size(self):
        return self.size if self.full else self.ptr
    
    def compute_gae(self, last_value=0.0):
        """Compute GAE advantages and returns."""
        size = self.get_size()
        advantages = np.zeros_like(self.advantages[:size])
        
        last_gae = 0
        for step in reversed(range(size)):
            if step == size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + GAMMA * next_value * next_non_terminal - self.values[step]
            advantages[step] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae
        
        # Clip advantages
        advantages = np.clip(advantages, -MAX_ADVANTAGE, MAX_ADVANTAGE)
        
        # Compute returns
        returns = advantages + self.values[:size]
        returns = np.clip(returns, -MAX_RETURN, MAX_RETURN)
        
        self.advantages[:size] = advantages
        self.returns[:size] = returns
    
    def get_batch(self):
        """Get all data as tensors."""
        size = self.get_size()
        
        # Normalize advantages
        advantages = self.advantages[:size]
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': torch.FloatTensor(self.states[:size]).to(DEVICE),
            'action_masks': torch.FloatTensor(self.action_masks[:size]).to(DEVICE),
            'actions': torch.LongTensor(self.actions[:size]).to(DEVICE),
            'logprobs': torch.FloatTensor(self.logprobs[:size]).to(DEVICE),
            'returns': torch.FloatTensor(self.returns[:size]).to(DEVICE),
            'values': torch.FloatTensor(self.values[:size]).to(DEVICE),
            'advantages': torch.FloatTensor(advantages).to(DEVICE)
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.full = False

# Create buffers for both players
white_buffer = ChessRolloutBuffer(BUFFER_SIZE, STATE_DIM, CHESS_ACTIONS)
black_buffer = ChessRolloutBuffer(BUFFER_SIZE, STATE_DIM, CHESS_ACTIONS)

# ------------------ State Preprocessing (like RL_PPO_fresh) ------------------

class RunningMeanStd:
    """Running statistics for state normalization."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        # Ensure x is numpy array to avoid deprecation warnings
        x = np.asarray(x, dtype=np.float32)
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

# Global state normalizer for consistent normalization
state_normalizer = RunningMeanStd(shape=(STATE_DIM,))

def normalize_state(state_input):
    """Normalize state using running statistics."""
    # Convert tensor to numpy if needed
    if torch.is_tensor(state_input):
        state_array = state_input.detach().cpu().numpy()
    else:
        state_array = np.array(state_input, dtype=np.float32)
    
    # Ensure proper numpy operations to avoid deprecation warnings
    mean_array = np.asarray(state_normalizer.mean, dtype=np.float32)
    var_array = np.asarray(state_normalizer.var, dtype=np.float32)
    
    normalized = (state_array - mean_array) / np.sqrt(var_array + 1e-8)
    return normalized.astype(np.float32)

def preprocess_chess_state(raw_state):
    """Preprocess chess state to ensure consistent STATE_DIM."""
    if raw_state is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    # If already the right size, use as is
    if isinstance(raw_state, (list, tuple)) and len(raw_state) == STATE_DIM:
        return np.array(raw_state, dtype=np.float32)
    
    # Pad or truncate as needed
    state_array = np.array(raw_state, dtype=np.float32)
    if len(state_array) < STATE_DIM:
        state_array = np.pad(state_array, (0, STATE_DIM - len(state_array)))
    elif len(state_array) > STATE_DIM:
        state_array = state_array[:STATE_DIM]
    
    return state_array

# ------------------ PPO Update ------------------

def ppo_update_player(actor, critic, actor_optimizer, critic_optimizer, buffer):
    """Perform PPO update for one player with action masking."""
    if buffer.get_size() < BATCH_SIZE:
        return None
    
    # Compute GAE
    buffer.compute_gae()
    
    # Get batch
    batch = buffer.get_batch()
    
    # Convert states to normalized tensors consistently
    normalized_states = []
    for s in batch['states']:
        # Handle both tensor and numpy inputs
        norm_state = normalize_state(s)
        normalized_states.append(norm_state)
    
    # Convert list to numpy array first to avoid slow tensor creation
    normalized_states = np.array(normalized_states, dtype=np.float32)
    states_norm = torch.FloatTensor(normalized_states).to(DEVICE)
    
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0
    
    # PPO epochs
    for epoch in range(PPO_EPOCHS):
        # Forward pass with normalized states
        values = critic(states_norm)
        
        # Policy forward pass with action masking
        new_logprobs, entropy = actor.get_action_logprobs(
            states_norm, 
            batch['actions'], 
            batch['action_masks']
        )
        entropy_mean = entropy.mean()
        
        # Ratio and clipped objective
        ratio = torch.exp(new_logprobs - batch['logprobs'])
        surr1 = ratio * batch['advantages']
        surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * batch['advantages']
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss with clipping like in fresh
        values_clipped = batch['values'] + torch.clamp(
            values - batch['values'], -VALUE_CLIP, VALUE_CLIP
        )
        value_loss1 = F.mse_loss(values, batch['returns'])
        value_loss2 = F.mse_loss(values_clipped, batch['returns'])
        value_loss = torch.max(value_loss1, value_loss2)
        
        # Entropy loss (negative like in fresh)
        entropy_loss = -entropy_mean
        
        # Total loss like in fresh - combined update
        total_loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
        
        # Combined update like in fresh
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
        actor_optimizer.step()
        critic_optimizer.step()
        
        # Accumulate metrics
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy_mean.item()  # Positive entropy now
        n_updates += 1
        
        # Early stopping on KL divergence - use stored vs new like fresh
        with torch.no_grad():
            kl_div = torch.mean(batch['logprobs'] - new_logprobs).item()  # Stored vs new like fresh
            if kl_div > TARGET_KL:
                break
    
    # Compute explained variance
    old_values = batch['values']
    returns = batch['returns']
    advantages = batch['advantages']
    
    # Compute explained variance with normalized states
    with torch.no_grad():
        current_values = critic(states_norm).cpu().numpy()
    
    y_pred = current_values
    y_true = returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
    # Calculate approx KL divergence with normalized states - fix direction
    old_logprobs = batch['logprobs']
    with torch.no_grad():
        new_logprobs_for_kl, _ = actor.get_action_logprobs(
            states_norm, 
            batch['actions'], 
            batch['action_masks']
        )
        approx_kl = torch.mean(batch['logprobs'] - new_logprobs_for_kl).item()  # Stored vs new like fresh
    
    # Clear buffer
    buffer.clear()
    
    # Return comprehensive metrics matching RL_PPO_fresh.py format
    return {
        'policy_loss': total_policy_loss / max(n_updates, 1),
        'value_loss': total_value_loss / max(n_updates, 1),
        'entropy': total_entropy / max(n_updates, 1),  # Positive like in fresh
        'approx_kl': approx_kl,
        'explained_variance': explained_var,
        'n_updates': n_updates,
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item(),
        'return_mean': returns.mean().item(),
        'return_std': returns.std().item(),
        'value_mean': old_values.mean().item(),
        'value_std': old_values.std().item(),
    }

# ------------------ Stats Tracking ------------------

# Global stats
episode_count = 0
white_wins = 0
black_wins = 0
stalemate_white = 0
stalemate_black = 0
draws = 0
recent_rewards = deque(maxlen=100)  # Combined rewards for compatibility

# Separate reward tracking for white and black
white_recent_rewards = deque(maxlen=100)
black_recent_rewards = deque(maxlen=100)

# Game length tracking for wins/losses (excluding draws)
win_game_lengths = deque(maxlen=100)  # Track game lengths when there's a winner
white_win_lengths = deque(maxlen=50)  # Track white win lengths
stalemate_white_lengths = deque(maxlen=50)  # Track white draw by exhaust move lengths
stalemate_black_lengths = deque(maxlen=50)  # Track black draw by exhaust move lengths
black_win_lengths = deque(maxlen=50)  # Track black win lengths
recent_game_lengths = deque(maxlen=10)  # Track last 10 game lengths for averaging

# Comprehensive stats tracking matching RL_PPO_fresh.py format
stats = {
    'total_messages': 0,
    'total_updates': 0,
    'white_total_reward': 0.0,
    'black_total_reward': 0.0,
    'avg_game_length': 0.0,
    'avg_win_length': 0.0,
    'avg_white_win_length': 0.0,
    'avg_black_win_length': 0.0,
    'avg_stalemate_white_length': 0.0,
    'avg_stalemate_black_length': 0.0,
    # Opponent sampling stats
    'white_historical_actions': 0,
    'white_current_actions': 0,
    'black_historical_actions': 0,
    'black_current_actions': 0,
}

# Backend data tracking
last_game_stats = {}
last_move_tracking = None  # Move tracking array from backend (cumulative)
period_move_tracking = np.zeros(CHESS_ACTIONS, dtype=np.int32)  # Moves since last log save

last_metrics = {'white': None, 'black': None}

def convert_to_json_serializable(obj):
    """Convert numpy and torch objects to JSON serializable types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # torch tensors
        return float(obj.item()) if obj.numel() == 1 else obj.tolist()
    elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
        return float(obj)
    elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    else:
        return obj

async def save_episode_log(episode, white_reward, black_reward):
    """Save comprehensive episode log with all metrics for both players."""
    global period_move_tracking  # Declare global at start of function
    import time
    
    timestamp = int(time.time())
    
    # Calculate recent average returns properly for dual training
    white_recent_returns = list(white_recent_rewards)[-50:]  # Last 50 white rewards
    black_recent_returns = list(black_recent_rewards)[-50:]  # Last 50 black rewards
    
    # Calculate average game lengths from recent games (last 10)
    avg_white_win_length = stats.get('avg_white_win_length', 0.0)
    avg_black_win_length = stats.get('avg_black_win_length', 0.0) 
    recent_lengths = list(recent_game_lengths)
    avg_recent_game_length = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
    
    log_data = {
        'timestamp': timestamp,
        'episode': episode,
        'config': {
            'LEARNING_RATE': float(LEARNING_RATE),
            'GAMMA': float(GAMMA),
            'GAE_LAMBDA': float(GAE_LAMBDA),
            'PPO_EPOCHS': int(PPO_EPOCHS),
            'PPO_CLIP': float(PPO_CLIP),
            'BATCH_SIZE': int(BATCH_SIZE),
            'ENTROPY_COEF': float(ENTROPY_COEF),
            'VALUE_COEF': float(VALUE_COEF),
            'MAX_GRAD_NORM': float(MAX_GRAD_NORM),
            'WHITE_PIECE_TYPE': WHITE_PIECE_TYPE,
            'BLACK_PIECE_TYPE': BLACK_PIECE_TYPE,
            'OPPONENT_POOL_SIZE': OPPONENT_POOL_SIZE,
            'OPPONENT_SAMPLE_PROB': OPPONENT_SAMPLE_PROB,
        },
        'game_stats': {
            'white_wins': white_wins,
            'black_wins': black_wins,
            'draws': draws,
            'stalemate_white': stalemate_white,
            'stalemate_black': stalemate_black,
            'total_games': white_wins + black_wins + draws + stalemate_white + stalemate_black,
            'white_win_rate': white_wins / max(white_wins + black_wins + draws + stalemate_white + stalemate_black, 1),
            'black_win_rate': black_wins / max(white_wins + black_wins + draws + stalemate_white + stalemate_black, 1),
        },
        'white_metrics': {
            'recent_avg_return': float(sum(white_recent_returns) / len(white_recent_returns)) if white_recent_returns else 0.0,
            'total_reward': float(stats.get('white_total_reward', 0.0)),
            **(last_metrics.get('white') if last_metrics.get('white') is not None else {})
        },
        'black_metrics': {
            'recent_avg_return': float(sum(black_recent_returns) / len(black_recent_returns)) if black_recent_returns else 0.0,
            'total_reward': float(stats.get('black_total_reward', 0.0)),
            **(last_metrics.get('black') if last_metrics.get('black') is not None else {})
        },
        'game_length_analysis': {
            'avg_white_win_length': float(avg_white_win_length),
            'avg_black_win_length': float(avg_black_win_length),
            'avg_recent_game_length': float(avg_recent_game_length),
            'total_games_analyzed': int(white_wins + black_wins + draws + stalemate_white + stalemate_black)
        },
        'opponent_sampling': {
            'pool_status': opponent_pool.get_pool_status(),
            'white_pool_size': len(opponent_pool.white_actors),
            'black_pool_size': len(opponent_pool.black_actors),
            'white_historical_actions': stats.get('white_historical_actions', 0),
            'white_current_actions': stats.get('white_current_actions', 0),
            'black_historical_actions': stats.get('black_historical_actions', 0),
            'black_current_actions': stats.get('black_current_actions', 0),
        },
        'move_tracking': period_move_tracking.tolist()  # Only moves from last N episodes
    }
    
    # Reset period move tracking after saving
    period_move_tracking = np.zeros(CHESS_ACTIONS, dtype=np.int32)
    
    log_path = LOG_DIR / f"episode_{episode}_{timestamp}.json"
    with open(log_path, 'w') as f:
        # Convert all data to JSON serializable format
        serializable_data = convert_to_json_serializable(log_data)
        json.dump(serializable_data, f, indent=2)
    print(f"[Chess PPO] Saved episode log: {log_path}")

async def save_checkpoint(episode):
    """Save model checkpoints."""
    checkpoint = {
        'episode': episode,
        'white_actor_state_dict': white_actor.state_dict(),
        'white_critic_state_dict': white_critic.state_dict(),
        'black_actor_state_dict': black_actor.state_dict(),
        'black_critic_state_dict': black_critic.state_dict(),
        'white_actor_optimizer': white_actor_optimizer.state_dict(),
        'white_critic_optimizer': white_critic_optimizer.state_dict(),
        'black_actor_optimizer': black_actor_optimizer.state_dict(),
        'black_critic_optimizer': black_critic_optimizer.state_dict(),
        'stats': stats
    }
    
    checkpoint_path = LOG_DIR / f"chess_dual_checkpoint_episode_{episode}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"[Chess PPO] Saved checkpoint: {checkpoint_path}")

# ------------------ WebSocket Handler ------------------

async def websocket_handler(websocket):
    """Handle WebSocket communication for dual chess training."""
    global episode_count, white_wins, black_wins, draws, stalemate_white, stalemate_black, stats, recent_rewards, last_metrics
    global white_recent_rewards, black_recent_rewards, last_game_stats, last_move_tracking, period_move_tracking, recent_game_lengths
    
    print(f"[Chess PPO Dual] Client connected")
    print(f"[Chess PPO Dual] Configured for: White {WHITE_PIECE_TYPE}s vs Black {BLACK_PIECE_TYPE}s")
    print(f"[Chess PPO Dual] Opponent sampling enabled: pool_size={OPPONENT_POOL_SIZE}, sample_prob={OPPONENT_SAMPLE_PROB}")
    
    games_in_curr_ep_white_reward = 0.0
    games_in_curr_ep_black_reward = 0.0
    
    games_completed = 0
    total_games_for_opponent_update = 0  # Track games for opponent checkpoint updates
    GAMES_PER_EPISODE = 100
    
    # Current opponents (None means use current training model)
    current_white_opponent = None  # Opponent for black to face
    current_black_opponent = None  # Opponent for white to face

    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            continue
        
        msg_type = data.get('type', '')
        
        # Track PPO update count separately from episodes
        ppo_update_count = getattr(websocket_handler, 'ppo_update_count', 0)
        
        # Check if buffers are full - trigger PPO update
        if white_buffer.get_size() >= white_buffer.size or black_buffer.get_size() >= black_buffer.size:
            ppo_update_count += 1
            websocket_handler.ppo_update_count = ppo_update_count
            
            
            # Perform PPO updates for both players
            white_metrics = ppo_update_player(white_actor, white_critic, 
                                            white_actor_optimizer, white_critic_optimizer, 
                                            white_buffer)
            black_metrics = ppo_update_player(black_actor, black_critic,
                                            black_actor_optimizer, black_critic_optimizer,
                                            black_buffer)
            
            if white_metrics:
                last_metrics['white'] = white_metrics
                print(f"[White PPO] "
                    f"policy_loss={white_metrics['policy_loss']:.4f}, "
                    f"value_loss={white_metrics['value_loss']:.4f}, "
                    f"entropy={white_metrics['entropy']:.4f}")
            
            if black_metrics:
                last_metrics['black'] = black_metrics
                print(f"[Black PPO] "
                    f"policy_loss={black_metrics['policy_loss']:.4f}, "
                    f"value_loss={black_metrics['value_loss']:.4f}, "
                    f"entropy={black_metrics['entropy']:.4f}")
        
        # Handle episode/game end
        if msg_type in ('game_end', 'episode_done', 'gameOver'):
            # Track game completion
            games_completed += 1
            
            # Check if episode is complete (100 games)
            if games_completed >= GAMES_PER_EPISODE:
                games_completed = 0
                episode_count += 1
                
                # Print game statistics at end of episode
                total_games = white_wins + black_wins + draws + stalemate_white + stalemate_black
                if total_games > 0:
                    print(f"[Chess Stats] Episode {episode_count} ({white_piece_type} vs {black_piece_type}):")
                    print(f"  W:{white_wins}({100*white_wins/total_games:.1f}%) "
                        f"B:{black_wins}({100*black_wins/total_games:.1f}%) "
                        f"D:{draws}({100*draws/total_games:.1f}%) "
                        f"SW:{stalemate_white}({100*stalemate_white/total_games:.1f}%) "
                        f"SB:{stalemate_black}({100*stalemate_black/total_games:.1f}%)")
                
                # Save episode log every 10 episodes
                if episode_count % 10 == 0:
                    await save_episode_log(episode_count, games_in_curr_ep_white_reward, games_in_curr_ep_black_reward)
                
                # Save checkpoint periodically
                if episode_count % 100 == 0:
                    await save_checkpoint(episode_count)

            # Update opponent pool periodically
            total_games_for_opponent_update += 1
            if total_games_for_opponent_update >= OPPONENT_UPDATE_FREQ:
                total_games_for_opponent_update = 0
                # Add current models to opponent pool
                opponent_pool.add_white_checkpoint(white_actor.state_dict())
                opponent_pool.add_black_checkpoint(black_actor.state_dict())
            
            # Sample new opponents for the next game
            current_white_opponent, current_black_opponent = opponent_pool.sample_new_opponents()

            # Extract game info
            winner = data.get('winner', 'unknown')
            move_count = data.get('move_count', 0)
            white_piece_type = data.get('white_piece_type', 'unknown')
            black_piece_type = data.get('black_piece_type', 'unknown')
            raw_reward = float(data.get('reward', 0.0))     
            terminal = bool(data.get('terminal', False))
            prey_id = data.get('preyId', '')
            
            # Capture enhanced backend data if available
            if 'game_stats' in data:
                last_game_stats = data['game_stats']
                # Update our stats with backend averages
                if 'white_win_lengths' in last_game_stats and last_game_stats['white_win_lengths']:
                    stats['avg_white_win_length'] = sum(last_game_stats['white_win_lengths']) / len(last_game_stats['white_win_lengths'])
                if 'black_win_lengths' in last_game_stats and last_game_stats['black_win_lengths']:
                    stats['avg_black_win_length'] = sum(last_game_stats['black_win_lengths']) / len(last_game_stats['black_win_lengths'])
                if 'draw_lengths' in last_game_stats and last_game_stats['draw_lengths']:
                    stats['avg_stalemate_white_length'] = sum(last_game_stats['draw_lengths']) / len(last_game_stats['draw_lengths'])
            
            # Capture move_tracking from backend and update period tracking
            if 'move_tracking' in data:
                new_move_tracking = np.array(data['move_tracking'], dtype=np.int32)
                # Calculate delta since last update to add to period tracking
                if last_move_tracking is not None:
                    delta = new_move_tracking - last_move_tracking
                    period_move_tracking += np.maximum(delta, 0)  # Only add positive deltas
                else:
                    period_move_tracking += new_move_tracking
                last_move_tracking = new_move_tracking

            recent_game_lengths.append(move_count)  # Track for averaging
            
            # Track total rewards from game_end messages
            if 'white' in prey_id:
                stats['white_total_reward'] += raw_reward
                games_in_curr_ep_white_reward += raw_reward
            elif 'black' in prey_id:
                stats['black_total_reward'] += raw_reward
                games_in_curr_ep_black_reward += raw_reward
            
            white_recent_rewards.append(games_in_curr_ep_white_reward)
            black_recent_rewards.append(games_in_curr_ep_black_reward)
            games_in_curr_ep_white_reward = 0.0
            games_in_curr_ep_black_reward = 0.0
            

            
            # Update stats and track game lengths for wins/losses
            if winner == 'white':
                white_wins += 1
                win_game_lengths.append(move_count)
                white_win_lengths.append(move_count)
            elif winner == 'black':
                black_wins += 1
                win_game_lengths.append(move_count)
                black_win_lengths.append(move_count)
            elif winner == 'stalemate_white':
                stalemate_white += 1
                stalemate_white_lengths.append(move_count)
            elif winner == 'stalemate_black':
                stalemate_black += 1
                stalemate_black_lengths.append(move_count)
            else:
                draws += 1
            
            # Update average game lengths
            if win_game_lengths:
                stats['avg_win_length'] = sum(win_game_lengths) / len(win_game_lengths)
            if white_win_lengths:
                stats['avg_white_win_length'] = sum(white_win_lengths) / len(white_win_lengths)
            if black_win_lengths:
                stats['avg_black_win_length'] = sum(black_win_lengths) / len(black_win_lengths)
            if stalemate_white_lengths:
                stats['avg_stalemate_white_length'] = sum(stalemate_white_lengths) / len(stalemate_white_lengths)
            if stalemate_black_lengths:
                stats['avg_stalemate_black_length'] = sum(stalemate_black_lengths) / len(stalemate_black_lengths)
            
                

            continue
        
        # Extract data
        prey_id = data.get('preyId', '')
        raw_state = data.get('state', [])
        raw_action_mask = data.get('action_mask', [])
        raw_reward = float(data.get('reward', 0.0))
        terminal = bool(data.get('terminal', False))

        # Determine player color from prey_id
        if 'white' in prey_id:
            player_color = 'white'
            critic = white_critic
            buffer = white_buffer
            games_in_curr_ep_white_reward += raw_reward
            
            # Opponent sampling: White may face historical black opponent
            # When black uses historical version, white sees different opponent behavior
            # But white always uses current actor for ITS actions and training
            # The key insight: we use historical opponent for the OTHER player's actions
            # Here white is acting, so we use white's current actor
            # But black (opponent) may be using historical version - this happens
            # when black gets its turn and uses current_white_opponent
            actor = white_actor  # White always uses current for its own moves
            
        elif 'black' in prey_id:
            player_color = 'black'
            critic = black_critic
            buffer = black_buffer
            games_in_curr_ep_black_reward += raw_reward
            
            # Similarly, black always uses current actor for its own moves
            # The opponent sampling affects what VERSION of white black faces
            # This happens implicitly through current_black_opponent
            actor = black_actor  # Black always uses current for its own moves
            
        else:
            print(f"[Warning] Unknown player ID: {prey_id}")
            continue
        
        # OPPONENT SAMPLING IMPLEMENTATION:
        # Instead of always using current actor, sometimes use historical version
        # This creates diversity in training data and stabilizes learning
        # 
        # When white's turn: use current_white_opponent if set (white faces historical self-play)
        # When black's turn: use current_black_opponent if set (black faces historical self-play)
        #
        # The "opponent" is actually the player themselves from a past checkpoint.
        # This is equivalent to having the player face opponents of varying skill levels.
        action_actor = actor  # Default to current actor
        use_historical = False
        
        if player_color == 'white' and current_white_opponent is not None:
            # Use historical white actor for action selection (but train current)
            action_actor = current_white_opponent
            use_historical = True
        elif player_color == 'black' and current_black_opponent is not None:
            # Use historical black actor for action selection (but train current)
            action_actor = current_black_opponent
            use_historical = True
        
        # Update stats
        stats['total_messages'] += 1
        if player_color == 'white':
            stats['white_total_reward'] += raw_reward
            if use_historical:
                stats['white_historical_actions'] += 1
            else:
                stats['white_current_actions'] += 1
        else:
            stats['black_total_reward'] += raw_reward
            if use_historical:
                stats['black_historical_actions'] += 1
            else:
                stats['black_current_actions'] += 1
        
        # Preprocess state using new function
        state = preprocess_chess_state(raw_state)
        
        # Update state normalizer and normalize state
        state_normalizer.update(state.reshape(1, -1))
        state_norm = normalize_state(state)
        
        # Convert to tensor using normalized state
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(DEVICE)
        
        # Preprocess action mask
        if len(raw_action_mask) != CHESS_ACTIONS:
            print(f"[Warning] Action mask dimension mismatch: expected {CHESS_ACTIONS}, got {len(raw_action_mask)}")
            # Create default mask (all invalid - should not happen)
            action_mask = np.zeros(CHESS_ACTIONS, dtype=np.float32)
        else:
            action_mask = np.array(raw_action_mask, dtype=np.float32)
        
        # Debug action mask - compare with what backend sent
        valid_count = np.sum(action_mask > 0.5)
        expected_valid_count = data.get('valid_count', -1)
        msg_seq = data.get('msg_seq', -1)
        
        if valid_count == 0:
            print(f"[ERROR] Received action mask with no valid actions for {player_color} with terminal {terminal}!")
        
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(DEVICE)
        
        # Get action using action_actor (may be historical) and value using current critic
        # Action comes from potentially historical model (for diverse training)
        # Value always comes from current model (for proper training signal)
        with torch.no_grad():
            action, logprob, entropy = action_actor.get_action(state_tensor, action_mask_tensor)
            value = critic(state_tensor).item()
            
            # If using historical actor, we need to get logprob from CURRENT actor for PPO
            # The stored logprob should be from the policy that generated the action
            if use_historical:
                # Get logprob from current actor for the action taken by historical actor
                current_logprob, _ = actor.get_action_logprobs(
                    state_tensor, action.unsqueeze(0), action_mask_tensor
                )
                logprob = current_logprob.squeeze()
        
        # Convert to numpy
        action_idx = action.item()
        # Verify action is valid (sanity check)
        # Handle batch dimension when checking mask
        mask_for_check = action_mask
        if action_mask_tensor.dim() > 1:
            mask_for_check = action_mask  # Keep original numpy array for indexing
        
        if mask_for_check[action_idx] < 0.5:
            print(f"[ERROR] {player_color} selected invalid action {action_idx}! Action mask failed!")
            print(f"Valid actions available: {np.nonzero(mask_for_check > 0.5)[0][:10].tolist()}...")
            print(f"Action mask shape: {action_mask.shape}, tensor shape: {action_mask_tensor.shape}")
            # Emergency: pick first valid action
            valid_actions = np.nonzero(mask_for_check > 0.5)[0]
            if len(valid_actions) > 0:
                action_idx = int(valid_actions[0])
                print(f"Emergency override: using action {action_idx}")
            else:
                print(f"No valid actions found! Using action 0")
                action_idx = 0
        
        # Store in buffer - store original state for normalization updates, value from normalized state
        buffer.add(
            state=state,  # Original state for normalization updates
            action_mask=action_mask,
            action=action_idx,
            logprob=logprob.item(),
            value=value,  # Value computed from normalized state
            reward=raw_reward,
            done=terminal
        )
        
        # Debug buffer filling
        if buffer.get_size() % 1000 == 0:
            print(f"[{player_color.title()} Buffer] Size: {buffer.get_size()}/{buffer.size}")
        
        # Also log every 500 messages to track total flow
        stats['total_messages'] += 1
        # if stats['total_messages'] % 500 == 0:
        #     print(f"[Flow Debug] Total messages: {stats['total_messages']}, "
        #           f"White buffer: {white_buffer.get_size()}, Black buffer: {black_buffer.get_size()}")
        
        # Get sequence info from request for echo back
        msg_seq = data.get('msg_seq', -1)
        mask_hash = data.get('mask_hash', -1)
        
        # Prepare response with sequence info echoed back
        response = {
            'preyId': prey_id,
            'action_idx': action_idx,
            'msg_seq': msg_seq,
            'mask_hash': mask_hash
        }
        
        await websocket.send(json.dumps(response))


# ------------------ Main ------------------

async def main():
    """Main training loop."""
    print(f"[Chess PPO Dual] Starting server on ws://{WS_HOST}:{WS_PORT}")
    print(f"[Chess PPO Dual] Training both White and Black players")
    print(f"[Chess PPO Dual] Buffer size: {BUFFER_SIZE}")
    
    async with websockets.serve(websocket_handler, WS_HOST, WS_PORT):
        await asyncio.Future()  # Run forever


if __name__ == '__main__':
    asyncio.run(main())