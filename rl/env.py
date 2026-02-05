"""
Reinforcement Learning Environment for Dynamic Inference

The environment simulates the decision-making process of when to exit
during neural network inference.
"""

import gym
from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F


class DynamicInferenceEnv(gym.Env):
    """
    RL Environment for learning when to exit early.
    
    State Space:
        - Confidence (entropy) of current exit
        - Current layer index (normalized)
        - Cumulative latency so far (normalized)
        - Remaining compute budget (normalized)
        
    Action Space:
        - 0: CONTINUE to next layer
        - 1: EXIT and use current prediction
        
    Reward:
        reward = α × correct - β × latency - γ × compute
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, model, device_config, reward_config, device='cpu'):
        """
        Args:
            model: Vision model with early exits
            device_config: Device constraints (latency, compute)
            reward_config: Reward function weights
            device: torch device
        """
        super().__init__()
        
        self.model = model
        self.device_config = device_config
        self.reward_config = reward_config
        self.device = device
        
        self.num_exits = model.get_num_exits()
        
        # State space: [confidence, layer_idx, latency, compute_budget]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )
        
        # Action space: CONTINUE (0) or EXIT (1)
        self.action_space = spaces.Discrete(2)
        
        # Episode state
        self.current_image = None
        self.current_label = None
        self.current_exit = 0
        self.cumulative_latency = 0.0
        self.cumulative_compute = 0.0
        self.features_cache = []
        self.done = False
        
    def reset(self, image=None, label=None):
        """
        Reset environment with new image.
        
        Args:
            image: Input image tensor [1, 3, 32, 32]
            label: Ground truth label
            
        Returns:
            state: Initial state observation
        """
        self.current_image = image
        self.current_label = label
        self.current_exit = 0
        self.cumulative_latency = 0.0
        self.cumulative_compute = 0.0
        self.done = False
        
        # Precompute all features
        with torch.no_grad():
            _, self.features_cache, _ = self.model(
                image, 
                return_features=True
            )
        
        # Get initial state
        state = self._get_state()
        return state
    
    def step(self, action):
        """
        Take action (CONTINUE or EXIT).
        
        Args:
            action: 0 (CONTINUE) or 1 (EXIT)
            
        Returns:
            state: Next state
            reward: Reward for this action
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")
        
        # Action 1: EXIT
        if action == 1:
            reward, info = self._compute_reward_and_exit()
            self.done = True
            next_state = self._get_state()
            return next_state, reward, self.done, info
        
        # Action 0: CONTINUE
        else:
            self.current_exit += 1
            
            # Update costs
            layer_cost = self._get_layer_cost(self.current_exit)
            self.cumulative_latency += layer_cost['latency']
            self.cumulative_compute += layer_cost['compute']
            
            # Check if we've reached the last exit
            if self.current_exit >= self.num_exits:
                # Forced exit at final layer
                reward, info = self._compute_reward_and_exit()
                self.done = True
                next_state = self._get_state()
                return next_state, reward, self.done, info
            
            # Check if we've exceeded budget
            if self._exceeds_budget():
                # Penalty for exceeding budget
                reward = -2.0
                self.done = True
                info = {
                    'exit_layer': self.current_exit,
                    'correct': False,
                    'latency': self.cumulative_latency,
                    'compute': self.cumulative_compute,
                    'budget_exceeded': True
                }
                next_state = self._get_state()
                return next_state, reward, self.done, info
            
            # Continue to next layer
            next_state = self._get_state()
            reward = -0.01  # Small penalty for continuing (encourages efficiency)
            info = {'continue': True}
            
            return next_state, reward, self.done, info
    
    def _get_state(self):
        """
        Construct state observation.
        
        Returns:
            state: [confidence, layer_idx, latency, compute_budget]
        """
        if self.current_exit >= len(self.features_cache):
            # Final state (shouldn't happen normally)
            return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
        
        # Get confidence from current exit
        features = self.features_cache[self.current_exit]
        with torch.no_grad():
            confidence = self.model.exit_heads.exits[self.current_exit].get_confidence(
                features.unsqueeze(0)
            ).item()
        
        # Normalize layer index
        layer_idx = self.current_exit / max(self.num_exits - 1, 1)
        
        # Normalize latency
        max_latency = self.device_config['constraints']['max_latency_ms']
        latency_norm = min(self.cumulative_latency / max_latency, 1.0)
        
        # Normalize compute budget
        max_compute = self.device_config['constraints']['max_flops']
        compute_budget = max(1.0 - (self.cumulative_compute / max_compute), 0.0)
        
        state = np.array([
            confidence,
            layer_idx,
            latency_norm,
            compute_budget
        ], dtype=np.float32)
        
        return state
    
    def _compute_reward_and_exit(self):
        """
        Compute reward when exiting.
        
        Returns:
            reward: Scalar reward
            info: Dictionary with episode information
        """
        # Get prediction from current exit
        features = self.features_cache[self.current_exit]
        with torch.no_grad():
            logits = self.model.exit_heads.exits[self.current_exit](
                features.unsqueeze(0)
            )
            prediction = torch.argmax(logits, dim=1).item()
        
        # Check if correct
        correct = (prediction == self.current_label.item())
        
        # Compute reward
        alpha = self.reward_config['alpha']
        beta = self.reward_config['beta']
        gamma = self.reward_config['gamma']
        
        accuracy_reward = alpha * (1.0 if correct else 0.0)
        latency_penalty = beta * (self.cumulative_latency / 
                                   self.device_config['constraints']['max_latency_ms'])
        compute_penalty = gamma * (self.cumulative_compute / 
                                    self.device_config['constraints']['max_flops'])
        
        reward = accuracy_reward - latency_penalty - compute_penalty
        
        # Bonus/penalty
        if correct:
            reward += self.reward_config.get('correct_prediction_bonus', 0.0)
        else:
            reward += self.reward_config.get('incorrect_prediction_penalty', 0.0)
        
        info = {
            'exit_layer': self.current_exit,
            'correct': correct,
            'prediction': prediction,
            'ground_truth': self.current_label.item(),
            'latency': self.cumulative_latency,
            'compute': self.cumulative_compute,
            'accuracy_reward': accuracy_reward,
            'latency_penalty': latency_penalty,
            'compute_penalty': compute_penalty,
            'total_reward': reward
        }
        
        return reward, info
    
    def _get_layer_cost(self, exit_index):
        """
        Get latency and compute cost for a layer.
        
        Returns:
            dict with 'latency' and 'compute' keys
        """
        layer_costs = self.device_config.get('layer_costs', {})
        layer_name = f'layer_{exit_index + 1}'
        
        latency = layer_costs.get(layer_name, 1.0)
        compute = 50_000_000 * (exit_index + 1) / self.num_exits  # Approximate
        
        return {'latency': latency, 'compute': compute}
    
    def _exceeds_budget(self):
        """Check if current costs exceed device budget."""
        max_latency = self.device_config['constraints']['max_latency_ms']
        max_compute = self.device_config['constraints']['max_flops']
        
        return (self.cumulative_latency > max_latency or 
                self.cumulative_compute > max_compute)
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Exit: {self.current_exit}/{self.num_exits}")
            print(f"Latency: {self.cumulative_latency:.2f}ms")
            print(f"Compute: {self.cumulative_compute/1e6:.2f}M FLOPs")
