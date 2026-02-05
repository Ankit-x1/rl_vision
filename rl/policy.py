"""
Policy and Value Networks for PPO

These networks learn to decide when to exit during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    
    Maps state → action distribution (CONTINUE or EXIT)
    """
    
    def __init__(self, state_dim=4, hidden_dims=[256, 128], action_dim=2):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
            action_dim: Number of actions (2: CONTINUE, EXIT)
        """
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Args:
            state: State tensor [B, state_dim]
            
        Returns:
            action_logits: Logits for each action [B, action_dim]
        """
        return self.network(state)
    
    def get_action_probs(self, state):
        """
        Get action probabilities.
        
        Returns:
            probs: Action probabilities [B, action_dim]
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs
    
    def sample_action(self, state):
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
        """
        logits = self.forward(state)
        dist = Categorical(logits=logits)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Value network that estimates state value.
    
    Maps state → expected return
    """
    
    def __init__(self, state_dim=4, hidden_dims=[256, 128]):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        in_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state):
        """
        Args:
            state: State tensor [B, state_dim]
            
        Returns:
            value: Estimated value [B, 1]
        """
        return self.network(state)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    """
    
    def __init__(self, state_dim=4, action_dim=2, 
                 policy_hidden=[256, 128], value_hidden=[256, 128]):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            policy_hidden: Hidden dims for policy network
            value_hidden: Hidden dims for value network
        """
        super().__init__()
        
        self.policy = PolicyNetwork(state_dim, policy_hidden, action_dim)
        self.value = ValueNetwork(state_dim, value_hidden)
        
    def forward(self, state):
        """
        Forward pass through both networks.
        
        Returns:
            action_logits: Policy logits
            value: State value estimate
        """
        action_logits = self.policy(state)
        value = self.value(state)
        return action_logits, value
    
    def get_action_and_value(self, state):
        """
        Sample action and get value estimate.
        
        Returns:
            action: Sampled action
            log_prob: Log probability
            entropy: Policy entropy
            value: Value estimate
        """
        action, log_prob, entropy = self.policy.sample_action(state)
        value = self.value(state)
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions taken in states.
        
        Used during PPO update.
        
        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Policy entropy
        """
        action_logits = self.policy(states)
        dist = Categorical(logits=action_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value(states)
        
        return log_probs, values.squeeze(-1), entropy
