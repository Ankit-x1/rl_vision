"""
PPO Agent for Dynamic Inference

Implements Proximal Policy Optimization for learning exit policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .policy import ActorCritic


class PPOAgent:
    """
    PPO agent that learns when to exit during inference.
    """
    
    def __init__(self, state_dim=4, action_dim=2, config=None, device='cpu'):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            config: PPO configuration dict
            device: torch device
        """
        self.device = device
        self.config = config or {}
        
        # Hyperparameters
        self.lr = self.config.get('learning_rate', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.epochs_per_update = self.config.get('epochs_per_update', 10)
        self.batch_size = self.config.get('batch_size', 64)
        self.ent_coef = self.config.get('ent_coef', 0.01)
        self.vf_coef = self.config.get('vf_coef', 0.5)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        
        # Networks
        policy_hidden = self.config.get('policy_hidden', [256, 128])
        value_hidden = self.config.get('value_hidden', [256, 128])
        
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            policy_hidden=policy_hidden,
            value_hidden=value_hidden
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=self.lr
        )
        
        # Storage for rollout data
        self.reset_rollout()
        
    def reset_rollout(self):
        """Reset rollout buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action given state.
        
        Args:
            state: State observation
            deterministic: If True, select argmax action
            
        Returns:
            action: Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_logits, _ = self.actor_critic(state_tensor)
                action = torch.argmax(action_logits, dim=-1)
            else:
                action, log_prob, _, value = self.actor_critic.get_action_and_value(state_tensor)
                
                # Store for training
                self.states.append(state)
                self.actions.append(action.item())
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
        
        return action.item()
    
    def store_reward_and_done(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0.0):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            next_value: Value of next state (0 if terminal)
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] + 
                    self.gamma * values[t + 1] * (1 - self.dones[t]) - 
                    values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self):
        """
        Update policy using PPO.
        
        Returns:
            metrics: Dictionary of training metrics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.epochs_per_update):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.vf_coef * value_loss + 
                       self.ent_coef * entropy_loss)
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Reset rollout buffer
        self.reset_rollout()
        
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return metrics
    
    def save(self, path):
        """Save agent checkpoint."""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.config = checkpoint.get('config', self.config)


# Import for convenience
import torch.nn.functional as F
