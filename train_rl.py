"""
Train RL policy for dynamic inference.

This script trains a PPO agent to learn when to exit during inference,
optimizing for accuracy, latency, and compute constraints.
"""

import argparse
import yaml
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np

from models import create_model
from rl import DynamicInferenceEnv, PPOAgent
from utils import (
    get_cifar100_loaders,
    compute_rl_metrics,
    plot_reward_curve,
    plot_exit_distribution,
    plot_latency_vs_accuracy,
    create_results_summary
)


def collect_episodes(env, agent, data_loader, num_episodes, device):
    """
    Collect episodes for RL training.
    
    Args:
        env: RL environment
        agent: PPO agent
        data_loader: DataLoader for images
        num_episodes: Number of episodes to collect
        device: torch device
        
    Returns:
        episode_infos: List of episode information dicts
    """
    episode_infos = []
    data_iter = iter(data_loader)
    
    for _ in range(num_episodes):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
        
        # Reset environment with new image
        image = images[0:1].to(device)
        label = labels[0:1].to(device)
        
        state = env.reset(image, label)
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store reward and done
            agent.store_reward_and_done(reward, done)
            
            state = next_state
        
        episode_infos.append(info)
    
    return episode_infos


def evaluate_policy(env, agent, data_loader, num_episodes, device):
    """
    Evaluate RL policy (deterministic).
    
    Returns:
        metrics: Evaluation metrics
        episode_infos: Episode information
    """
    episode_infos = []
    data_iter = iter(data_loader)
    
    for _ in range(num_episodes):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
        
        image = images[0:1].to(device)
        label = labels[0:1].to(device)
        
        state = env.reset(image, label)
        done = False
        
        while not done:
            # Deterministic action
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        episode_infos.append(info)
    
    metrics = compute_rl_metrics(episode_infos)
    return metrics, episode_infos


def main(args):
    # Load configs
    with open(args.config, 'r') as f:
        rl_config = yaml.safe_load(f)
    
    with open(args.edge_config, 'r') as f:
        edge_configs = yaml.safe_load(f)
    
    # Get device profile
    device_config = edge_configs[args.device_profile]
    print(f"Using device profile: {device_config['name']}")
    print(f"  Max latency: {device_config['constraints']['max_latency_ms']}ms")
    print(f"  Max FLOPs: {device_config['constraints']['max_flops']/1e6:.1f}M")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(rl_config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load pretrained vision model
    print("Loading pretrained vision model...")
    model = create_model(num_classes=100, exit_points=[1, 2, 3]).to(device)
    
    if rl_config['checkpoint']['load_pretrained_vision']:
        vision_checkpoint = torch.load(
            rl_config['checkpoint']['vision_checkpoint'],
            map_location=device
        )
        model.load_state_dict(vision_checkpoint['model_state_dict'])
        print(f"Loaded vision model from {rl_config['checkpoint']['vision_checkpoint']}")
    
    model.eval()  # Keep vision model frozen
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir='./data',
        batch_size=1,  # Single image for RL
        num_workers=0,
        augment=False
    )
    
    # Create RL environment
    print("Creating RL environment...")
    reward_config = {
        **rl_config['reward'],
        **device_config.get('reward_weights', {})
    }
    
    env = DynamicInferenceEnv(
        model=model,
        device_config=device_config,
        reward_config=reward_config,
        device=device
    )
    
    # Create PPO agent
    print("Creating PPO agent...")
    agent = PPOAgent(
        state_dim=4,
        action_dim=2,
        config=rl_config['rl']['ppo'],
        device=device
    )
    
    # Training loop
    print("\nStarting RL training...")
    total_timesteps = rl_config['rl']['total_timesteps']
    n_steps = rl_config['rl']['ppo']['n_steps']
    eval_frequency = rl_config['training']['eval_frequency']
    
    all_episode_rewards = []
    timestep = 0
    
    pbar = tqdm(total=total_timesteps, desc='RL Training')
    
    while timestep < total_timesteps:
        # Collect episodes
        num_episodes = n_steps // 4  # Approximate episodes per update
        episode_infos = collect_episodes(
            env, agent, train_loader, num_episodes, device
        )
        
        # Update agent
        update_metrics = agent.update()
        
        # Track rewards
        episode_rewards = [info['total_reward'] for info in episode_infos]
        all_episode_rewards.extend(episode_rewards)
        
        timestep += num_episodes * 4  # Approximate timesteps
        pbar.update(num_episodes * 4)
        
        # Log metrics
        if len(update_metrics) > 0:
            pbar.set_postfix({
                'avg_reward': f'{np.mean(episode_rewards):.3f}',
                'policy_loss': f'{update_metrics["policy_loss"]:.4f}',
                'value_loss': f'{update_metrics["value_loss"]:.4f}'
            })
        
        # Evaluate
        if timestep % eval_frequency == 0:
            print(f"\n\nEvaluation at timestep {timestep}")
            eval_metrics, eval_infos = evaluate_policy(
                env, agent, val_loader, 100, device
            )
            
            print(f"  Accuracy: {eval_metrics['accuracy']:.2f}%")
            print(f"  Avg Latency: {eval_metrics['avg_latency']:.2f}ms")
            print(f"  Avg Exit Layer: {eval_metrics['avg_exit_layer']:.2f}")
            print(f"  Avg Reward: {eval_metrics['avg_reward']:.3f}")
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'rl_agent_{timestep}.pth'
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}\n")
    
    pbar.close()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_metrics, final_infos = evaluate_policy(
        env, agent, test_loader, 1000, device
    )
    
    print(f"\nTest Set Results:")
    print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
    print(f"  Avg Latency: {final_metrics['avg_latency']:.2f}ms")
    print(f"  Avg Compute: {final_metrics['avg_compute']/1e6:.2f}M FLOPs")
    print(f"  Avg Exit Layer: {final_metrics['avg_exit_layer']:.2f}")
    print(f"  Avg Reward: {final_metrics['avg_reward']:.3f}")
    
    # Analyze exit distribution
    exit_layers = [info['exit_layer'] for info in final_infos]
    exit_counts = [exit_layers.count(i) for i in range(model.get_num_exits())]
    
    print(f"\nExit Distribution:")
    for i, count in enumerate(exit_counts):
        print(f"  Exit {i+1}: {count} ({count/len(exit_layers)*100:.1f}%)")
    
    # Save final model
    final_path = checkpoint_dir / 'rl_agent_final.pth'
    agent.save(final_path)
    print(f"\nSaved final model to {final_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_reward_curve(
        all_episode_rewards,
        window=100,
        save_path=results_dir / 'reward_curve.png'
    )
    
    plot_exit_distribution(
        exit_counts,
        save_path=results_dir / 'exit_distribution.png'
    )
    
    # Compare with baselines
    print("\nComparing with baselines...")
    
    # Always exit at layer 1 (fastest)
    early_exit_metrics = []
    for images, labels in test_loader:
        if len(early_exit_metrics) >= 100:
            break
        image = images[0:1].to(device)
        label = labels[0:1].to(device)
        with torch.no_grad():
            _, features, exit_logits = model(image, return_features=True)
            pred = torch.argmax(exit_logits[0], dim=1)
            correct = (pred == label).item()
            early_exit_metrics.append({
                'correct': correct,
                'latency': device_config['layer_costs']['layer_1']
            })
    
    early_acc = np.mean([m['correct'] for m in early_exit_metrics]) * 100
    early_lat = np.mean([m['latency'] for m in early_exit_metrics])
    
    # Always use final layer (most accurate)
    late_exit_metrics = []
    for images, labels in test_loader:
        if len(late_exit_metrics) >= 100:
            break
        image = images[0:1].to(device)
        label = labels[0:1].to(device)
        with torch.no_grad():
            final_logits = model(image)
            pred = torch.argmax(final_logits, dim=1)
            correct = (pred == label).item()
            late_exit_metrics.append({
                'correct': correct,
                'latency': sum(device_config['layer_costs'].values())
            })
    
    late_acc = np.mean([m['correct'] for m in late_exit_metrics]) * 100
    late_lat = np.mean([m['latency'] for m in late_exit_metrics])
    
    print(f"\nBaseline Comparison:")
    print(f"  Always Early Exit: {early_acc:.2f}% acc, {early_lat:.2f}ms latency")
    print(f"  Always Late Exit: {late_acc:.2f}% acc, {late_lat:.2f}ms latency")
    print(f"  RL Policy: {final_metrics['accuracy']:.2f}% acc, {final_metrics['avg_latency']:.2f}ms latency")
    
    # Plot comparison
    plot_latency_vs_accuracy(
        latencies=[early_lat, final_metrics['avg_latency'], late_lat],
        accuracies=[early_acc, final_metrics['accuracy'], late_acc],
        labels=['Always Early', 'RL Policy', 'Always Late'],
        save_path=results_dir / 'latency_vs_accuracy.png'
    )
    
    # Save summary
    summary_metrics = {
        'RL Policy Accuracy': final_metrics['accuracy'],
        'RL Policy Latency': final_metrics['avg_latency'],
        'RL Policy Compute': final_metrics['avg_compute'],
        'Early Exit Accuracy': early_acc,
        'Early Exit Latency': early_lat,
        'Late Exit Accuracy': late_acc,
        'Late Exit Latency': late_lat,
        'Device Profile': device_config['name']
    }
    
    create_results_summary(
        summary_metrics,
        save_path=results_dir / 'rl_results_summary.txt'
    )
    
    print("\nRL training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL policy')
    parser.add_argument('--config', type=str, default='configs/rl_config.yaml',
                       help='Path to RL config file')
    parser.add_argument('--edge_config', type=str, default='configs/edge_config.yaml',
                       help='Path to edge config file')
    parser.add_argument('--device_profile', type=str, default='low_power',
                       choices=['low_power', 'medium_power', 'high_power', 'adaptive'],
                       help='Device profile to use')
    
    args = parser.parse_args()
    main(args)
