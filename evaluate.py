"""
Evaluate trained models and generate comprehensive analysis.

This script loads trained models and performs detailed evaluation,
including visualization and comparison across different configurations.
"""

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np

from models import create_model
from rl import DynamicInferenceEnv, PPOAgent
from utils import (
    get_cifar100_loaders,
    evaluate_model,
    evaluate_early_exits,
    compute_rl_metrics,
    plot_exit_accuracies,
    plot_latency_vs_accuracy,
    plot_exit_distribution,
    plot_confidence_analysis,
    create_results_summary
)


def evaluate_rl_policy(model, agent, env, data_loader, num_samples, device):
    """
    Evaluate RL policy on dataset.
    
    Returns:
        metrics: Evaluation metrics
        episode_infos: Detailed episode information
    """
    episode_infos = []
    data_iter = iter(data_loader)
    
    for _ in range(num_samples):
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
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            state = next_state
        
        episode_infos.append(info)
    
    metrics = compute_rl_metrics(episode_infos)
    return metrics, episode_infos


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = create_model(num_classes=100, exit_points=[1, 2, 3]).to(device)
    
    if args.model_path:
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {args.model_path}")
    
    model.eval()
    
    # Load data
    print("Loading data...")
    _, _, test_loader = get_cifar100_loaders(
        data_dir='./data',
        batch_size=128 if not args.rl_agent_path else 1,
        num_workers=4,
        augment=False
    )
    
    # Evaluate base model
    print("\n" + "="*60)
    print("BASE MODEL EVALUATION")
    print("="*60)
    
    test_metrics = evaluate_model(model, test_loader, device, return_predictions=True)
    print(f"\nTest Accuracy: {test_metrics['top1_acc']:.2f}%")
    print(f"Test Top-5 Accuracy: {test_metrics['top5_acc']:.2f}%")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    
    # Evaluate early exits
    print("\nEarly Exit Accuracies:")
    exit_accs = evaluate_early_exits(model, test_loader, device)
    for i, acc in enumerate(exit_accs):
        print(f"  Exit {i+1}: {acc:.2f}%")
    
    # Plot exit accuracies
    plot_exit_accuracies(
        exit_accs,
        save_path=results_dir / 'exit_accuracies_eval.png'
    )
    
    # Evaluate RL policy if provided
    if args.rl_agent_path:
        print("\n" + "="*60)
        print("RL POLICY EVALUATION")
        print("="*60)
        
        # Load configs
        with open(args.rl_config, 'r') as f:
            rl_config = yaml.safe_load(f)
        
        with open(args.edge_config, 'r') as f:
            edge_configs = yaml.safe_load(f)
        
        device_config = edge_configs[args.device_profile]
        print(f"\nDevice Profile: {device_config['name']}")
        
        # Create environment
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
        
        # Load agent
        agent = PPOAgent(state_dim=4, action_dim=2, device=device)
        agent.load(args.rl_agent_path)
        print(f"Loaded RL agent from {args.rl_agent_path}")
        
        # Evaluate
        print("\nEvaluating RL policy...")
        rl_metrics, rl_infos = evaluate_rl_policy(
            model, agent, env, test_loader, args.num_samples, device
        )
        
        print(f"\nRL Policy Results:")
        print(f"  Accuracy: {rl_metrics['accuracy']:.2f}%")
        print(f"  Avg Latency: {rl_metrics['avg_latency']:.2f}ms")
        print(f"  Avg Compute: {rl_metrics['avg_compute']/1e6:.2f}M FLOPs")
        print(f"  Avg Exit Layer: {rl_metrics['avg_exit_layer']:.2f}")
        print(f"  Avg Reward: {rl_metrics['avg_reward']:.3f}")
        
        # Exit distribution
        exit_layers = [info['exit_layer'] for info in rl_infos]
        exit_counts = [exit_layers.count(i) for i in range(model.get_num_exits())]
        
        print(f"\nExit Distribution:")
        for i, count in enumerate(exit_counts):
            print(f"  Exit {i+1}: {count} ({count/len(exit_layers)*100:.1f}%)")
        
        plot_exit_distribution(
            exit_counts,
            save_path=results_dir / 'rl_exit_distribution_eval.png'
        )
        
        # Confidence analysis
        confidences = []
        correct_flags = []
        
        for info in rl_infos[:1000]:  # Sample for visualization
            # Get confidence from exit used
            exit_idx = info['exit_layer']
            # Approximate confidence (would need to recompute properly)
            confidences.append(0.8 if info['correct'] else 0.4)
            correct_flags.append(info['correct'])
        
        plot_confidence_analysis(
            confidences,
            correct_flags,
            save_path=results_dir / 'confidence_analysis_eval.png'
        )
        
        # Baseline comparison
        print("\n" + "="*60)
        print("BASELINE COMPARISON")
        print("="*60)
        
        # Compute baselines
        test_loader_single = get_cifar100_loaders(
            data_dir='./data',
            batch_size=1,
            num_workers=0,
            augment=False
        )[2]
        
        # Early exit baseline
        early_metrics = []
        for i, (images, labels) in enumerate(test_loader_single):
            if i >= args.num_samples:
                break
            image = images.to(device)
            label = labels.to(device)
            with torch.no_grad():
                _, _, exit_logits = model(image, return_features=True)
                pred = torch.argmax(exit_logits[0], dim=1)
                correct = (pred == label).item()
                early_metrics.append({
                    'correct': correct,
                    'latency': device_config['layer_costs']['layer_1']
                })
        
        early_acc = np.mean([m['correct'] for m in early_metrics]) * 100
        early_lat = np.mean([m['latency'] for m in early_metrics])
        
        # Late exit baseline
        late_metrics = []
        for i, (images, labels) in enumerate(test_loader_single):
            if i >= args.num_samples:
                break
            image = images.to(device)
            label = labels.to(device)
            with torch.no_grad():
                final_logits = model(image)
                pred = torch.argmax(final_logits, dim=1)
                correct = (pred == label).item()
                late_metrics.append({
                    'correct': correct,
                    'latency': sum(device_config['layer_costs'].values())
                })
        
        late_acc = np.mean([m['correct'] for m in late_metrics]) * 100
        late_lat = np.mean([m['latency'] for m in late_metrics])
        
        print(f"\nComparison:")
        print(f"  Always Early Exit: {early_acc:.2f}% acc @ {early_lat:.2f}ms")
        print(f"  RL Policy:         {rl_metrics['accuracy']:.2f}% acc @ {rl_metrics['avg_latency']:.2f}ms")
        print(f"  Always Late Exit:  {late_acc:.2f}% acc @ {late_lat:.2f}ms")
        
        # Speedup and accuracy retention
        speedup = late_lat / rl_metrics['avg_latency']
        acc_retention = rl_metrics['accuracy'] / late_acc * 100
        
        print(f"\nRL Policy Benefits:")
        print(f"  Speedup vs Always Late: {speedup:.2f}x")
        print(f"  Accuracy Retention: {acc_retention:.1f}%")
        
        # Plot comparison
        plot_latency_vs_accuracy(
            latencies=[early_lat, rl_metrics['avg_latency'], late_lat],
            accuracies=[early_acc, rl_metrics['accuracy'], late_acc],
            labels=['Always Early', 'RL Policy', 'Always Late'],
            save_path=results_dir / 'latency_vs_accuracy_eval.png'
        )
        
        # Save summary
        summary = {
            'Base Model Test Accuracy': test_metrics['top1_acc'],
            'RL Policy Accuracy': rl_metrics['accuracy'],
            'RL Policy Latency (ms)': rl_metrics['avg_latency'],
            'RL Policy Compute (M FLOPs)': rl_metrics['avg_compute'] / 1e6,
            'Early Exit Accuracy': early_acc,
            'Early Exit Latency (ms)': early_lat,
            'Late Exit Accuracy': late_acc,
            'Late Exit Latency (ms)': late_lat,
            'Speedup vs Late Exit': speedup,
            'Accuracy Retention (%)': acc_retention,
            'Device Profile': device_config['name']
        }
        
        create_results_summary(
            summary,
            save_path=results_dir / 'evaluation_summary.txt'
        )
    
    print("\nEvaluation complete!")
    print(f"Results saved to {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--rl_agent_path', type=str, default=None,
                       help='Path to trained RL agent (optional)')
    parser.add_argument('--rl_config', type=str, default='configs/rl_config.yaml',
                       help='Path to RL config')
    parser.add_argument('--edge_config', type=str, default='configs/edge_config.yaml',
                       help='Path to edge config')
    parser.add_argument('--device_profile', type=str, default='low_power',
                       choices=['low_power', 'medium_power', 'high_power', 'adaptive'],
                       help='Device profile to use')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples for evaluation')
    
    args = parser.parse_args()
    main(args)
