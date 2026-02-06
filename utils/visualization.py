"""
Visualization utilities for results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_exit_accuracies(exit_accuracies, save_path=None):
    """
    Plot accuracy of each early exit.
    
    Args:
        exit_accuracies: List of accuracies for each exit
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exits = [f'Exit {i+1}' for i in range(len(exit_accuracies))]
    colors = sns.color_palette('viridis', len(exit_accuracies))
    
    bars = ax.bar(exits, exit_accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Early Exit Accuracies', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved exit accuracies to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_latency_vs_accuracy(latencies, accuracies, labels, save_path=None):
    """
    Plot latency vs accuracy tradeoff (Pareto frontier).
    
    Args:
        latencies: List of latency values
        accuracies: List of accuracy values
        labels: List of labels for each point
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette('husl', len(latencies))
    
    for i, (lat, acc, label) in enumerate(zip(latencies, accuracies, labels)):
        ax.scatter(lat, acc, s=200, c=[colors[i]], alpha=0.7, 
                  edgecolors='black', linewidth=2, label=label)
    
    ax.set_xlabel('Latency (ms)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Latency vs Accuracy Tradeoff', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latency vs accuracy plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_exit_distribution(exit_counts, save_path=None):
    """
    Plot distribution of exit decisions.
    
    Args:
        exit_counts: List of counts for each exit
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exits = [f'Exit {i+1}' for i in range(len(exit_counts))]
    colors = sns.color_palette('coolwarm', len(exit_counts))
    
    wedges, texts, autotexts = ax.pie(
        exit_counts,
        labels=exits,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )
    
    ax.set_title('Exit Layer Distribution', fontsize=15, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved exit distribution to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_reward_curve(rewards, window=100, save_path=None):
    """
    Plot RL training reward curve.
    
    Args:
        rewards: List of episode rewards
        window: Moving average window
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episodes = range(1, len(rewards) + 1)
    
    # Raw rewards
    ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(rewards) + 1), moving_avg, 
               color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax.set_xlabel('Episode', fontsize=13)
    ax.set_ylabel('Reward', fontsize=13)
    ax.set_title('RL Training Progress', fontsize=15, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved reward curve to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_analysis(confidences, correct_flags, save_path=None):
    """
    Plot confidence vs correctness analysis.
    
    Args:
        confidences: List of confidence scores
        correct_flags: List of boolean correctness flags
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    correct_conf = [c for c, flag in zip(confidences, correct_flags) if flag]
    incorrect_conf = [c for c, flag in zip(confidences, correct_flags) if not flag]
    
    ax.hist(correct_conf, bins=30, alpha=0.7, color='green', 
           label='Correct Predictions', edgecolor='black')
    ax.hist(incorrect_conf, bins=30, alpha=0.7, color='red', 
           label='Incorrect Predictions', edgecolor='black')
    
    ax.set_xlabel('Confidence Score', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title('Confidence Distribution by Correctness', fontsize=15, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confidence analysis to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_results_summary(metrics, save_path=None):
    """
    Create a text summary of results.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save summary
    """
    summary = "=" * 60 + "\n"
    summary += "EVALUATION RESULTS SUMMARY\n"
    summary += "=" * 60 + "\n\n"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            summary += f"{key:30s}: {value:10.4f}\n"
        else:
            summary += f"{key:30s}: {value}\n"
    
    summary += "\n" + "=" * 60 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(summary)
        print(f"Saved results summary to {save_path}")
    else:
        print(summary)
