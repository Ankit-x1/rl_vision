"""
Evaluation metrics for vision and RL.
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(outputs, targets, topk=(1, 5)):
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model predictions [B, num_classes]
        targets: Ground truth labels [B]
        topk: Tuple of k values
        
    Returns:
        List of top-k accuracies
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    
    return res


def evaluate_model(model, data_loader, device, return_predictions=False):
    """
    Evaluate model on dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
        device: torch device
        return_predictions: If True, return predictions and labels
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    
    all_predictions = []
    all_labels = []
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Compute accuracy
            acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            
            if return_predictions:
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    metrics = {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }
    
    if return_predictions:
        metrics['predictions'] = np.array(all_predictions)
        metrics['labels'] = np.array(all_labels)
    
    return metrics


def evaluate_early_exits(model, data_loader, device):
    """
    Evaluate accuracy of each early exit head.
    
    Args:
        model: Model with early exits
        data_loader: DataLoader
        device: torch device
        
    Returns:
        exit_accuracies: List of accuracies for each exit
    """
    model.eval()
    
    num_exits = model.get_num_exits()
    exit_correct = [0] * num_exits
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get all exit predictions
            _, features_list, exit_logits = model(images, return_features=True)
            
            # Check each exit
            for i, logits in enumerate(exit_logits):
                predictions = torch.argmax(logits, dim=1)
                exit_correct[i] += (predictions == labels).sum().item()
            
            total += labels.size(0)
    
    exit_accuracies = [correct / total * 100 for correct in exit_correct]
    
    return exit_accuracies


def compute_rl_metrics(episode_infos):
    """
    Compute RL training metrics from episode info.
    
    Args:
        episode_infos: List of episode info dicts
        
    Returns:
        metrics: Aggregated metrics
    """
    if not episode_infos:
        return {}
    
    accuracies = [info['correct'] for info in episode_infos]
    latencies = [info['latency'] for info in episode_infos]
    computes = [info['compute'] for info in episode_infos]
    exit_layers = [info['exit_layer'] for info in episode_infos]
    rewards = [info['total_reward'] for info in episode_infos]
    
    metrics = {
        'accuracy': np.mean(accuracies) * 100,
        'avg_latency': np.mean(latencies),
        'avg_compute': np.mean(computes),
        'avg_exit_layer': np.mean(exit_layers),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards)
    }
    
    return metrics
