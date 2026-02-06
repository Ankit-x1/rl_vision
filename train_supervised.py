"""
Train supervised base model with early exits.

This script trains the ResNet-18 backbone with all early exit heads
using standard cross-entropy loss.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models import create_model
from utils import (
    get_cifar100_loaders,
    AverageMeter,
    compute_accuracy,
    evaluate_model,
    evaluate_early_exits,
    plot_training_curves,
    plot_exit_accuracies
)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        final_logits, features_list, exit_logits = model(images, return_features=True)
        
        # Compute loss for final classifier
        loss = criterion(final_logits, labels)
        
        # Add loss from early exits (weighted)
        exit_weight = 0.3  # Weight for early exit losses
        for exit_logit in exit_logits:
            loss += exit_weight * criterion(exit_logit, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy (using final classifier)
        acc1, acc5 = compute_accuracy(final_logits, labels, topk=(1, 5))
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}%',
            'top5': f'{top5.avg:.2f}%'
        })
    
    return losses.avg, top1.avg, top5.avg


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model = create_model(
        num_classes=config['model']['num_classes'],
        exit_points=config['model']['early_exits']['exit_points'],
        dropout=config['model']['backbone']['dropout']
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        augment=config['training']['augmentation']['random_crop']
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['lr_scheduler']['min_lr']
    )
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc1:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['top1_acc']:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Store for plotting
        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        train_accs.append(train_acc1)
        val_accs.append(val_metrics['top1_acc'])
        
        # Save checkpoint
        if epoch % config['checkpoint']['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_acc': train_acc1,
                'val_acc': val_metrics['top1_acc']
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            best_path = checkpoint_dir / 'supervised_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': best_val_acc
            }, best_path)
            print(f"New best model! Val Acc: {best_val_acc:.2f}%")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Test set evaluation
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_metrics['top1_acc']:.2f}%")
    print(f"Test Top-5 Accuracy: {test_metrics['top5_acc']:.2f}%")
    
    # Evaluate early exits
    print("\nEarly Exit Accuracies:")
    exit_accs = evaluate_early_exits(model, test_loader, device)
    for i, acc in enumerate(exit_accs):
        print(f"  Exit {i+1}: {acc:.2f}%")
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        save_path=results_dir / 'supervised_training_curves.png'
    )
    
    plot_exit_accuracies(
        exit_accs,
        save_path=results_dir / 'exit_accuracies.png'
    )
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_metrics['top1_acc']:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train supervised base model')
    parser.add_argument('--config', type=str, default='configs/model_config.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    main(args)
