"""
Utility functions for data loading and preprocessing.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_cifar100_loaders(data_dir='./data', batch_size=128, num_workers=4, 
                         augment=True, val_split=0.1):
    """
    Get CIFAR-100 data loaders.
    
    Args:
        data_dir: Directory to store/load data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to use data augmentation
        val_split: Fraction of training data for validation
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Data augmentation for training
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    ])
    
    # Load datasets
    full_train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training into train/val
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_single_image_loader(dataset, batch_size=1):
    """
    Get loader that returns single images (for RL training).
    
    Args:
        dataset: PyTorch dataset
        batch_size: Should be 1 for RL
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for RL
        pin_memory=True
    )
