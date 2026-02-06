"""
ResNet-18 Backbone with Early Exit Heads

Modified ResNet-18 that outputs intermediate features for early exits.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from .early_exit import MultiExitHead


class ResNet18EarlyExit(nn.Module):
    """
    ResNet-18 with early exit heads at multiple depths.
    
    Architecture:
        Input (3x32x32)
        ↓
        Conv1 + BN + ReLU + MaxPool
        ↓
        Layer1 (64 channels)  → Exit 1 (optional)
        ↓
        Layer2 (128 channels) → Exit 2 (optional)
        ↓
        Layer3 (256 channels) → Exit 3 (optional)
        ↓
        Layer4 (512 channels) → Final Classifier
    """
    
    def __init__(self, num_classes=100, exit_points=[1, 2, 3], dropout=0.3):
        """
        Args:
            num_classes: Number of output classes
            exit_points: Which ResNet layers to add exits (1-4)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.exit_points = sorted(exit_points)
        
        # Load ResNet-18 backbone (without pretrained weights for CIFAR)
        backbone = resnet18(pretrained=False, num_classes=num_classes)
        
        # Modify first conv for CIFAR (32x32 instead of 224x224)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        
        # ResNet layers
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # Final classifier
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)
        
        # Early exit heads
        exit_configs = []
        channel_dims = {1: 64, 2: 128, 3: 256, 4: 512}
        
        for exit_point in exit_points:
            exit_configs.append({
                'in_channels': channel_dims[exit_point],
                'hidden_dim': 256,
                'dropout': dropout
            })
        
        self.exit_heads = MultiExitHead(exit_configs, num_classes)
        
        # Track which layer corresponds to which exit
        self.exit_layer_map = {i: exit_points[i] for i in range(len(exit_points))}
        
    def forward(self, x, return_features=False):
        """
        Forward pass through backbone.
        
        Args:
            x: Input images [B, 3, 32, 32]
            return_features: If True, return intermediate features
            
        Returns:
            If return_features=False:
                final_logits: Logits from final classifier
            If return_features=True:
                (final_logits, features_list, exit_logits)
        """
        features_list = []
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Layer 1
        x = self.layer1(x)
        if 1 in self.exit_points:
            features_list.append(x)
        
        # Layer 2
        x = self.layer2(x)
        if 2 in self.exit_points:
            features_list.append(x)
        
        # Layer 3
        x = self.layer3(x)
        if 3 in self.exit_points:
            features_list.append(x)
        
        # Layer 4
        x = self.layer4(x)
        if 4 in self.exit_points:
            features_list.append(x)
        
        # Final classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        final_logits = self.fc(x)
        
        if return_features:
            # Get predictions from all exit heads
            exit_logits = self.exit_heads(features_list)
            return final_logits, features_list, exit_logits
        else:
            return final_logits
    
    def forward_until_exit(self, x, exit_index):
        """
        Forward pass until a specific exit point.
        
        Args:
            x: Input images
            exit_index: Which exit to use (0-indexed)
            
        Returns:
            logits: Predictions from the specified exit
            latency: Simulated latency (number of layers processed)
            compute: Simulated compute (FLOPs approximation)
        """
        features_list = []
        layers_processed = 0
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        layers_processed += 1
        
        # Process layers until exit
        target_layer = self.exit_layer_map[exit_index]
        
        if target_layer >= 1:
            x = self.layer1(x)
            layers_processed += 1
            if 1 in self.exit_points:
                features_list.append(x)
        
        if target_layer >= 2:
            x = self.layer2(x)
            layers_processed += 1
            if 2 in self.exit_points:
                features_list.append(x)
        
        if target_layer >= 3:
            x = self.layer3(x)
            layers_processed += 1
            if 3 in self.exit_points:
                features_list.append(x)
        
        if target_layer >= 4:
            x = self.layer4(x)
            layers_processed += 1
            if 4 in self.exit_points:
                features_list.append(x)
        
        # Get prediction from exit head
        logits = self.exit_heads.exits[exit_index](features_list[exit_index])
        
        # Compute metrics
        latency = layers_processed
        compute = layers_processed * 1.0  # Simplified compute metric
        
        return logits, latency, compute
    
    def get_num_exits(self):
        """Return number of available exits."""
        return len(self.exit_points)


def create_model(num_classes=100, exit_points=[1, 2, 3], dropout=0.3):
    """
    Factory function to create ResNet-18 with early exits.
    
    Args:
        num_classes: Number of output classes
        exit_points: Which layers to add exits
        dropout: Dropout probability
        
    Returns:
        model: ResNet18EarlyExit instance
    """
    return ResNet18EarlyExit(
        num_classes=num_classes,
        exit_points=exit_points,
        dropout=dropout
    )
