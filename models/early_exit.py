"""
Early Exit Head Architecture

Each exit head is a lightweight classifier that can produce predictions
at intermediate layers of the backbone network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyExitHead(nn.Module):
    """
    Lightweight exit head for intermediate predictions.
    
    Architecture:
        - Adaptive pooling to fixed size
        - Fully connected layers with dropout
        - Output logits for classification
    """
    
    def __init__(self, in_channels, num_classes, hidden_dim=256, dropout=0.3):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for FC layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Feature map from backbone [B, C, H, W]
            
        Returns:
            logits: Class logits [B, num_classes]
        """
        x = self.pool(x)
        logits = self.classifier(x)
        return logits
    
    def get_confidence(self, x):
        """
        Compute prediction confidence using entropy.
        
        Lower entropy = higher confidence
        
        Args:
            x: Feature map from backbone
            
        Returns:
            confidence: Confidence score [0, 1] where 1 = very confident
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        # Normalize entropy to [0, 1]
        # Max entropy for uniform distribution over num_classes
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        # Convert to confidence (1 - entropy)
        confidence = 1.0 - normalized_entropy
        
        return confidence


class MultiExitHead(nn.Module):
    """
    Container for multiple exit heads at different depths.
    """
    
    def __init__(self, exit_configs, num_classes):
        """
        Args:
            exit_configs: List of dicts with 'in_channels' for each exit
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.num_exits = len(exit_configs)
        self.exits = nn.ModuleList([
            EarlyExitHead(
                in_channels=config['in_channels'],
                num_classes=num_classes,
                hidden_dim=config.get('hidden_dim', 256),
                dropout=config.get('dropout', 0.3)
            )
            for config in exit_configs
        ])
        
    def forward(self, features_list):
        """
        Args:
            features_list: List of feature maps from different depths
            
        Returns:
            outputs: List of logits from each exit
        """
        outputs = []
        for i, features in enumerate(features_list):
            if i < len(self.exits):
                logits = self.exits[i](features)
                outputs.append(logits)
        return outputs
    
    def get_confidences(self, features_list):
        """
        Get confidence scores from all exits.
        
        Returns:
            confidences: List of confidence tensors
        """
        confidences = []
        for i, features in enumerate(features_list):
            if i < len(self.exits):
                conf = self.exits[i].get_confidence(features)
                confidences.append(conf)
        return confidences
