import torch
import torch.nn as nn

import torch.nn as nn

class SimpleCNNsb(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNNsb, self).__init__()

        self.ver = 'sb1'
        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            # No Dropout2d here - preserve early features
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        
        # Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 8x8 -> 8x8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # No Dropout2d here - preserve mid-level features
            nn.MaxPool2d(2)  # 8x8 -> 4x4
        )
        
        # Third conv block
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 4x4 -> 4x4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.3),  # Add Dropout2d here for later features
            nn.MaxPool2d(2)  # 4x4 -> 2x2
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # 3 classes output
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x


class LightweightCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(LightweightCNN, self).__init__()

        self.ver = 'sb2'

        # Simplified MBConv-inspired block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU6(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
        
        # Depthwise separable convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 32, 1),
            nn.ReLU6(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)  # 3 classes output
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x

import torch
import torch.nn as nn

class HybridCNN(nn.Module):
    """
    Hybrid CNN combining standard and depthwise separable convolutions for 16x16 input.
    
    Architecture:
    - Standard conv block (better feature learning at early stage)
    - Depthwise separable conv block (efficient computation)
    - Linear classifier
    
    Input: (batch_size, 1, 16, 16)
    Output: (batch_size, 3)
    """
    
    def __init__(self, num_classes=3):
        super(HybridCNN, self).__init__()
        
        self.ver = 'sb3'

        # Standard convolution block
        # Input: 1x16x16 -> Output: 16x8x8
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )
        
        # Depthwise separable convolution block
        # Input: 16x8x8 -> Output: 32x4x4
        self.conv2 = nn.Sequential(
            # Depthwise conv
            nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
            # Pointwise conv
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        
        # Classifier
        # Input: 32x4x4 (flattened) -> Output: 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 16, 16)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3)
        """
        # Extract features using standard conv
        x = self.conv1(x)  # 1x16x16 -> 16x8x8
        
        # Extract features using depthwise separable conv
        x = self.conv2(x)  # 16x8x8 -> 32x4x4
        
        # Classification
        x = self.classifier(x)  # 32x4x4 -> 3
        
        return x
    
    def get_num_params(self):
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnhancedHybridCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(EnhancedHybridCNN, self).__init__()
        
        self.ver = 'sb4'

        # Standard conv blocks for better feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )
        
        # First Depthwise block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)  # 8x8 -> 4x4
        )
        
        # Second Depthwise block with squeeze-excitation
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # SE block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Apply SE attention
        se = self.se(x)
        x = x * se.view(-1, 128, 1, 1)
        
        x = self.classifier(x)
        return x


# save model
def savecheckpoint(model, info:dict, filename):
    checkpoint = {
      "model": model,
      "info": info
	}
    torch.save(checkpoint, filename)