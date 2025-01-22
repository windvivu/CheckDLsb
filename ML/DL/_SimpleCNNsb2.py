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

# save model
def savecheckpoint(model, info:dict, filename):
    checkpoint = {
      "model": model,
      "info": info
	}
    torch.save(checkpoint, filename)