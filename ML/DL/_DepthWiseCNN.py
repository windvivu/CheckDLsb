import torch
import torch.nn as nn

class DepthWiseCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(DepthWiseCNN, self).__init__()

        self.ver = 'dsb'
        
        # Initial feature extraction with dilated convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32)
        )
        
        # Dense feature extraction
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        
        # Residual connection
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64)
        )
        
        # Channel attention
        self.ca = ChannelAttention(64)
        
        # Spatial attention
        self.sa = SpatialAttention()
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # Preserve more spatial information
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # Residual connection
        x3 = x3 + x2
        
        # Apply attention
        x3 = self.ca(x3) * x3
        x3 = self.sa(x3) * x3
        
        out = self.classifier(x3)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out)
        return out.view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


# save model
def savecheckpoint(model, info:dict, filename):
    checkpoint = {
      "model": model,
      "info": info
	}
    torch.save(checkpoint, filename)