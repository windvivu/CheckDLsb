import torch
import torch.nn as nn
from torchvision import models

class Resnet18sb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.ver = 'res18sb'
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=False)
        
        # Modify first conv layer for grayscale input (1 channel)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify final fully connected layer for your number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

class EfficientNetV2sb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        
        self.ver = 'effv2sb'
        
        # Load EfficientNetV2_S (smallest variant)
        self.model = models.efficientnet_v2_s(pretrained=False)
        
        # Modify first conv layer for grayscale input (1 channel)
        # Keep small kernel for 16x16 input
        self.model.features[0][0] = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Modify classifier for your classes
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# save model
def savecheckpoint(model, info:dict, filename):
    checkpoint = {
      "model": model,
      "accu": info["bestacu"],
      "info": info
	}
    torch.save(checkpoint, filename)