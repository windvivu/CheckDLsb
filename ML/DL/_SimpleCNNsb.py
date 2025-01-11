import torch
import torch.nn as nn

class SimpleCNNsb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.ver = 'sb0'

        # First conv block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
        )

		# Second conv block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
        )

		# Third conv block
        self.conv3 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Dropout2d(0.4),
			# nn.MaxPool2d(2),
		)

		# Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )


    def forward(self,x):
       x = self.conv1(x)
       x = self.conv2(x)
       x = self.conv3(x)
       x = x.view(x.size(0), -1)  # Flatten
       x = self.fc(x)
       return x

# save model
def savecheckpoint(model, info:dict, filename):
    checkpoint = {
      "model": model,
      "info": info
	}
    torch.save(checkpoint, filename)