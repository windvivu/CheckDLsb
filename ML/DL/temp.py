# Improvement Plan
#   Add regularization layers
#   Add more complexity to feature extraction
#   Add batch normalization
#   Adjust learning rate
import torch.nn as nn
import torch

class SimpleCNNsb(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2),
            
            # Third block - New
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.4),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Adjust learning rate and optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                           lr=0.0005,  # Increased learning rate
                           weight_decay=1e-4)  # Increased regularization

# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.1,
    patience=5,
    verbose=True
)

# Changes:
#   Added third conv block
#   Increased feature maps gradually
#   Added BatchNorm layers
#   Progressive dropout rates
#   Adjusted learning rate
#   Added scheduler

# Add scheduler after optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Look for maximum accuracy
    factor=0.1,        # Reduce LR by 10x when plateauing
    patience=5,        # Wait 5 epochs before reducing LR
    verbose=True       # Print when LR changes
)

# 2. Add scheduler step in training loop
for epoch in range(num_epochs):
    # ...existing training code...
    
    # Calculate accuracy
    accu = (torch.tensor(all_predictions) == torch.tensor(all_labels)).sum().item()/len(all_labels)
    print('Accuracy:', accu)
    
    # Update learning rate based on accuracy
    scheduler.step(accu)
    
    if accu > bestaccu:
        bestaccu = accu