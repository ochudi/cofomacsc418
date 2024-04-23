import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define a simple CNN model
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

# Define the classes
classes = ['person', 'chair', 'table']
num_classes = len(classes)

# Initialize the model
model = MyModel(num_classes)

# Optionally, train the model
# Here you would typically load your dataset, define a loss function, and train the model
# For simplicity, I'm skipping those steps and just showing the saving of the model

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
