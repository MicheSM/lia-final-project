import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from src.models import get_model
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / 'data'
SAVED_MODELS_PATH = Path(__file__).parent.parent.parent / 'saved_models'

# Define a transformation pipeline
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(), # Convert to tensor and normalize to [0,1]
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], # ImageNet means
    std=[0.229, 0.224, 0.225] # ImageNet standard deviations
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Convert to tensor and normalize to [0,1]
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], # ImageNet means
    std=[0.229, 0.224, 0.225] # ImageNet standard deviations
    )
])

# Load dataset (e.g., a folder with class subfolders)
train_data = datasets.ImageFolder(DATA_PATH / 'train', transform=train_transform)
test_data = datasets.ImageFolder(DATA_PATH / 'test', transform=test_transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model_name):
    
    model = get_model(model_name)  
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # pyright: ignore[reportAttributeAccessIssue]

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Reset gradients
            optimizer.zero_grad()
            # 2. Forward pass
            output = model(inputs)
            # 3. Compute loss
            loss = criterion(output, labels)
            # 4. Compute gradients
            loss.backward()
            # 5. Update weights
            optimizer.step()
            
            running_loss += loss.item()

        # Set model to evaluation mode
        model.eval()
        # Disable gradient computation for evaluation
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in test_loader:
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy}%')
            # Set model back to training mode
        model.train()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

    print('Training complete')
    torch.save(model.state_dict(), SAVED_MODELS_PATH / 'v2' / (model_name + '.pth'))

