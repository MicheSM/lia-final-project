import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np
import random
from torch.utils.data import Subset

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Reduce dataset size by a factor of 10
def get_subset(dataset, fraction=0.1, seed=42):
    random.seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subset_size = max(1, int(len(indices) * fraction))
    subset_indices = indices[:subset_size]
    return Subset(dataset, subset_indices)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset (e.g., a folder with class subfolders)
train_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'train'), transform=transform)
test_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'test'), transform=transform)
class_names = train_folder.classes

#reduce dataset size by a factor of 10
train_data = get_subset(train_folder, fraction=0.1)
test_data = get_subset(test_folder, fraction=0.1)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    
# Modify final layer
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

print('Training complete')
torch.save(model.state_dict(), str(BASE_DIR / 'app' / 'machine_learning' / 'resnet18_transfer.pth'))

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

# Confusion matrix
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]
        
    return class_names[predicted.item()], probability[int(predicted.item())].item()

# Example usage
#image_path = 'image.jpg'
#class_names = train_data.classes
#class_name, confidence = predict_image(image_path, model, transform, class_names)
#print(f'Predicted: {class_name} with confidence {confidence:.2f}')

