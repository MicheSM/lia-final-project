import torch
from torchvision import models, transforms, datasets
from PIL import Image
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
num_classes = 38  # set this to your number of classes

# Define the same transform as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Recreate and load the model
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(str(BASE_DIR / 'app' / 'machine_learning' / 'resnet18_transfer.pth'), map_location='cpu'))
model.eval()

# Load test data
test_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'test'), transform=transform)
class_names = test_folder.classes
test_loader = torch.utils.data.DataLoader(test_folder, batch_size=32, shuffle=False)

# Full evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

cm = confusion_matrix(all_labels, all_preds)

# Read existing results
results_path = BASE_DIR / 'app' / 'data_analysis' / 'results.json'
with open(results_path, 'r') as f:
    results = json.load(f)

# Add/update resnet18 results
results['resnet18'] = {
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist()
}

# Write back to JSON
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"ResNet18 test accuracy: {accuracy:.4f} and results saved to {results_path}")