import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
import json
from pathlib import Path
from progetto.app.machine_learning.training.model_utils import get_model

BASE_DIR = Path(__file__).resolve().parent.parent.parent
num_classes = 38
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'train'), transform=transform)
test_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'test'), transform=transform)
class_names = train_folder.classes

train_loader = DataLoader(train_folder, batch_size=32, shuffle=True)
test_loader = DataLoader(test_folder, batch_size=32, shuffle=False)

results = {}

for model_name in ["mobilenet_v2", "customcnn"]:
    print(f"Training {model_name}...")
    model = get_model(model_name, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    # Training loop (shortened for brevity)
    for epoch in range(5):  # Use more epochs for real training
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds).tolist()
    results[model_name] = {"accuracy": acc, "confusion_matrix": cm}
    # Optionally save model
    torch.save(model.state_dict(), str(BASE_DIR / f'app/machine_learning/{model_name}.pth'))

# Save results to JSON
with open(BASE_DIR / 'app' / 'data_analysis' / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)