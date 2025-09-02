import torch
from torchvision import models, transforms, datasets
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
num_classes = 38 # set this to your number of classes

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
model.load_state_dict(torch.load(str(BASE_DIR / 'app' / 'machine_learning' / 'resnet18_transfer.pth')))
model.eval()

# Example prediction
def predict_image(image_path, model, transform, class_names):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0]
    idx = int(predicted.item())
    return class_names[idx], float(probability[idx])

# Usage
test_folder = datasets.ImageFolder(str(BASE_DIR / 'dataset' / 'test'), transform=transform)
class_names = test_folder.classes
# class_names = [...] # list of class names in the same order as during training
# image_path = 'path/to/image.jpg'
image_path = BASE_DIR / 'dataset' / 'test' / 'Apple___Apple_scab' / '0a769a71-052a-4f19-a4d8-b0f0cb75541c___FREC_Scab 3165.JPG'
print(predict_image(image_path, model, transform, class_names))