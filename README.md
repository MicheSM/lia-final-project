# TO DO
- organize folders (flask_app, data_analysis, machine_learning)
- review/change EDA.ipynb(RGB histogram) -> de-vibecode-ficate it + mean rgb compute with larger sample and save to json file
- start machine learning

app/
|- data/
    |- images/
    |- classes.json
|- EDA.ipynb
|- __init__.py
|- data_analysis.py
|- routes.py
|-
.dockerignore
.gitignore
Dockerfile
README.Docker.md
compose.yaml
requirements.txt

app/
|- data/
|- machine_learning/
|- flask_app/
|- data_analysis/
.dockerignore
.gitignore
Dockerfile
README.Docker.md
compose.yaml
requirements.txt
-----------------------------
• Study the data (data preparation)
• Select a model or learning algorithm
• Train on the available data and assess performance
• Apply the model to make predictions on new cases
----------------------------------
Scikit-learn implements a consistent API across all models:
• Estimators: objects that learn from data (e.g., LinearRegression,
RandomForestClassifier)
• Transformers: estimators that can transform data (e.g., StandardScaler,
PCA)
• Predictors: estimators that implement predict() and/or predict_proba()
Common methods:
• fit(): learn from data
• transform(): apply transformation
• predict(): make predictions
• score(): evaluate model 
--------------------------------------------------------
Scikit-learn comes with several built-in datasets for practice:
65
Loading a Dataset
# Load the iris dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data # Features
y = iris.target # Target variable
# Real-world datasets can be loaded from files
import pandas as pd
data = pd.read_csv('your_dataset.csv')
X = data.drop('target_column', axis=1)
y = data['target_column']
-----------------------------------------------------
multiclass classification: ovo or ovr
-------------------------------------------------
k-nearest neighbours in scikit-learn
---------------------------------------------------
k-means in scikit-learn
--------------------------------------------------
mlp for classification, pytorch and tensors, training loop, data loaders, model evaluation, saving and loading models, transfer learning with pytorch, torchvision, deploying model with TorchScript
------------------------------------------------------
Solution minst classification 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 1. Data preparation
transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)
# 2. Define the model
class MNISTClassifier(nn.Module):
def __init__(self):
super(MNISTClassifier, self).__init__()
self.conv1 = nn.Conv2d(1, 32, 3, 1)
self.conv2 = nn.Conv2d(32, 64, 3, 1)
self.dropout1 = nn.Dropout2d(0.25)
self.dropout2 = nn.Dropout2d(0.5)
self.fc1 = nn.Linear(9216, 128)
self.fc2 = nn.Linear(128, 10)
def forward(self, x):
x = self.conv1(x)
x = nn.functional.relu(x)
x = self.conv2(x)
x = nn.functional.relu(x)
x = nn.functional.max_pool2d(x, 2)
x = self.dropout1(x)
x = torch.flatten(x, 1)
x = self.fc1(x)
x = nn.functional.relu(x)
x = self.dropout2(x)
x = self.fc2(x)
return nn.functional.log_softmax(x, dim=1)
--------------------------------------------------
PyTorch provides tools for image processing through torchvision.transforms:
from torchvision import transforms
# Define a transformation pipeline
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(), # Convert to tensor and normalize to [0,1]
transforms.Normalize(
mean=[0.485, 0.456, 0.406], # ImageNet means
std=[0.229, 0.224, 0.225] # ImageNet standard deviations
 )
])
# Apply transformations to an image
from PIL import Image
img = Image.open('image.jpg')
img_tensor = transform(img) # Shape: [3, 224, 224]
----------------------------------------------------------
Complete CNN architecture
Typically CNNs are are composed of a cascade of:
• Convolutional-ReLu layers: perform convolution over the input and non-linear transform over
the convolution output
• Spatial pooling layers: output a summary statistics of local input
• Fully-connected ReLu layers (one or several): perform combination of final convolutional
outputs 
Layers of a CNN have neurons arranged in 3 dimensions (width, Input
height, depth). Each layer of a ConvNet transforms one volume
of activations to another through a differentiable function.
A Softmax classifier is used to perform final classification CNNs
are trained by supervised learning: i.e. the convolutional filters
are trained by back‐propagating the classification error
------------------------------------------------------------
Feature map size
Three hyperparameters control the size of the feature map i.e. the convolved feature is controlled by
three parameters that we need to decide before the convolution step is performed:

Output size O
Input size I
Filter size K
Zero padding P
Stride S
Output has size
--------------------------------------------------------------
Convolution with RGB images
Tensor Size The size of the output is proportional to the number of feature maps and depends on the filter
size, padding and stride
ReLU Rectified Linear Unit- is applied after every convolution operation
------------------------------------------------------------------
CNN Pooling layers
Pooling layer effects
----------------------------------------------------------------------
In PyTorch
Here is a simple CNN implementation:
import torch.nn as nn
# Define a simple CNN
class SimpleCNN(nn.Module):
def __init__(self):
super(SimpleCNN, self).__init__()
# Conv layer: (in_channels, out_channels, kernel_size)
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
# Pooling layer
self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
# ReLU activation
self.relu = nn.ReLU()
# Fully connected layer
self.fc = nn.Linear(16 * 112 * 112, 10) # For 224x224 input
def forward(self, x):
x = self.conv1(x)
x = self.relu(x)
x = self.pool(x)
x = x.view(-1, 16 * 112 * 112) # Flatten
x = self.fc(x)
return x
----------------------------------------------------------------------
CNNs are typically composed of two parts:
the feature extraction part which consists of a series of convolutional layers
the classification part which consists of a series of fully-connected layers
------------------------------------------------------------------------------
Feature maps
The input data at each layer is an image. With each layer we are applying a new
convolution over a new image. Each image is a 3D object that has a height,
width, and depth. Depth is referred to as the color channel where depth = 1 for
grayscale images and 3 for color images.
In the later layers, the images still have depth but they are not colors per se.
They are feature maps that represent the features extracted from the previous
layers. Typically depth increases as we go deeper through the network layers.
----------------------------------------------------------------------------
FC layers
Typically, either all fully-connected layers in a network have the same number of
hidden units or decrease at each layer.
Research has found that keeping the number of units constant doesn’t hurt the
neural network so it may be a good approach if you want to limit the number of
choices you have to make when designing your network.
Pick a number of units per layer and apply that to all your FC layers
------------------------------------------------------------------------------------
AlexNet vs. LeNet-5
AlexNet in PyTorch
import torch
import torch.nn as nn
class AlexNet(nn.Module):
def __init__(self, num_classes=1000):
super(AlexNet, self).__init__()
self.features = nn.Sequential(
nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2), # output: (96, 55, 55)
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2), # output: (96, 27, 27)
nn.Conv2d(96, 256, kernel_size=5, padding=2), # output: (256, 27, 27)
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2), # output: (256, 13, 13)
nn.Conv2d(256, 384, kernel_size=3, padding=1), # output: (384, 13, 13)
nn.ReLU(inplace=True),
nn.Conv2d(384, 384, kernel_size=3, padding=1), # output: (384, 13, 13)
nn.ReLU(inplace=True),
nn.Conv2d(384, 256, kernel_size=3, padding=1), # output: (256, 13, 13)
nn.ReLU(inplace=True),
nn.MaxPool2d(kernel_size=3, stride=2) # output: (256, 6, 6)
 )
self.classifier = nn.Sequential(
nn.Dropout(),
nn.Linear(256 * 6 * 6, 4096),
nn.ReLU(inplace=True),
nn.Dropout(),
nn.Linear(4096, 4096),
nn.ReLU(inplace=True),
nn.Linear(4096, num_classes)
 )
def forward(self, x):
x = self.features(x)
x = torch.flatten(x, 1) # flatten all dimensions except batch
x = self.classifier(x)
return x
--------------------------------------------------------------
LeNet5
AlexNet
VGGNet
Inception and GoogLeNet
ResNet
ConvNext
----------------------------------------------------------------
ResNet vs. Inception vs. VGG/AlexNet/LeNet
----------------------------------------------------------------
Classification/recognition: classification is about predicting a class label.
The last fully connected layer has the fixed dimension of classes
Transfer learning: use pretrained knowledge to train with few data
----------------------------------------------------------------
Popular pre-trained models:
ResNet
VGG
MobileNet
EfficientNet
YOLO (for object detection) 
--------------------------------------------------------------------
Transfer learning - Implementation
import torch
import torchvision.models as models
from torch import nn
# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)
# Freeze all layers
for param in model.parameters():
param.requires_grad = False
# Replace final layer for new task (e.g., 5 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)
# Now only fc layer parameters will be updated during training
--------------------------------------------------------------------------
Object Detection
YOLO
------------------------------------------------------------------
Exercise
Create an image classifier that identifies different types of fruits:
- Use a pre-trained model (ResNet, etc.)
- Fine-tune on a fruit dataset (e.g., cifar-10, cifar-100, fruits-360)
- Evaluate the model's performance
- Test on new images
---------------------------------------------------------------------
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset (e.g., a folder with class subfolders)
train_data = datasets.ImageFolder('data/train', transform=transform)
test_data = datasets.ImageFolder('data/test', transform=transform)

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
num_classes = len(train_data.classes)
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
        
    return class_names[predicted.item()], probability[predicted.item()].item()

# Example usage
image_path = 'image.jpg'
class_names = train_data.classes
class_name, confidence = predict_image(image_path, model, transform, class_names)
print(f'Predicted: {class_name} with confidence {confidence:.2f}')

----------------------------------------------------------------------------------------