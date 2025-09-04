import torchvision.models as models
import torch.nn as nn
import torch

NUM_CLASSES = 38  

# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Conv layer: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ReLU activation
        self.relu = nn.ReLU()
        # Fully connected layer
        self.fc = nn.Linear(16 * 112 * 112, num_classes) # For 224x224 input
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 112 * 112) # Flatten
        x = self.fc(x)
        return x
    
class AlexNet(nn.Module):
    def __init__(self, num_classes=38):
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

def get_resnet_model(new_num_classes):
    # Load a pre-trained ResNet model
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, new_num_classes)
    # Now only the parameters of the new layer will be updated during training
    return resnet

def get_mobilenet_v2_model(new_num_classes):
    # Load a pre-trained MobileNetV2 model
    mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze all layers
    for param in mobilenet_v2.parameters():
        param.requires_grad = False

    # Replace the final classifier layer
    num_features = mobilenet_v2.classifier[1].in_features
    mobilenet_v2.classifier[1] = nn.Linear(num_features, new_num_classes) # pyright: ignore[reportArgumentType]
    # Now only the parameters of the new layer will be updated during training
    return mobilenet_v2

def get_vgg16_model(new_num_classes):
    # Load a pre-trained VGG16 model
    vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # Freeze all layers
    for param in vgg16.parameters():
        param.requires_grad = False

    # Replace the final classifier layer
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, new_num_classes) # type: ignore
    # Now only the parameters of the new layer will be updated during training
    return vgg16

def get_efficientnet_b0_model(new_num_classes):
    # Load a pre-trained EfficientNet-B0 model
    efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # Freeze all layers
    for param in efficientnet_b0.parameters():
        param.requires_grad = False

    # Replace the final classifier layer
    num_features = efficientnet_b0.classifier[1].in_features
    efficientnet_b0.classifier[1] = nn.Linear(num_features, new_num_classes) # pyright: ignore[reportArgumentType]
    # Now only the parameters of the new layer will be updated during training
    return efficientnet_b0

def get_model(model_name):
    # ResNet VGG MobileNet EfficientNet
    if model_name == "resnet18":
        return get_resnet_model(new_num_classes=NUM_CLASSES)  # Example for binary classification
    elif model_name == "simple_cnn":
        return SimpleCNN(num_classes=NUM_CLASSES)
    elif model_name == "alexnet":
        return AlexNet(num_classes=NUM_CLASSES)
    elif model_name == "mobilenet_v2":
        return get_mobilenet_v2_model(new_num_classes=NUM_CLASSES)
    elif model_name == "vgg16":
        return get_vgg16_model(new_num_classes=NUM_CLASSES)
    elif model_name == "efficientnet_b0":
        return get_efficientnet_b0_model(new_num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Model {model_name} not recognized.")