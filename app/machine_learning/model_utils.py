import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes):
    from torchvision import models
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        # Check type before replacing
        if isinstance(model.classifier[1], nn.Linear):
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            print("model.classifier structure:", model.classifier)
            raise TypeError(f"Expected model.classifier[1] to be nn.Linear, got {type(model.classifier[1])}")
        return model
    elif model_name == "customcnn":
        return CustomCNN(num_classes)
    else:
        raise ValueError("Unknown model name")