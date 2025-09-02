import torch

class ModelLoader:
    def __init__(self):
        self.model_configs = {
            'customcnn': {'class': "CustomCNN", 'num_classes': 38, 'path': 'saved_models/customcnn.pth'},
            'resnet18': {'class': "ResNet18Transfer", 'num_classes': 38, 'path': 'saved_models/resnet18.pth'},
            'mobilenetv2': {'class': "MobileNetV2Transfer", 'num_classes': 38, 'path': 'saved_models/mobilenetv2.pth'}
        }
    
    def load_model(self, model_name):
        """Load a specific model"""
        config = self.model_configs[model_name]
        
        # Create model instance
        model = config['class'](config['num_classes'])
        
        # Load trained weights
        model.load_state_dict(torch.load(config['path'], map_location='cpu'))
        model.eval()  # Set to evaluation mode
        
        return model