import torch
from training.model_utils import get_model

class ModelLoader:
    def __init__(self):
        # Use model_name as in training.py and keep num_classes and path
        self.model_configs = {
            'customcnn': {'model_name': "customcnn", 'num_classes': 38, 'path': 'saved_models/customcnn.pth'},
            'resnet18': {'model_name': "resnet18", 'num_classes': 38, 'path': 'saved_models/resnet18.pth'},
            'mobilenetv2': {'model_name': "mobilenet_v2", 'num_classes': 38, 'path': 'saved_models/mobilenetv2.pth'}
        }

    def load_model(self, model_name, device='cpu'):
        """Load a specific model by name and return it on the specified device."""
        config = self.model_configs[model_name]
        model = get_model(config['model_name'], config['num_classes'])
        model.load_state_dict(torch.load(config['path'], map_location=device))
        model.to(device)
        model.eval()
        return model