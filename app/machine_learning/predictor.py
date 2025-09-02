import torch
import torchvision.transforms as transforms
from PIL import Image
import base64
from io import BytesIO
import time
from model_loader import ModelLoader

class ImagePredictor:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.models = {}  # Cache loaded models
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_all_models()  # Load once at startup!
    
    def _load_all_models(self):
        """Load all models into memory at startup"""
        for model_name in ['resnet18', 'mobilenetv2', 'customcnn']:
            try:
                self.models[model_name] = self.model_loader.load_model(model_name)
                print(f"✓ Loaded {model_name}")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
    
    def preprocess_image(self, base64_image):
        """Convert base64 string to tensor"""
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        tensor = self.transform(image)
        # Ensure we have a torch.Tensor before calling .unsqueeze to satisfy static analyzers
        if not isinstance(tensor, torch.Tensor):
            tensor = transforms.ToTensor()(image)
        return tensor.unsqueeze(0)
    
    def predict(self, base64_image, model_name):
        """Main prediction function"""
        start_time = time.time()
        
        # Get model (already loaded!)
        model = self.models.get(model_name)
        if not model:
            return {'error': f'Model {model_name} not available'}
        
        # Preprocess
        tensor = self.preprocess_image(base64_image)
        
        # Predict (inference)
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        processing_time = time.time() - start_time
        
        return {
            'prediction': predicted.item(),
            'confidence': confidence.item(),
            'processing_time': processing_time,
            'model_used': model_name
        }