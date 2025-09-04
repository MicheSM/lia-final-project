from src.trainer import train_model
from pathlib import Path


DATA_PATH = Path(__file__).parent / 'data'

model_names = ["resnet18", "alexnet", "simple_cnn", "vgg16", "mobilenet_v2", "efficientnet_b0"]

MODEL_PATH = Path(__file__).parent.parent / 'saved_models/v2/simple_cnn.pth'
#model = torch.load(MODEL_PATH, weights_only=False)



train_model(model_names[2])
