import torch
from pathlib import Path


MODEL_PATH = Path(__file__).parent.parent / 'saved_models/v2/simple_cnn.pth'
model = torch.load(MODEL_PATH, weights_only=False)