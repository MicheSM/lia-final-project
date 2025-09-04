from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset
class MyDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y
    
# Create a dataset from NumPy arrays
data = np.random.randn(1000, 10)
targets = np.random.randint(0, 2, (1000,))
dataset = MyDataset(data, targets)

# Create a DataLoader
train_loader = DataLoader(
dataset,
batch_size=32, # Samples per batch
shuffle=True, # Shuffle data
num_workers=4 # Parallel data loading processes
)