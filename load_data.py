from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import glob
import torch

class FoodDataset(Dataset):
    def __init__(self, args):
        self.rootPath = args.dataset
    
    def __len__(self):
        return len(self.ob_rgb)

    def __getitem__(self, index):
        return 0
