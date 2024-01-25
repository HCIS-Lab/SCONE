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
        '''
        output data format

         - current observation -
        ob_rgb:              [N(sequence length), C, H, W] -> [10, 3, 96, 96]
        ob_depth:            [N(sequence length), C, H, W]] -> [10, 1, 96, 96]
        ob_ee:               (x, y, z, ry) -> [4]
        local_rgb:           [C, H, W] -> [3, 96, 96]
        local_depth:         [C, H, W] -> [1, 96, 96]

        - active perception -
        interact_rgb:   [K(sequence length), C, H, W] -> [7, 3, 96, 96]
        interact_depth: [K(sequence length), C, H, W]] -> [7, 1, 96, 96]
        interact_ee:    [K(sequence length), D] -> [7, 4]

        - robot action (delta of end-effector poses) - 
        action:         (x, y, z, ry) -> [4]
        '''
        return 0
