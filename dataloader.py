
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2


class CustomDataset(Dataset):
    def __init__(self, file_paths_file):
        self.data_list = self.load_file_paths(file_paths_file)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.im_size = 1000

    def load_file_paths(self, file_paths_file):
        with open(file_paths_file, 'r') as file:
            lines = file.readlines()
        data_list = [line.strip().split(',') for line in lines]
        return data_list

    def __len__(self):
        return len(self.data_list)
    

    def __getitem__(self, idx):
        img_path, mask_path = self.data_list[idx]
        image = cv2.imread(img_path,1)
        image = cv2.resize(image, (self.im_size,self.im_size))
        image = image/255.0
        image = np.moveaxis(image, 2, 0)
        image = np.float32(image)
        image = torch.from_numpy(image)
        image = self.normalize(image)

        mask = cv2.imread(mask_path,0)
        mask = cv2.resize(mask, (self.im_size,self.im_size), interpolation = cv2.INTER_NEAREST)
        mask = mask/255.0
        mask = torch.from_numpy(mask)
        
        return image, mask
    


