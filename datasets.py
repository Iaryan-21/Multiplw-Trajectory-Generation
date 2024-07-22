# datasets.py

import os
import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class NuScenesDataset(Dataset):
    def __init__(self, dataroot, split='train'):
        self.dataroot = os.path.join(dataroot, split)
        if not os.path.isdir(self.dataroot):
            raise NameError('dataroot does not exist')

        with open(os.path.join(self.dataroot, 'fn_list.txt'), 'r') as f:
            self.fns = [line.strip() for line in f]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        fn = self.fns[index]

        image_path = os.path.join(self.dataroot, 'image', fn + '.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        state_path = os.path.join(self.dataroot, 'state', fn + '.state')
        traj_path = os.path.join(self.dataroot, 'traj', fn + '.traj')
        agent_state_vector = torch.load(state_path)
        ground_truth = torch.load(traj_path)

        return image, agent_state_vector, ground_truth
        
    def __len__(self):
        return len(self.fns)
