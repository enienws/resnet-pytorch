from torch.utils.data import Dataset
import torch
import json
import os
import cv2
import pickle
import itertools
import numpy as np
from PIL import Image

class KineticsClustered(Dataset):
    """Kinetics dataset."""

    def __init__(self, base_path,):
        self.base_path = base_path

        with open("./data/clustering/8_32_32/-2gnGuakDzI/-2gnGuakDzI_8.pickle", 'rb') as f:
            self.data = pickle.load(f)
        self.features_L_list = list(torch.split(self.data['features_L'], split_size_or_sections=1, dim=0))
        self.clusters = list(torch.split(self.data['clusters'], split_size_or_sections=1, dim=0))
        self.class_name = self.data['class_name']

    def __len__(self):
        #TODO correct this line
        # return len(self.index)
        return 15000

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = idx % 15
        features = self.features_L_list[idx*4:(idx+1)*4]
        features = [x.squeeze(0) for x in features]
        # features = [torch.stack([x,x,x]) for x in features]
        labels = self.clusters[idx * 4:(idx + 1) * 4]
        return torch.stack(features), torch.stack(labels)
