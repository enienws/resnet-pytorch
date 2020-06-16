from torch.utils.data import Dataset
import torch
import json
import os
import cv2
import pickle
import itertools
import numpy as np
from PIL import Image
from torchvision import transforms

class KineticsClustering(Dataset):
    """Kinetics dataset."""

    def __init__(self, base_path, num_frames=1, skips=(0,)):
        super(KineticsClustering, self).__init__()
        self.base_path = base_path
        self.num_frames = num_frames
        self.skips = skips
        self.metas = []

        self.transformBigColor = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                                     transforms.ToTensor()
                                         ])

        self.transformBig = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])

        self.transformSmall = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                          transforms.Resize(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])

        metas = json.load(open(os.path.join(base_path, 'kinetics_train_1_sample.json')))

        self.keys = sorted(metas.keys())
        for _, key in enumerate(self.keys):
            metas[key]['key'] = key
            self.metas.append(metas[key])

        self.index = list(range(len(self.keys)))
        self.existing_keys = self.get_existing()

    def __len__(self):
        #TODO correct this line
        return len(self.existing_keys)
        # return 10000

    def get_existing(self):
        existing_keys = []
        for key in self.keys:
            path = os.path.join(self.base_path, "processed", key + ".mp4")
            if os.path.exists(path):
                if not os.path.exists(os.path.join(self.base_path, "clustering", key+"_8.pickle")):
                    existing_keys.append((key, path))
        return existing_keys

    def get_filename(self, name):
        if name not in self.keys:
            raise KeyError('not exists name at %s' % name)
        # LOGGER.debug('[Kinetics.get] %s', name)
        filename_label = os.path.join(self.base_path, name + '.label')
        filename_image = os.path.join(self.base_path, name + '.mp4')
        exists = os.path.exists(filename_image)
        labels_existance = os.path.exists(filename_label)
        if exists and not labels_existance:
            print("Video exists but labels are not: {}".format(name))
            exists = False
        return exists, filename_label, filename_image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #TODO correct this line
        # name = self.metas[idx]['key']
        name, filename_image = self.existing_keys[idx]
        # name = self.metas[0]['key']
        # exists, filename_label, filename_image = self.get_filename(name)
        # if not exists:
        #     print("Not existing video.")
        #     return None, None, None, None
        images_big = []
        images_small = []
        images_big_color = []
        capture = cv2.VideoCapture(filename_image)
        for _, skip in itertools.cycle(enumerate(self.skips)):
            for _ in range(skip):
                capture.read()
            ret, image = capture.read()
            if not ret:
                break

            image_original = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            image_original = Image.fromarray(image_original)
            if self.transformBigColor is not None:
                image_big_color = self.transformBigColor(Image.fromarray(image))
            if self.transformBig is not None:
                image_big = self.transformBig(image_original)
            if self.transformSmall is not None:
                image_small = self.transformSmall(image_original)

            images_big.append(image_big)
            images_small.append(image_small)
            images_big_color.append(image_big_color)


        if len(images_big) == 0:
            print("No frames are retreived: {}".format(name))
            return torch.empty(3,3,256,256), torch.empty(3,3,256,256), torch.empty(3,3,32,32), "Err"
        crop_size = int(len(images_big)/self.num_frames) * 4
        if crop_size < self.num_frames:
            print("Video has less than 4 frames: {}".format(name))
            return torch.empty(3,3,256,256), torch.empty(3,3,256,256), torch.empty(3,3,32,32), "Err"
        images_big = images_big[:crop_size]
        images_small = images_small[:crop_size]
        images_big_color = images_big_color[:crop_size]
        images_big_color, images_big, images_small = torch.stack(images_big_color), torch.stack(images_big), torch.stack(images_small)

        #Get gray channel from bigger image
        features_splitted = torch.split(images_big, [1,2],dim=1)
        features_L = features_splitted[0]

        #Get AB channel from smaller image
        features_splitted = torch.split(images_small, [1,2],dim=1)
        features_AB = features_splitted[1]

        return images_big_color, features_L, features_AB, name
