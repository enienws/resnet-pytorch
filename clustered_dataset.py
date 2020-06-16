from torch.utils.data import Dataset
from torchvision import transforms
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
        self.num_frames = 4
        self.skips = [0, 4, 4, 4][:4]

        self.transformBig = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])

        self.keys = self.getKeys()
        self.numberOfSamplesFake = 100000000
        self.numberOfSamples = 0

        self.currentKey = -1
        # self.currentKey = self.keys.index("test")
        self.totalSamplesForKey = 0
        self.currentSampleIdxForKey = 0

        self.advanceKey()

    def getKeys(self):
        keys = []
        for root, _, files in os.walk(os.path.join(self.base_path, "clustering")):
            for file in files:
                # key = os.path.splitext(file)[0]
                key = file.split("_8.pickle")[0]
                keys.append(key)
        return keys



    def __len__(self):
        if self.numberOfSamplesFake > self.numberOfSamples:
            return self.numberOfSamplesFake
        else:
            return self.numberOfSamples

    def readVideoFile(self, key):
        print("Reading Video: {}".format(key))
        filename_image = os.path.join(self.base_path, "processed", key+".mp4")
        capture = cv2.VideoCapture(filename_image)
        images_big = []
        for _, skip in itertools.cycle(enumerate(self.skips)):
            for _ in range(skip):
                capture.read()
            ret, image = capture.read()
            if not ret:
                break

            image_original = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            image_original = Image.fromarray(image_original)
            if self.transformBig is not None:
                image_big = self.transformBig(image_original)

            images_big.append(image_big)


        crop_size = int(len(images_big)/self.num_frames) * 4
        images_big = images_big[:crop_size]
        images_big = torch.stack(images_big)
        features_splitted = torch.split(images_big, [1,2],dim=1)
        features_L = features_splitted[0]
        return features_L

    def readFeatures(self, key):
        path = os.path.join(self.base_path, "clustering", key + "_8.pickle")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.numberOfSamples = self.numberOfSamples + obj["number_of_objects"]
        return obj

    def advanceKey(self):
        self.currentKey = self.currentKey + 1
        if self.currentKey == len(self.keys):
            print("EPOCH")
        p_obj = self.readFeatures(self.keys[self.currentKey])
        self.features_L_list = self.readVideoFile(self.keys[self.currentKey])
        self.clusters = p_obj["clusters"]
        self.totalSamplesForKey = p_obj["number_of_objects"]
        self.currentSampleIdxForKey = 0

    def consumeASample(self):
        features = self.features_L_list[self.currentSampleIdxForKey * 4:(self.currentSampleIdxForKey + 1) * 4]
        labels = self.clusters[self.currentSampleIdxForKey * 4:(self.currentSampleIdxForKey + 1) * 4]
        self.currentSampleIdxForKey = self.currentSampleIdxForKey + 1
        return features, labels

    def __getitem__(self, idx):
        #Consume a sample
        features, labels = self.consumeASample()

        if (self.currentSampleIdxForKey) * 4 == self.totalSamplesForKey:
            self.advanceKey()


        return features, labels
