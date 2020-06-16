from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from indexedpng import imread_indexed
import cv2

class DavisDataset(Dataset):
    """Davis dataset."""

    def __init__(self, base_path, year = "2017", dataset = "val"):
        self.base_path = base_path
        self.year = year
        self.dataset = dataset
        self.keys = self.getKeys()

        self.transformImage = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))
                                         ])

        self.transformOriginal = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(256),
                                                     transforms.ToTensor()
                                         ])

        self.transformAnnot = transforms.Compose([transforms.Resize(32, interpolation=Image.NEAREST),
                                         transforms.CenterCrop(32)
                                         ])

    def getKeys(self):
        keys = []
        with open(os.path.join(self.base_path, "ImageSets", self.year, self.dataset+".txt")) as f:
            while True:
                key = f.readline()
                if not key:
                    break
                key = key.rstrip("\n")
                keys.append(key)
        return keys

    def readJPGs(self, key):
        imageList = []
        imageOriginalList = []
        path = os.path.join(self.base_path, "JPEGImages", "480p", key)

        for i in range(10000):
            image_original = cv2.imread(os.path.join(path, "{:05d}.jpg".format(i)))
            if image_original is None:
                break
            image = cv2.cvtColor(image_original, cv2.COLOR_BGR2Lab)
            image = image[:,:,0]
            image = Image.fromarray(image)
            image = self.transformImage(image)
            imageList.append(image)
            imageOriginalList.append(self.transformOriginal(Image.fromarray(image_original)))

        return imageList, imageOriginalList

    def readAnnot(self, key):
        imageList = []
        path = os.path.join(self.base_path, "Annotations", "480p", key)

        for i in range(10000):
            image = imread_indexed(os.path.join(path, "{:05d}.png".format(i)), self.transformAnnot)
            if image is not None:
                imageList.append(image)
            else:
                break

        return imageList

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #Get current key
        key = self.keys[idx]

        #Read JPEGs and annotations
        normed, original = self.readJPGs(key)
        annots = self.readAnnot(key)

        return normed, original, annots, key