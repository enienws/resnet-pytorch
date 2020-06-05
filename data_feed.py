from clustered_dataset import KineticsClustered
import torch
from kmeans_pytorch import kmeans
import pickle

if __name__ == "__main__":
    kinetics_path = "./data/"

    trainset = KineticsClustered(base_path=kinetics_path)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=False, num_workers=0)

    for i, data in enumerate(trainloader, 0):

        features_L, features_AB = data
        a = 0