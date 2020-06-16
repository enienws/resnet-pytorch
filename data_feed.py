from clustered_dataset import KineticsClustered
from davisloader import DavisDataset
import torch
from kmeans_pytorch import kmeans
import pickle

if __name__ == "__main__":
    # kinetics_path = "/opt/data/"
    # trainset = KineticsClustered(base_path=kinetics_path)
    #
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
    #                                           shuffle=False, num_workers=0)

    davis_path = "/opt/data_davis"
    evalset = DavisDataset(base_path=davis_path)
    trainloader = torch.utils.data.DataLoader(evalset, batch_size=1,
                                              shuffle=False, num_workers=0)

    for i, data in enumerate(trainloader, 0):

        features_L, features_AB = data
        a = 0
