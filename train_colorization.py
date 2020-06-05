from colorization import Colorization
import torch.nn as nn
from clustered_dataset import KineticsClustered
import torch
import torch.optim as optim
from torchvision import transforms
from labels_viewer import GenerateLabelImage
import cv2
import os
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kinetics_path = "./data"
    trainset = KineticsClustered(base_path=kinetics_path)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=15,
                                              shuffle=False, num_workers=0)

    model = Colorization()
    print(model)
    model.to(device)


    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_values = []

    flag = False
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels = inputs.float(), labels.type(torch.int64)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # features = number of batches, channels, number of samples, height, width
            features = model.forward(inputs)
            H = 32
            W = 32
            # features = number of batches, number of samples, channels, height * width
            features = features.transpose(1, 2).view(-1, 4, 64, H * W)

            features_splitted = torch.split(features, [3, 1], dim=1)
            # reference_features = number of batches, number of reference, height * width, channels
            reference_features = features_splitted[0].transpose(2, 3).view(-1, 3*H*W, 64)
            # target_features = number of batches, number of target, height * width, channels
            target_features = features_splitted[1].transpose(2, 3).view(-1, H*W, 64)

            labels_splitted = torch.split(labels, [3, 1], dim=1)
            reference_labels = labels_splitted[0].view(-1, 3*H*W)
            target_labels = labels_splitted[1].view(-1, H*W)

            innerproduct = torch.matmul(target_features, reference_features.transpose(1, 2))
            similarity = nn.functional.softmax(innerproduct, dim=2)
            dense_reference_labels = torch.nn.functional.one_hot(reference_labels)

            prediction = torch.matmul(similarity, dense_reference_labels.type(torch.float32))

            loss = criterion(prediction.transpose(1, 2), target_labels)
            loss.backward()
            optimizer.step()

            if i == 0:
                #Write the images
                prediction_cpu = prediction.cpu()
                prediction_matrix = nn.functional.softmax(prediction_cpu, dim=2)
                prediction_matrix = torch.argmax(prediction_matrix, dim=2).view(-1, 32, 32)
                prediction_matrix = torch.split(prediction_matrix, 1)
                target_labels_list = torch.split(target_labels, 1)
                outpath = "./data/train/{}_{}".format(epoch, i+1)
                if os.path.exists(outpath) == False:
                    os.mkdir("./data/train/{}_{}".format(epoch, i+1))
                j = 0
                for pm, tl in zip(prediction_matrix, target_labels_list):
                    prediction_img = GenerateLabelImage(pm)
                    cv2.imwrite("./data/train/{}_{}/pred{}.png".format(epoch, i+1, j), prediction_img)
                    tl = tl.view(32, 32)
                    gt_img = GenerateLabelImage(tl)
                    cv2.imwrite("./data/train/{}_{}/gt{}.png".format(epoch, i+1, j), gt_img)
                    j = j  + 1

                with open("./data/train/{}_{}/loss".format(epoch, i+1), "w") as f:
                    f.write(str(loss.item()))

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                running_loss = running_loss / 20.
                loss_values.append(running_loss)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss))

                #Write the images
                prediction_cpu = prediction.cpu()
                prediction_matrix = nn.functional.softmax(prediction_cpu, dim=2)
                prediction_matrix = torch.argmax(prediction_matrix, dim=2).view(-1, 32, 32)
                prediction_matrix = torch.split(prediction_matrix, 1)
                target_labels_list = torch.split(target_labels, 1)
                outpath = "./data/train/{}_{}".format(epoch, i+1)
                if os.path.exists(outpath) == False:
                    os.mkdir("./data/train/{}_{}".format(epoch, i+1))
                j = 0
                for pm, tl in zip(prediction_matrix, target_labels_list):
                    prediction_img = GenerateLabelImage(pm)
                    cv2.imwrite("./data/train/{}_{}/pred{}.png".format(epoch, i+1, j), prediction_img)
                    j = j  + 1

                with open("./data/train/{}_{}/loss".format(epoch, i+1), "w") as f:
                    f.write(str(running_loss))

                #Reset loss
                running_loss = 0.0
    print ("Training end!!")