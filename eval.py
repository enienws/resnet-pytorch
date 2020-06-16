import torch
from colorization import Colorization
import torch.nn as nn
from davisloader import DavisDataset
import os
import cv2
from indexedpng import overlay, palette_mem, imwrite_indexed
from torchvision import transforms
import numpy as np


model_path = "/opt/data/8/test/models/model67999.pth"
outdir = "/opt/data_out/test"
used_model = 67999

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    davis_path = "/opt/data_davis"
    trainset = DavisDataset(base_path=davis_path)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=0)


    #Create network
    model = Colorization()
    #load model
    model.load_state_dict(torch.load(model_path))
    # model.to(device)
    #Set model to eval mode.
    model.eval()

    #Read from data
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # Inputs is normalized 256,256 grayscale image: 2, 1, 256, 256
        # Labels is annotations: 2, 1, 256, 256
        inputs, originals, labels_dummy, key = data
        inputs = [x.squeeze(0) for x in inputs]
        originals = [x.squeeze(0) for x in originals]
        labels_dummy = [x.type(torch.int64) for x in labels_dummy]
        labels = None
        for i in range(len(inputs)-1):

            if i==0:
                labels = labels_dummy[i]

            #Create input batch
            input_features = inputs[i:i+2]
            input_features = torch.stack(input_features)
            input_features = torch.unsqueeze(input_features, dim=0)

            # features = number of batches, channels, number of samples, height, width
            features = model.forward(input_features)
            H = 32
            W = 32
            # features = number of batches, number of samples, channels, height * width
            features = features.transpose(1, 2).view(-1, 2, 64, H * W)

            #Split the features for reference and target images
            features_splitted = torch.split(features, [1, 1], dim=1)
            # reference_features = number of batches, number of reference, height * width, channels
            reference_features = features_splitted[0].transpose(2, 3).view(-1, H * W, 64)
            # target_features = number of batches, number of target, height * width, channels
            target_features = features_splitted[1].transpose(2, 3).view(-1, H * W, 64)

            reference_labels = labels.view(1, H * W)

            innerproduct = torch.matmul(target_features, reference_features.transpose(1, 2))
            similarity = nn.functional.softmax(innerproduct, dim=2)
            dense_reference_labels = torch.nn.functional.one_hot(reference_labels)

            prediction = torch.matmul(similarity, dense_reference_labels.type(torch.float32))
            predicted_labels = nn.functional.softmax(prediction, dim=2)
            predicted_labels = torch.argmax(predicted_labels, dim=2).view(-1,32,32).squeeze(0)
            labels_temp = torch.unsqueeze(predicted_labels, dim=0)
            predicted_labels = predicted_labels.data.numpy()
            mask = cv2.resize(predicted_labels, (256,256), interpolation=cv2.INTER_NEAREST)

            #Create dir
            output_dir = os.path.join(outdir, str(used_model), key[0])
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                os.mkdir(os.path.join(output_dir, "result"))
                os.mkdir(os.path.join(output_dir, "masked"))

            #Write resulting annotations
            imwrite_indexed(os.path.join(output_dir, "result", str(i+1) + ".png"), mask)

            #Visualize results
            input_image = transforms.ToPILImage()(originals[i+1])
            input_image = np.asarray(input_image)
            overlayed_image = overlay(input_image, mask, colors=palette_mem)
            cv2.imwrite(os.path.join(output_dir, "masked", str(i+1)+".png"), overlayed_image)

            if i==0:
                labels = labels.squeeze(0)
                predicted_labels = labels.data.numpy()
                mask = cv2.resize(predicted_labels, (256, 256), interpolation=cv2.INTER_NEAREST)
                imwrite_indexed(os.path.join(output_dir, "result", str(i) + ".png"), mask)
                input_image = transforms.ToPILImage()(originals[i])
                input_image = np.asarray(input_image)
                overlayed_image = overlay(input_image, mask, colors=palette_mem)
                cv2.imwrite(os.path.join(output_dir, "masked", str(i) + ".png"), overlayed_image)
            labels = labels_temp