import cv2
import pickle
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import colorsys

def _get_colors(num_colors):
    # colors=[]
    # for i in np.arange(0., 360., 360. / num_colors):
    #     hue = i/360.
    #     lightness = (40 + np.random.rand() * 10)/100.
    #     saturation = (90 + np.random.rand() * 10)/100.
    #     colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    # colors = np.array(colors)
    # colors = (colors * 255).astype(np.uint8)
    # np.random.shuffle(colors)
    # colors = np.squeeze(np.dstack([colors[:, 2], colors[:, 1], colors[:, 0]]), 0).astype(np.uint8)
    with open("./data/palette.pickle", "rb") as file:
        colors = pickle.load(file)
    return colors

colors = _get_colors(16)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def GenerateLabelImage(labels):
    labels = np.squeeze(labels)
    blank_image = np.zeros((256, 256, 3), np.uint8)
    for i in range(32):
        for j in range(32):
            color_index = labels[j][i]
            color = tuple(colors[color_index])
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.rectangle(blank_image, (i*8, j*8), ((i+1)*8, (j+1)*8), color, thickness=cv2.FILLED)
    # cv2.imwrite("./data/deneme.png", blank_image)
    return blank_image

def GenerateImage(images, labels):
    overall_image = np.zeros((512, 1024, 3), np.uint8)
    for i, (image, label) in enumerate(zip(images, labels)):
        _,_,C = image.shape
        if C == 1:
            image = cv2.merge((image, image, image))
        label_image = GenerateLabelImage(label)
        blank_image = np.zeros((256, 512, 3), np.uint8)
        blank_image[0:256, 0:256] = image
        blank_image[0:256, 256:512] = label_image
        if i == 0:
            overall_image[0:256,0:512] = blank_image
        elif i == 1:
            overall_image[0:256, 512:1024] = blank_image
        elif i == 2:
            overall_image[256:512, 0:512] = blank_image
        elif i == 3:
            overall_image[256:512, 512:1024] = blank_image

    return overall_image

def viewer_main(data, outpath, images_color=None):
    if images_color is not None:
        features_L_list = [np.array(x) for x in images_color]
    else:
        features_L_list = list(torch.split(data['features_L'], split_size_or_sections=1, dim=0))
    clusters = list(torch.split(data['clusters'], split_size_or_sections=1, dim=0))

    #Create directory for the visualizations
    outpath = os.path.join(outpath, data["class_name"], "visualizations")
    if os.path.exists(outpath) == False:
        os.mkdir(outpath)

    for idx in range(int(len(features_L_list)/4)):
        features = features_L_list[idx * 4:(idx + 1) * 4]
        if images_color is None:
            features = [x.squeeze(0) for x in features]
            features = [np.array(transforms.ToPILImage()(x)) for x in features]
        labels = clusters[idx * 4:(idx + 1) * 4]
        labels = [x.data.numpy() for x in labels]

        image = GenerateImage(features, labels)
        cv2.imwrite(os.path.join(outpath, "sample_{}.png".format(idx)), image)

def main():
    #Read the pickle
    with open("./data/-2gnGuakDzI.out", 'rb') as file:
        data = pickle.load(file)

    viewer_main(data, "./data")


if __name__ == "__main__":
    main()

