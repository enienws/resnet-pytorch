from clustering_dataset import KineticsClustering
import torch
from kmeans_pytorch import kmeans
import pickle
from labels_viewer import viewer_main
from datetime import datetime
import os
import time
from torchvision import transforms
from showcentroid import ShowCentroid

output_path = "/opt/data/clustering/"
num_clusters = 16
if __name__ == "__main__":
    kinetics_path = "/opt/data"
    num_reference = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = KineticsClustering(base_path=kinetics_path, num_frames=num_reference + 1, skips=[0, 4, 4, 4][:num_reference + 1])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=0)

    total_processed = 0
    for i, data in enumerate(trainloader, 0):

        images_big_color, features_L, features_AB, sample_name = data
        B,S,C,W,H = features_L.shape
        sample_name = sample_name[0]

        a1 = time.time()

        #Check the results
        if sample_name == "Err":
            print("Passing sample: {}".format(sample_name))
            continue

        #Check whether clustering was performed before.
        #Create a path for sample class
        # if os.path.exists(os.path.join(output_path, sample_name)) == False:
        #     os.mkdir(os.path.join(output_path, sample_name))
        if os.path.exists(os.path.join(output_path, sample_name + ".pickle")) != False:
            print("{} is processed before.".format(sample_name))
            continue

        print("Processing: {}".format(sample_name))

        #Calculate clusters
        # features_AB = torch.randn(10000, 2, dtype=torch.float32) / 6 + .5
        features_AB = features_AB.squeeze(0).transpose(0,1).reshape(2,-1).transpose(0,1).contiguous().type(torch.float32)
        cluster_ids_x, cluster_centers = kmeans(
            X=features_AB,
            num_clusters=num_clusters,
            distance='euclidean', device=torch.device('cuda'),
            tol=0.0000005
        )
        cluster_ids_x = cluster_ids_x.reshape(-1, 1, 32, 32)

        now = datetime.now()
        data_obj = {
            # 'features_L': features_L.squeeze(0),
                  'clusters': cluster_ids_x,
                  'centers': cluster_centers,
                  'number_of_objects': S,
                  'class_name': sample_name,
                  'timeofday': now.strftime("%d/%m/%y %H:%M")}

        # #View the labels
        # # features_L.shape
        # # Create gray image
        # images_big_color_split = torch.split(images_big_color.squeeze(0), 1, dim=0)
        # transformer = transforms.ToPILImage()
        # images_color = []
        # for i, image_big_color in enumerate(images_big_color_split):
        #     image_big_color = transformer(image_big_color.squeeze(0))
        #     images_color.append(image_big_color)
        #     # image_big_color.save("./data/deneme/{}.png".format(i))
        # viewer_main(data_obj, output_path, images_color)

        #View the centroids
        # ShowCentroid(cluster_centers, os.path.join(output_path, sample_name), sample_name)

        #Write the output file.
        with open(os.path.join(output_path, "{}_{}.pickle".format(sample_name, num_clusters)), "wb") as f:
            pickle.dump(data_obj, f)

        a2 = time.time()
        proc_time = (a2-a1)*1000
        total_processed += proc_time
        # av_proc = total_processed / float(i)
        print("Av. processed: %.2f" % proc_time)

