import matplotlib.pyplot as plt
import os

def ShowCentroid(centroids, outdir, sample):
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    plt.clf()
    centroids = centroids.data.numpy()
    x = centroids.transpose()[0]
    y = centroids.transpose()[1]
    plt.scatter(x, y, color=colors[0])
    # plt.show(block=False)
    plt.savefig(os.path.join(outdir, "{}.png".format(sample)))
    # plt.pause(5)
    return

if __name__ == "__main__":
    ShowCentroid()
    a = 0
