import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import lpips
from pytorch_msssim import MS_SSIM
from torch import from_numpy
#from dataset2 import ClimateHackDataset
from evaluate import Evaluator
from einops import repeat, rearrange

def normalize(t):
    t -= torch.min(t).detach()
    t /= (torch.max(t).detach() - torch.min(t).detach())
    t *= 2
    t -= 1
    return t

def visualize(x,y_true,y_pred):

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 12, figsize=(20,8))
    print(y_true.shape)
    # plot the twelve 128x128 input images
    for i in range(0,12):
        ax1[i].imshow(x[i,:,:], cmap='viridis')
        ax1[i].get_xaxis().set_visible(False)
        ax1[i].get_yaxis().set_visible(False)

    # plot twelve 64x64 true output images
    for i in range(0,12):
        ax2[i].imshow(y_true[i,:,:], cmap='viridis')
        ax2[i].get_xaxis().set_visible(False)
        ax2[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 true output images
    for i in range(12,24):
        ax3[i-12].imshow(y_true[i,:,:], cmap='viridis')
        ax3[i-12].get_xaxis().set_visible(False)
        ax3[i-12].get_yaxis().set_visible(False)

    # plot twelve 64x64 true output images
    for i in range(0,12):
        ax4[i].imshow(y_pred[i,:,:], cmap='viridis')
        ax4[i].get_xaxis().set_visible(False)
        ax4[i].get_yaxis().set_visible(False)

    # plot twelve more 64x64 true output images
    for i in range(12,24):
        ax5[i-12].imshow(y_pred[i,:,:], cmap='viridis')
        ax5[i-12].get_xaxis().set_visible(False)
        ax5[i-12].get_yaxis().set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig("Predictions.png")
    return True

def main():
    features = np.load("../features.npz")
    targets = np.load("../targets.npz")
    #dataset = ClimateHackDataset("../../data.npz", crops_per_slice=1, in_channels=12, out_channels=24)
    #coords, features, targets = dataset.load_data()
    #print(features.shape)


    #criterion = MS_SSIM(data_range=1023.0, size_average=False, win_size=3, channel=1)
    criterion = lpips.LPIPS(net='alex')
    #criterion = nn.MSELoss()
    evaluator = Evaluator()
    #pred = evaluator.predict(features["osgb"][0], features["data"][0])
    #visualize(features["data"][0], targets["data"][0], pred)

    scores = [
        criterion(
            normalize(from_numpy(evaluator.predict(*datum)).view(24,64,64).unsqueeze(dim=1).repeat(1,3,1,1)),
            normalize(from_numpy(target).view(24,64,64).unsqueeze(dim=1).repeat(1,3,1,1)),
        ).unsqueeze(0)
        for *datum, target in zip(features["osgb"], features["data"], targets["data"])
        #for *datum, target in zip(dataset.coordinates,dataset.features,dataset.labels)
    ]

    scores = torch.cat(scores, dim=0)
    #print(scores.shape)
    print(torch.mean(scores,dim=0).squeeze().detach().numpy())
    print(torch.std(scores,dim=0).squeeze().detach().numpy())
    #print(scores.shape)
    #print(scores[0].shape)
    #print(scores)
    #ax.set_title('Line plot with error bars')
    #plt.save("test.png")

    plt.show()
    np.savez("Persistence_Lpips",mean=1-torch.mean(scores,dim=0).detach().numpy(), std=1-torch.std(scores,dim=0).detach().numpy())
    #print(f"Score: {np.mean(scores)} ({np.std(scores)})")


if __name__ == "__main__":
    main()