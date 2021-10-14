# Example from README
from topologylayer.nn import AlphaLayer, PartialSumBarcodeLengths, BarcodePolyFeature, RipsLayer
import torch, numpy as np, matplotlib.pyplot as plt
import numpy as np
from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

# paper1: A Topology Layer for Machine Learning: https://arxiv.org/pdf/1905.12200.pdf
# paper2: A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology
#                    https://arxiv.org/pdf/1910.01877.pdf


np.random.seed(0)

def heatmap(n_components=3):
    n = 10
    X = np.random.randn(n,n)*0.1
    x = torch.autograd.Variable(torch.tensor(X).type(torch.float), requires_grad=True)

    layer = LevelSetLayer2D(size=(n,n),  sublevel=False)
    f1 = PartialSumBarcodeLengths(dim=0, skip=(n_components-1))
    # f1 = SumBarcodeLengths(dim=0)

    optimizer = torch.optim.Adam([x], lr=1e-3)
    for i in range(500):
        optimizer.zero_grad()
        loss = f1(layer(x))
        loss += torch.norm(x.sum() - n_components)
        loss.backward()
        print(i, loss.item())
        optimizer.step()

    # save figure
    fig, ax = plt.subplots(ncols=3, figsize=(15,5))
    x = x.detach().numpy()
    ax[0].imshow(X)
    ax[0].set_title("Truth")
    ax[1].imshow(X)
    ax[1].set_title("OLS")
    ax[2].imshow(x)
    ax[2].set_title("Topology Regularization")
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, left=False)
    plt.savefig('noisy_circle.png')


def scatter(n_components=1):
    X = np.random.rand(100, 2)
    x = torch.autograd.Variable(torch.tensor(X).type(torch.float), requires_grad=True)
    
    layer = AlphaLayer(maxdim=1)
    f1 = PartialSumBarcodeLengths(dim=0, skip=n_components)

    optimizer = torch.optim.Adam([x], lr=1e-2)
    for i in range(200):
        optimizer.zero_grad()
        loss = f1(layer(x))
        loss.backward()
        print(i, loss.item())
        optimizer.step()

    # # save figure
    y = x.detach().numpy()
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))
    ax[0].scatter(X[:,0], X[:,1])
    ax[0].set_title("Before")
    ax[1].scatter(y[:,0], y[:,1])
    ax[1].set_title("After")
    for i in range(2):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].tick_params(bottom=False, left=False)
    plt.savefig('holes.png')

scatter(n_components=20)