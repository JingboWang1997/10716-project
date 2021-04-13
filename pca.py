import torchvision.datasets as datasets
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from data import get_numpy_data

def build_pca(dataset: np.ndarray, variance: float) -> PCA:
    """
    build a pca model on the dataset using the variance
    """
    # obtaining the mnist data using the data loader and reshaping it
    # principal component analysis expects one-dimensional features
    flattened_dataset = dataset.reshape((-1, 784))
    pca_model = PCA(n_components=variance)
    pca_model.fit(flattened_dataset)
    return pca_model

def plot_latent_space(pca_model: PCA):
    """
    plot the top 20 dimensions in the latent space learned using PCA
    """
    fig, axes = plt.subplots(2,10,figsize=(9,3),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i, ax in enumerate(axes.flat):
        if i < len(pca_model.components_):
            ax.imshow(pca_model.components_[i].reshape(28,28),cmap='gray')
    plt.show()

def reconstruct(img: np.ndarray, pca_model: PCA):
    
    flattened = img.flatten()
    projected = pca_model.transform(flattened.reshape((1, len(flattened))))
    reconstructed = pca_model.inverse_transform(projected)
    
    fig, axes = plt.subplots(1,2,figsize=(9,3),
        subplot_kw={'xticks':[], 'yticks':[]},
        gridspec_kw=dict(hspace=0.01, wspace=0.01))
    axes.flat[0].imshow(img, cmap='gray')
    axes.flat[1].imshow(reconstructed.reshape(28,28), cmap='gray')
    plt.show()

numpy_data = get_numpy_data()
pca_model = build_pca(numpy_data[0], 0.8)
# plot_latent_space(pca_model)
reconstruct(numpy_data[2][0].squeeze(), pca_model)
