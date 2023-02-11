from GMM import GaussianMixtureModel
from matplotlib import pyplot as plt
import pandas as pd


INPUT_FILES = [
    "data/data2D.txt",
    "data/data3D.txt",
    "data/data6D.txt",
]


def main():

    input_file = INPUT_FILES[2]
    data = pd.read_csv(input_file, header=None, sep=" ")
    print(f"File read complete! Data shape: {data.shape}")
    
    k_range = range(1, 12)
    log_likelihood = []

    for k in k_range:
        print(f"Running for k = {k}")
        model = GaussianMixtureModel(n_comp=k)
        model.fit(data.values)
        log_likelihood.append(model.log_likelihood)

    ###################################################
    #   Task 1 - Plotting k vs. max-log-likelihood
    ###################################################
    plt.plot(k_range, log_likelihood, marker="o")
    plt.xlabel(f"Max number of components: {len(k_range)}")
    plt.ylabel("Log Likelihood")
    plt.savefig(f"{input_file}_out.png")
    plt.clf()

    #####################################################################
    #   Task 3 - Compressing multi-dimensional data (only for plotting)
    #####################################################################
    if data.shape[1] > 2:
        print("Dimension greater than 2. Using PCA/UMAP/TSNE for compression...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data = pca.fit_transform(data.values)
        print(f"[PCA] Compression complete! New data shape: {data.shape}")

        # from umap import UMAP
        # umap = UMAP(n_components=2)
        # data = umap.fit_transform(data.values)
        # print(f"[UMAP] Compression complete! New data shape: {data.shape}")

        # from sklearn.manifold import TSNE
        # tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        # data = tsne.fit_transform(data.values)
        # print(f"[TSNE] Compression complete! New data shape: {data.shape}")

    ###################################################
    #   Task 2 - Animating Contour over Iterations
    ###################################################
    k_star = 4
    data = pd.DataFrame(data)

    model = GaussianMixtureModel(n_comp=k_star, anim=True)
    model.fit(data.values)
    model.plot_contour(data.values)


if __name__ == "__main__":
    main()