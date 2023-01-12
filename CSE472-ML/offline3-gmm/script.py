from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from GMM import GaussianMixtureModel

input_files = [
    "data/data2D.txt",
    "data/data3D.txt",
    "data/data6D.txt",
]




def main():

    # for input_file in input_files:
    # Read the data
    input_file = input_files[0]
    data = pd.read_csv(input_file, header=None, sep=" ")

    #   Data Reading & Compression
    ###########################################
    if data.shape[1] > 2:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=0, perplexity=30)
        data = tsne.fit_transform(data.values)

    data = pd.DataFrame(data)
    print(data)
    k_range = range(1, 3)
    log_likelihood = []

    for k in k_range:
        model = GaussianMixtureModel(n_comp=k)
        model.fit(data.values)
        log_likelihood.append(model.log_likelihood)

    # Plot the results
    # Task 1
    plt.plot(k_range, log_likelihood)
    plt.xlabel(f"Number of components: {len(k_range)}")
    plt.ylabel("Log Likelihood")
    plt.savefig(f"{input_file}_out.png")
    plt.clf()

    # Task 2
    model.plot(data.values)
    plt.clf()


if __name__ == "__main__":
    main()