from sklearn.cluster import KMeans
# from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import numpy as np

def find_optimal_k(X):
    # model = KMeans(random_state=42)
    # model = KMeans(n_init=10, random_state=42)

    # visualizer = KElbowVisualizer(model, k=(2,10))

    # visualizer.fit(X)
    # visualizer.show()

    # return visualizer.elbow_value_

    inertia = []
    K_range = range(2, 10)

    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        inertia.append(model.inertia_)

    plt.plot(K_range, inertia)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.show

    # Calculate second derivative (elbow detection)
    diff = np.diff(inertia)
    diff2 = np.diff(diff)

    elbow_index = np.argmin(diff2) + 1
    optimal_k = K_range[elbow_index]

    print("Inertia:", inertia)
    print("Choosen k:", optimal_k)

    return optimal_k


def train_kmeans(X, k: int):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model