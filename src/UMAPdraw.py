import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes 
from sklearn.svm import SVR
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

test_data = standardization(np.load("./dataset/seed_small/" + str(15) + "_data.npy"))
# print(test_data)
test_label = np.load("./dataset/seed_small/" + str(15) + "_label.npy")
test_label += 1

# test_data = test_data[::10]
# test_label = test_label[::10]

test_data = test_data.reshape(test_data.shape[0], -1)

reducer = umap.UMAP(random_state=42)
reducer.fit(test_data)
embedding = reducer.transform(test_data)

plt.figure(figsize=(4*3.5, 2*3))

color_map = ['indigo','limegreen','orchid','g','b','m','c'] # 7个类，准备7种颜色
def plot_embedding_2D(data, label, title, id):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    plt.subplot(2,4,id)
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontdict={'fontsize': 25,})

# plt.scatter(embedding[:, 0], embedding[:, 1], c=test_label, s=5)
plot_embedding_2D(embedding, test_label, 'Raw data', (1,1))
# fig.legend(["negative", "neutral", "postive"])

plt.savefig("./UMAP/vis.pdf")

list = [
        'seed_mlp_iid_lin_a-0.2_cross_lr-0.01', 'seed_mlp_iid_cha_a-5.0_random_lr-0.01', 'seed_mlp_iid_fre_a-5.0_cut_lr-0.01',
        'seed_mlp_iid_none_a-0.2_cross_lr-0.01',
        'seed_mlp_iid_lin_a-5.0_cross_lr-0.01', 'seed_mlp_iid_cha_a-5.0_binary_lr-0.01', 'seed_mlp_iid_fre_a-5.0_cross_lr-0.01',]

title_list = [ r'Linear $\alpha$=0.2', 'Cha random', r'Freq {$\alpha, \beta, \gamma$}',
              'FedAvg',
              r'Linear $\alpha$=5.0', 'Cha binary', r'Freq {$\delta, \alpha, \gamma$}']

for i in range(7):
    test_data = np.load("./LearnedFeature/" + list[i] + ".npy")
    reducer = umap.UMAP(random_state=42)
    reducer.fit(test_data)
    embedding = reducer.transform(test_data)
    y = i + 2
    x = 1
    if(y > 4):
        x = 2
        y -= 4
    plot_embedding_2D(embedding, test_label, title_list[i], i+2)

plt.savefig("./UMAP/visall_mlp_new1.pdf")