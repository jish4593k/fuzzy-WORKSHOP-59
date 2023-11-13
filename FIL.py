import pandas as pd
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt

class FuzzyCMeansClustering:
    def __init__(self, data, k=2, m=2.0, max_iter=100):
        self.data = data
        self.num_data_points, self.num_attributes = data.shape
        self.k = k
        self.m = m
        self.max_iter = max_iter

    def initialize_membership_matrix(self):
        membership_mat = np.random.rand(self.num_data_points, self.k)
        membership_mat = membership_mat / membership_mat.sum(axis=1, keepdims=True)
        return membership_mat.tolist()

    def calculate_cluster_center(self, membership_mat):
        cluster_mem_val = list(map(list, zip(*membership_mat)))
        cluster_centers = []

        for j in range(self.k):
            x = cluster_mem_val[j]
            x_raised = [e ** self.m for e in x]
            denominator = sum(x_raised)
            temp_num = []

            for i in range(self.num_data_points):
                data_point = list(self.data.iloc[i])
                prod = [x_raised[i] * val for val in data_point]
                temp_num.append(prod)

            numerator = [sum(x) for x in zip(*temp_num)]
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)

        return cluster_centers

    def update_membership_value(self, membership_mat, cluster_centers):
        p = float(2 / (self.m - 1))

        for i in range(self.num_data_points):
            x = list(self.data.iloc[i])
            distances = [np.linalg.norm(np.subtract(x, cluster_centers[j])) for j in range(self.k)]

            for j in range(self.k):
                den = sum([(distances[j] / distances[c]) ** p for c in range(self.k)])
                membership_mat[i][j] = 1 / den

        return membership_mat

    def get_clusters(self, membership_mat):
        cluster_labels = [max(enumerate(x), key=operator.itemgetter(1))[0] for x in membership_mat]
        return cluster_labels

    def fuzzy_c_means_clustering(self):
        membership_mat = self.initialize_membership_matrix()
        curr_iter = 0

        while curr_iter <= self.max_iter:
            cluster_centers = self.calculate_cluster_center(membership_mat)
            membership_mat = self.update_membership_value(membership_mat, cluster_centers)
            curr_iter += 1

        return self.get_clusters(membership_mat), cluster_centers

def visualize_clusters(data, labels, centers):
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], c='red', marker='X', label='Centers')
    plt.title('Fuzzy C-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Load your data
df_full = pd.read_csv("SPECTF_New.csv")
columns = list(df_full.columns)
features = columns[:len(columns) - 1]
df = df_full[features]

# Create an instance of FuzzyCMeansClustering
fuzzy_c_means = FuzzyCMeansClustering(df, k=2, m=2.0, max_iter=100)

# Perform clustering
labels, centers = fuzzy_c_means.fuzzy_c_means_clustering()

# Evaluate and visualize
a, p, r = accuracy(labels, class_labels)
print("Accuracy = " + str(a))
print("Precision = " + str(p))
print("Recall = " + str(r))

visualize_clusters(df, labels, centers)
