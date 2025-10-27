import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler



def mini_batch_kmeans(df, no_clusters = 3, batch_size=8):

    mbkmeans =  MiniBatchKMeans(n_clusters=no_clusters, batch_size=batch_size)

    df['Cluster'] = mbkmeans.fit_predict(df)

    # cluster_labels = {df['Cluster'].min() : "weak"  , df['Cluster'].max() : "moderate", df['Cluster'].median() : "strong"}
    last_index = len(df.columns)  - 1 

    plt.figure(figsize = (20,10))

    plt.scatter(df.iloc[:last_index], [0]*len(df), c = df['Cluster'], cmap='rainbow')