import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler



def mini_batch_kmeans(df, no_clusters = 3, batch_size=8)

    mbkmeans =  MiniBatchKMeans(n_clusters=no_clusters, batch_size=8)



    df['Cluster'] = mbkmeans.fit_predict(df)


    cluster_labels = {df['Cluster'].min() : "weak"  , df_selected['Cluster'].max() : "moderate", df_selected['Cluster'].median() : "strong"}