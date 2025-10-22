import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


def hierarchical_clustering(df, method):
    linkage_matrix = linkage(df, method=method)
    
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, orientation='top', distance_sort='ascending')
    plt.title('Single Linkage Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
    return linkage_matrix