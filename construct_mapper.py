#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:26:47 2024

@author: meganfairchild

This script is for MAPPER: dimension reduction and clustering analysis. 
"""

import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from kepler_mapper import KeplerMapper
import networkx as nx
import matplotlib.pyplot as plt 
from scipy.spatial.distance import pdist, squareform
import numpy as np


def construct_mapper_graph_3D(data_array, num_intervals, overlap_frac, mapper_name):

    """
    Constructs a Mapper graph for the given data using dimension reduction and clustering.
    
    Parameters:
    - data_array: numpy array of data points.
    - num_intervals: Number of intervals to divide the range of the filter function.
    - overlap_frac: Fractional overlap between consecutive intervals.
    """

    try:
        if data_array.size == 0:
            raise ValueError("Input data array is empty.")
        
        # Initialize
        mapper = km.KeplerMapper(verbose=1)
        
        # Step 1: Dimension reduction using PCA and DB scan
        pca = PCA(n_components=3)  # Reduce the data to 3 principal components.
        reduced_data = pca.fit_transform(data_array)  # Apply PCA on the input data.
        
        """
        original_distances = squareform(pdist(data_array))  # Distances in original space
        pca_distances = squareform(pdist(reduced_data))  # Distances in PCA space     
        print(f"Original distance range: {original_distances.min()} to {original_distances.max()}")
        print(f"PCA distance range: {pca_distances.min()} to {pca_distances.max()}")

        I was concerned with the distances used in the Vietoris Rips compared to the mapper clustering for consistency issues. 
        The output from the print with eps=30, min_samples=10, num_intervals= 20, overlap_frac = 0.3 
        AND VR parameters: max_dimension = 2, max_edge_length = 2.0
        resulted in the following: 
        Original distance range: 0.0 to 20.553035938833688
        PCA distance range: 0.0 to 20.313857718359444
        """
        
        dbscan = DBSCAN(eps=30, min_samples=10) #epsilon is distance between points, min_samples is the minimum number of points required to form a dense region 
        #(i.e., a cluster). A point is considered "core" if it has at least min_samples points (including itself) within its eps-radius.
        clusters = dbscan.fit_predict(reduced_data)

        
        # Step 2: Apply Mapper algorithm
        #mapper = Mapper()  # Initialize the Mapper object from scikit-tda.
        #projected_data = mapper.fit_transform(data_array, projection=[0,1]) # X-Y axis
        
        # Create a cover with 10 elements
        #cover = km.Cover(n_cubes=10)
        cover = km.Cover(n_cubes=num_intervals, perc_overlap=overlap_frac)


        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(reduced_data, cover=cover)

        # Visualize it
        mapper.visualize(graph, path_html= f"3D_{mapper_name}.html", title= f"{mapper_name}")

        """
        # Configure the Mapper with necessary parameters:
        graph = mapper.fit_transform(reduced_data, 
                                     cover_params={'num_intervals': num_intervals, 'overlap_frac': overlap_frac}, 
                                     clustering_method=DBSCAN(eps=0.5))
        
        #Step 3: Visualize the Mapper graph
        plt.figure(figsize=(10, 10))  # Set the figure size for the plot.
        nx.draw_networkx(graph, node_size=30, with_labels=False)  # Draw the Mapper graph.
        plt.savefig(output_file)  # Save the graph to the specified output file.
        print(f"Mapper graph saved as {output_file}")  # Inform the user about the saved file.
        """
    except Exception as e:
        print(f"An error occurred: {e}")  # Print any error that occurs.
        raise  # Raise the exception to indicate an error.
        
        

def construct_mapper_graph_2D(data_array, num_intervals, overlap_frac, mapper_name):

    """
    Constructs a Mapper graph for the given data using dimension reduction and clustering.
    
    Parameters:
    - data_array: numpy array of data points.
    - num_intervals: Number of intervals to divide the range of the filter function.
    - overlap_frac: Fractional overlap between consecutive intervals.
    """

    try:
        if data_array.size == 0:
            raise ValueError("Input data array is empty.")
        
        # Initialize
        mapper = km.KeplerMapper(verbose=1)
        
        # Step 1: Dimension reduction using PCA and DB scan
        pca = PCA(n_components=2)  # Reduce the data to 2 principal components.
        reduced_data = pca.fit_transform(data_array)  # Apply PCA on the input data.

        np.set_printoptions(precision=6, suppress=True)

        # Get the eigenvectors (principal components)
        eigenvectors = pca.components_

        list_eigenvec = []

        i = 0
        while i < len(eigenvectors):
            list_eigenvec.append(eigenvectors[i])
            i = i+1

        print(eigenvectors)

        # Get the eigenvalues (explained variance)
        eigenvalues = pca.explained_variance_
        print(eigenvalues)

        print(pca.explained_variance_ratio_)
        
        dbscan = DBSCAN(eps=30, min_samples=10) #epsilon is distance between points, min_samples is the minimum number of points required to form a dense region 
        #(i.e., a cluster). A point is considered "core" if it has at least min_samples points (including itself) within its eps-radius.
        clusters = dbscan.fit_predict(reduced_data)

        
        # Step 2: Apply Mapper algorithm
        #mapper = Mapper()  # Initialize the Mapper object from scikit-tda.
        # projected_data = mapper.fit_transform(data_array, projection='sum') # X-Y axis
        
        # Create a cover with 10 elements
        #cover = km.Cover(n_cubes=10)
        cover = km.Cover(n_cubes=num_intervals, perc_overlap=overlap_frac)


        # Create dictionary called 'graph' with nodes, edges and meta-information
        graph = mapper.map(reduced_data, cover=cover)

        # Visualize it
        mapper.visualize(graph, path_html= f"2D_{mapper_name}.html", title= f"{mapper_name}")

    except Exception as e:
        print(f"An error occurred: {e}")  # Print any error that occurs.
        raise  # Raise the exception to indicate an error.        

"""
end of script
"""