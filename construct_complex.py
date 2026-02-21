#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:20:50 2024

@author: meganfairchild

This script is for the construction of the simplicial complex
"""

import gudhi
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False #so we do not get the LaTeX error

import csv


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = {i: {i} for i in range(n)}  # Initialize each point in its own component

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:            
            # Perform union
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
                self.components[root_x].update(self.components[root_y])  # Merge the components
                del self.components[root_y]
            else:
                self.parent[root_x] = root_y
                self.components[root_y].update(self.components[root_x])  # Merge the components
                del self.components[root_x]
                if self.rank[root_x] == self.rank[root_y]:
                    self.rank[root_y] += 1
            return root_x, root_y
        return None, None


def construct_calculate(data_array, max_dimension, max_edge_length,csv_file_name):
    """
    Constructs a Vietoris-Rips complex, computes persistence, and visualizes the persistence diagram.
    
    Parameters:
    - data_array: numpy array of data points.
    - max_dimension: Maximum dimension of the simplices.
    - max_edge_length: Maximum edge length for the Rips complex.
    """
    try:
        # Check if the input data array is empty; if so, raise an error
        if data_array.size == 0:
            raise ValueError("Input data array is empty.")
        
        # Construct a Vietoris-Rips complex from the data points with the specified maximum edge length
        rips_complex = gudhi.RipsComplex(points=data_array, max_edge_length=max_edge_length)
        print("rips constructed")
        
        # Create a simplex tree from the Rips complex with the specified maximum dimension
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        print("simplex constructed")

        # Compute the persistence of the simplex tree (the birth and death of homological features)
        persistence = simplex_tree.persistence()
        print("persistence computed")
        
        # Print the persistence intervals to the console for debugging and verification

        # print(f"Persistence intervals: {persistence}")
                
        # To get the points that contribute to the persistence we need the following.

        # print(simplex_tree.persistence()) # Simplex_tree.persistence() has the 0, 1 in the first component to tell you which bar it is.

        n_points = len(data_array)
        uf = UnionFind(n_points)

        # ok the above piece breaks our persistence list (dim, (birth, death)) into the dimension lists. The piece below does the 0-th homology pieces and includes the contributing points.

        csv_file_name_nocsv = csv_file_name.replace(".csv","")

        """ I commented the following out, because all I care about are the birth and death times of a given simplex."""


        # output_csv_contributing_points_file = f"{csv_file_name_nocsv}_0_information.csv"
        # print("working on" + output_csv_contributing_points_file)
        # with open(output_csv_contributing_points_file, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Dimension", "Filtration (Birth)", "Filtration (Death)", "Simplex", "Merged Components"])

        #     for birth_simplex, death_simplex in simplex_tree.persistence_pairs():

        #         death_filtration = simplex_tree.filtration(death_simplex)  # Extract the correct filtration value

        #         if len(birth_simplex) > 1: # The birth simplex are the vertices that contribute to the birth of a feature.
        #             birth_filtration = simplex_tree.filtration(birth_simplex)
        #         else:
        #             birth_filtration = 0  # Handles unbounded features. This would happen for the 0-th components.

        #         simplex_indices = tuple(sorted(death_simplex))

            #     # BLah blah lbllah 
                
            #     if len(simplex_indices) == 1:  # 0D feature (single point)
            #         writer.writerow([0, birth_filtration, death_filtration, simplex_indices, "N/A"])
            #     else:  # 1D or higher feature
            #         if len(simplex_indices) == 2:  # j-th simplex causing component merge
            #             p1, p2 = simplex_indices
            #             old_comp, new_comp = uf.union(p1, p2)
            #             if old_comp is not None:
            #                 merged_component = f"Component {uf.components.get(new_comp, old_comp)} merged with {uf.components.get(old_comp, new_comp)}"
            #                 # Ensure old_comp and new_comp exist in the dictionary before accessing their sizes
            #                 size_old = len(uf.components.get(old_comp, []))
            #                 size_new = len(uf.components.get(new_comp, []))

            #                 # Choose the larger component
            #                 if size_new > size_old:
            #                     components = uf.components.get(new_comp,old_comp)
            #                 else:
            #                     components = uf.components.get(old_comp,new_comp)
            #                 writer.writerow([len(simplex_indices)-2, birth_filtration, death_filtration, simplex_indices, components])

            # print(f"All bar information saved to {output_csv_contributing_points_file}")

        """If I need it , uncomment to here from above."""


        # The below piece does just the times for all the features.
        output_csv_file = f"{csv_file_name_nocsv}_all_dimensions_simple.csv"
        print(f"Writing birth-death info to {output_csv_file}")
        with open(output_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Dimension", "Birth", "Death"])
            for dim, (birth, death) in persistence:
                writer.writerow([dim, birth, death])

        # Return the persistence intervals for further use if needed
        return persistence

    except Exception as e:
        # Catch any exception that occurs, print an error message, and re-raise the exception
        print(f"An error occurred: {e}")
        raise


"""
a function to compute the persistence barcodes
"""        
        
# def persistence_barcodes(persistence, output_file):    
#         # Visualize persistence barcode
#         f, ax = plt.subplots(figsize=(10, 10))
#         #plot the persistence barcodes
#         gudhi.plot_persistence_barcode(persistence, axes=ax, legend=True)
#         plt.savefig(output_file)
#         print(f"Persistence barcode saved as {output_file}")     

"""
a function to compute the persistence graph
"""  

def persistence_graph(persistence, output_file):
        # Create a new figure and axis for plotting the persistence diagram with a specified size
        f, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the persistence diagram using Gudhi's built-in function, with a legend
        gudhi.plot_persistence_diagram(persistence, axes=ax, legend=True)
        
        # Save the plot as an image file with the specified output file name
        plt.savefig(output_file)
        
        # Print a confirmation message that the persistence diagram was saved
        print(f"Persistence diagram saved as {output_file}")        

"""
end of script
"""