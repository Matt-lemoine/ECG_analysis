# This file is for getting the Persistence Vector and the Betti Function for a given persistence diagram.

import numpy as np
import matplotlib as plt
import wfdb
import os

from pathlib import Path
import csv
import time

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Wavelet_approx as wa
import SWE as swe
import construct_complex as cc


# start = time.time()

def area_of_tri(b,d):

    height = d - b

    area = (height**2) / 2

    return area

def area_of_trap(d_1, b_2, d_2):

    # This is only for when the b_2 < d_1. If b_2 >= d_2, then use area_of_tri.
    # b_1 and d_1 are the birth and death times of the point that happens sooner in the time than b_2, d_2.
    # You don't need b_1 for this calculation.

    left_side = d_2 - d_1
    base = d_1 - b_2

    area = (left_side * base) + ((left_side**2) / 2)

    return area


def per_vec_dim(persistence_diagram, dimension):

    # persistence_diagram is a np.array

    count = 0
    total_area = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            triangle = area_of_tri(b,d)

            total_area = total_area + triangle
            count = count + 1
            i = i+1
        else:
            i = i+1
    
    return [count, total_area]

def per_vec_dim_nocount(persistence_diagram, dimension):

    # persistence_diagram is a np.array

    total_area = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            triangle = area_of_tri(b,d)

            total_area = total_area + triangle
            i = i+1
        else:
            i = i+1
    
    return [total_area]

def per_vec_dim_arctan(persistence_diagram, dimension):

    # persistence_diagram is a np.array

    count = 0
    total_area = 0

    i = 0
    while i < len(persistence_diagram):
        if persistence_diagram[i,0] == dimension and persistence_diagram[i,2] != float("inf"):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            triangle = area_of_tri(b,d)

            total_area = total_area + triangle
            count = count + 1
            i = i+1
        else:
            i = i+1
    
    count = np.arctan(count)
    
    return [count, total_area]

def per_vec_dim_overlap(persistence_diagram, dimension):

    # persistence_diagram is a np.array

    count = 0
    total_area = 0

    if dimension == 0:
        i = 0
        max_death = 0
        while i< len(persistence_diagram):
            if persistence_diagram[i,2] > max_death and persistence_diagram[i,2] != float("inf"):
                max_death = persistence_diagram[i,2]
                count = count + 1
                i = i+1
            else:
                i = i+1
        
        total_area = area_of_tri(0,max_death)
        count = np.arctan(count)
    else:
        persistence_diagram = persistence_diagram[persistence_diagram[:, 1].argsort()] # to sort so that all the birth times are in order.

        i = 0
        indices_of_max_points = []
        while i < len(persistence_diagram):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            dim = persistence_diagram[i,0]

            if dim == dimension:
                if len(indices_of_max_points) == 0:
                    indices_of_max_points.append(i)
                    count = count+1
                    i = i+1
                else:
                    length = len(indices_of_max_points)
                    last_index = indices_of_max_points[length-1]

                    if b >= persistence_diagram[last_index,1] and d <= persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        count = count +1
                        i = i+1
                    elif b >= persistence_diagram[last_index,1] and d > persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        count = count +1
                        indices_of_max_points.append(i)
                        i = i+1
                    else:
                        i = i+1
            else:
                i = i+1
        
        i = 0
        while i < len(indices_of_max_points):
            current = indices_of_max_points[i]
            b = persistence_diagram[current,1]
            d = persistence_diagram[current,2]
            if i == 0:
                initial_area = area_of_tri(b,d)
                total_area = total_area + initial_area
                i = i+1
            else:
                previous = indices_of_max_points[i-1]
                d_1 = persistence_diagram[previous,2]
                if b >= d_1:
                    area = area_of_tri(b,d)
                    total_area = total_area + area
                    i = i+1
                else:
                    area = area_of_trap(d_1, b, d)
                    total_area = total_area + area
                    i = i+1
        
    return [count, total_area]

def per_vec_dim_nocount_overlap(persistence_diagram, dimension):

    # persistence_diagram is a np.array

    total_area = 0

    if dimension == 0:
        i = 0
        max_death = 0
        while i< len(persistence_diagram):
            if persistence_diagram[i,2] > max_death and persistence_diagram[i,2] != float("inf"):
                max_death = persistence_diagram[i,2]
                i = i+1
            else:
                i = i+1
        
        total_area = area_of_tri(0,max_death)
    else:
        persistence_diagram = persistence_diagram[persistence_diagram[:, 1].argsort()] # to sort so that all the birth times are in order.

        i = 0
        indices_of_max_points = []
        while i < len(persistence_diagram):
            b = persistence_diagram[i,1]
            d = persistence_diagram[i,2]
            dim = persistence_diagram[i,0]

            if dim == dimension:
                if len(indices_of_max_points) == 0:
                    indices_of_max_points.append(i)
                    i = i+1
                else:
                    length = len(indices_of_max_points)
                    last_index = indices_of_max_points[length-1]

                    if b >= persistence_diagram[last_index,1] and d <= persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        i = i+1
                    elif b >= persistence_diagram[last_index,1] and d > persistence_diagram[last_index,2] and d != float("inf") and dim == dimension:
                        indices_of_max_points.append(i)
                        i = i+1
                    else:
                        i = i+1
            else:
                i = i+1
        
        i = 0
        while i < len(indices_of_max_points):
            current = indices_of_max_points[i]
            b = persistence_diagram[current,1]
            d = persistence_diagram[current,2]
            if i == 0:
                initial_area = area_of_tri(b,d)
                total_area = total_area + initial_area
                i = i+1
            else:
                previous = indices_of_max_points[i-1]
                d_1 = persistence_diagram[previous,2]
                if b >= d_1:
                    area = area_of_tri(b,d)
                    total_area = total_area + area
                    i = i+1
                else:
                    area = area_of_trap(d_1, b, d)
                    total_area = total_area + area
                    i = i+1
        
    return [total_area]

def betti_fun(persistence_diagram, dimension, pat_id, lead, N):

    i = 0
    bs = []
    ds = []
    while i<len(persistence_diagram):
        if persistence_diagram[i,2] == float("inf"):
            i = i+1
        else:
            bs.append(persistence_diagram[i,1])
            ds.append(persistence_diagram[i,2])
            i = i+1
    
    max_b = max(bs)
    max_d = max(ds)
    max_time = max(max_b, max_d)

    interval = max_time / N

    j = 0
    betti_vec = []
    betti_vec.append(pat_id)
    betti_vec.append(lead)
    betti_vec.append(max_time)
    while j < N+1:
        t = j*interval
        if dimension == 0:
            count = 0
            i = 0
            while i < len(persistence_diagram):
                if persistence_diagram[i,2] != float("inf") and persistence_diagram[i,0] == dimension and persistence_diagram[i,2] <= t:
                    count = count + 1
                    i = i+1
                else:
                    i = i+1
            betti_vec.append(count)
        else:
            count = 0
            i = 0
            while i < len(persistence_diagram):
                if persistence_diagram[i,2] != float("inf") and persistence_diagram[i,0] == dimension:
                    if (t <= persistence_diagram[i,1] < t+interval) or (persistence_diagram[i,1]<t and persistence_diagram[i,2]>=t):
                        count = count+1
                        i = i+1
                    else:
                        i = i+1
                else:
                    i = i+1
            betti_vec.append(count)
        j = j+1

    if dimension == 0:
        j = len(betti_vec)
        last_val = betti_vec[j-1]
        output = []
        output.append(betti_vec[0])
        output.append(betti_vec[1])
        output.append(betti_vec[2])
        i = 3
        while i < len(betti_vec):
            output.append(last_val - betti_vec[i] +1) #You add 1 for the one connected component you have at the end of the persistent homology.
            i = i+1
        betti_vec = output
    else:
        j = len(betti_vec)
        output = []
        output.append(betti_vec[2])
        i = 3
        while i < len(betti_vec):
            output.append(betti_vec[i])
            i = i+1
        betti_vec = output


    return betti_vec

def put_together(thing_1, thing_2):
    
    output = []

    l1 = len(thing_1)
    l2 = len(thing_2)

    i = 0
    while i < l1+l2:
        if i < l1:
            output.append(thing_1[i])
            i = i+1
        elif i >= l1 and i < l1+l2:
            output.append(thing_2[i-l1])
            i = i+1
        else:
            print('you messed up dude.')
            i = i+1
    
    return output





# # ##########For Chaos for Betti Vectors###############


# lead_s = ['V1', 'V2', 'V3']
# Ns = [5, 10]

# get_to_files = Path('./Brugada_dataset/files')

# all_folder_names = []

# for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
#     if subdir.is_dir():
#         all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names)

# annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

# annotation = np.array([annotation[:,0], annotation[:,3]]).T

# # This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

# n = 0
# while n < len(Ns):
    # start = time.time()
#     N = Ns[n]
#     output_csv_file = f"Betti_Vectorization_V2_V3_{N}.csv"
#     with open(output_csv_file, "w", newline="") as f:
#         writer = csv.writer(f)
                
#         i = 0
#         while i < num_patients:
#             # print(num_patients - i)
#             pat_id = all_folder_names[i]

#             x = np.searchsorted(annotation[:,0], float(pat_id))

#             value = annotation[x,1]

#             if value == 2: # This takes out all the 7 entries of Brugada + some other heart murmur.
#                 i = i+1
#             else:
#                 j = 0
#                 while j < len(lead_s):
#                     lead = lead_s[j]
#                     pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

#                     dim = 0
#                     while dim < 2:
#                         if dim == 0:
#                             part_0 = betti_fun(pers_diagram, dim, pat_id, lead, N)
#                             dim = dim +1
#                         else:
#                             part_1 = betti_fun(pers_diagram, dim, pat_id, lead, N)
#                             dim = dim +1

#                     blah = put_together(part_0, part_1)
#                     blah.append(value)
#                     blah = np.array(blah)

#                     if lead == 'V2' or lead == 'V3':
#                         writer.writerow(blah)
#                         j = j+1
#                     else:
#                         j = j+1

#                     # writer.writerow(blah)
#                     # j = j+1

#                 i = i+1
    
#     time.sleep(1)
#     end = time.time()

#     print(f"Total runtime of the program is {end - start} seconds")

#     n = n+1

# ######For Chaos for Persistence Vector##########

# lead_s = ['V1', 'V2', 'V3']

# get_to_files = Path('./Brugada_dataset/files')

# all_folder_names = []

# for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
#     if subdir.is_dir():
#         all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names)

# annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

# annotation = np.array([annotation[:,0], annotation[:,3]]).T

# # This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

# output_csv_file = "Pers_Vec_Vectorization_V3.csv"
# with open(output_csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Patient_ID", "lead", "Total 0 Count", "Total 0 area", "Total 1 Count", "Total 1 area", "Value"])
    
#     i = 0
#     while i < num_patients:
#         # print(num_patients - i)
#         pat_id = all_folder_names[i]

#         x = np.searchsorted(annotation[:,0], float(pat_id))

#         value = annotation[x,1]

#         if value == 2: # This takes care of the places where you have a patient with Brugada + some other heart murmur.
#             i = i+1
#         else:
#             j = 0
#             while j < len(lead_s):
#                 lead = lead_s[j]
#                 pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

#                 dim = 0
#                 while dim < 2:
#                     if dim == 0:
#                         part_0 = per_vec_dim(pers_diagram, dim)
#                         dim = dim +1
#                     else:
#                         part_1 = per_vec_dim(pers_diagram, dim)
#                         dim = dim +1

#                 blah = put_together(part_0, part_1)
#                 info = [pat_id, lead]
#                 blah = put_together(info, blah)
#                 blah.append(value)
#                 blah = np.array(blah)

#                 if lead == 'V3':
#                     writer.writerow(blah)
#                     j = j+1
#                 else:
#                     j = j+1

#                 # writer.writerow(blah)
#                 # j = j+1

#             i = i+1

# time.sleep(1)
# end = time.time()

# print(f"Total runtime of the program is {end - start} seconds")




# ######For Chaos for Persistence Vector with ARCTAN##########

# start = time.time()

# lead_s = ['V1', 'V2', 'V3']

# get_to_files = Path('./Brugada_dataset/files')

# all_folder_names = []

# for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
#     if subdir.is_dir():
#         all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names)

# annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

# annotation = np.array([annotation[:,0], annotation[:,3]]).T

# # This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

# output_csv_file = "Pers_Vec_Arctan_Vectorization_V2_V3.csv"
# with open(output_csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Patient_ID", "lead", "Total 0 Count", "Total 0 area", "Total 1 Count", "Total 1 area", "Value"])
    
#     i = 0
#     while i < num_patients:
#         # print(num_patients - i)
#         pat_id = all_folder_names[i]

#         x = np.searchsorted(annotation[:,0], float(pat_id))

#         value = annotation[x,1]

#         if value == 2: # This takes care of the places where you have a patient with Brugada + some other heart murmur.
#             i = i+1
#         else:
#             j = 0
#             while j < len(lead_s):
#                 lead = lead_s[j]
#                 pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

#                 dim = 0
#                 while dim < 2:
#                     if dim == 0:
#                         part_0 = per_vec_dim_arctan(pers_diagram, dim)
#                         dim = dim +1
#                     else:
#                         part_1 = per_vec_dim_arctan(pers_diagram, dim)
#                         dim = dim +1

#                 blah = put_together(part_0, part_1)
#                 info = [pat_id, lead]
#                 blah = put_together(info, blah)
#                 blah.append(value)
#                 blah = np.array(blah)

#                 if lead == 'V2' or lead == 'V3':
#                     writer.writerow(blah)
#                     j = j+1
#                 else:
#                     j = j+1

#                 # writer.writerow(blah)
#                 # j = j+1

#             i = i+1

# time.sleep(1)
# end = time.time()

# print(f"Total runtime of the program is {end - start} seconds")


# ######For Chaos for Persistence Vector no overlaps##########

# start = time.time()

# lead_s = ['V1', 'V2', 'V3']

# get_to_files = Path('./Brugada_dataset/files')

# all_folder_names = []

# for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
#     if subdir.is_dir():
#         all_folder_names.append(subdir.name)

# num_patients = len(all_folder_names)

# annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

# annotation = np.array([annotation[:,0], annotation[:,3]]).T

# # This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

# output_csv_file = "Pers_Vec_Vectorization_nooverlap_all.csv"
# with open(output_csv_file, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Patient_ID", "lead", "Total 0 Count", "Total 0 area", "Total 1 Count", "Total 1 area", "Value"])
    
#     i = 0
#     while i < num_patients:
#         # print(num_patients - i)
#         pat_id = all_folder_names[i]

#         x = np.searchsorted(annotation[:,0], float(pat_id))

#         value = annotation[x,1]

#         if value == 2: # This takes care of the places where you have a patient with Brugada + some other heart murmur.
#             i = i+1
#         else:
#             j = 0
#             while j < len(lead_s):
#                 lead = lead_s[j]
#                 pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

#                 dim = 0
#                 while dim < 2:
#                     if dim == 0:
#                         part_0 = per_vec_dim_overlap(pers_diagram, dim)
#                         dim = dim +1
#                     else:
#                         part_1 = per_vec_dim_overlap(pers_diagram, dim)
#                         dim = dim +1

#                 blah = put_together(part_0, part_1)
#                 info = [pat_id, lead]
#                 blah = put_together(info, blah)
#                 blah.append(value)
#                 blah = np.array(blah)

#                 # if lead == 'V1':
#                 #     writer.writerow(blah)
#                 #     j = j+1
#                 # else:
#                 #     j = j+1

#                 writer.writerow(blah)
#                 j = j+1

#             i = i+1

# time.sleep(1)
# end = time.time()

# print(f"Total runtime of the program is {end - start} seconds")



######For Chaos for Persistence Vector no count with overlap##########

start = time.time()

lead_s = ['V1', 'V2', 'V3']

get_to_files = Path('./Brugada_dataset/files')

all_folder_names = []

for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)

annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

annotation = np.array([annotation[:,0], annotation[:,3]]).T

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

output_csv_file = "Pers_Vec_Vectorization_nocount_V2_V3.csv"
with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Patient_ID", "lead", "Total 0 area", "Total 1 area", "Value"])
    
    i = 0
    while i < num_patients:
        # print(num_patients - i)
        pat_id = all_folder_names[i]

        x = np.searchsorted(annotation[:,0], float(pat_id))

        value = annotation[x,1]

        if value == 2: # This takes care of the places where you have a patient with Brugada + some other heart murmur.
            i = i+1
        else:
            j = 0
            while j < len(lead_s):
                lead = lead_s[j]
                pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                dim = 0
                while dim < 2:
                    if dim == 0:
                        part_0 = per_vec_dim_nocount(pers_diagram, dim)
                        dim = dim +1
                    else:
                        part_1 = per_vec_dim_nocount(pers_diagram, dim)
                        dim = dim +1

                blah = put_together(part_0, part_1)
                info = [pat_id, lead]
                blah = put_together(info, blah)
                blah.append(value)
                blah = np.array(blah)

                if lead == 'V2' or lead == 'V3':
                    writer.writerow(blah)
                    j = j+1
                else:
                    j = j+1

                # writer.writerow(blah)
                # j = j+1

            i = i+1

time.sleep(1)
end = time.time()

print(f"Total runtime of the program is {end - start} seconds")


######For Chaos for Persistence Vector no count without overlap##########

start = time.time()

lead_s = ['V1', 'V2', 'V3']

get_to_files = Path('./Brugada_dataset/files')

all_folder_names = []

for subdir in get_to_files.iterdir(): # makes a list of all the files names (which are the patient numbers)
    if subdir.is_dir():
        all_folder_names.append(subdir.name)

num_patients = len(all_folder_names)

annotation = np.genfromtxt('Brugada_dataset/metadata.csv', delimiter=',', skip_header=1)

annotation = np.array([annotation[:,0], annotation[:,3]]).T

# This while loop cycles through all the folders and thus all the patients in the 'Brugada/files' file.

output_csv_file = "Pers_Vec_Vectorization_nocount_nooverlap_V2_V3.csv"
with open(output_csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Patient_ID", "lead", "Total 0 area", "Total 1 area", "Value"])
    
    i = 0
    while i < num_patients:
        # print(num_patients - i)
        pat_id = all_folder_names[i]

        x = np.searchsorted(annotation[:,0], float(pat_id))

        value = annotation[x,1]

        if value == 2: # This takes care of the places where you have a patient with Brugada + some other heart murmur.
            i = i+1
        else:
            j = 0
            while j < len(lead_s):
                lead = lead_s[j]
                pers_diagram = np.genfromtxt(f'Big_Output_thorough/{pat_id}_{lead}_pers_info_all_dimensions_simple.csv', delimiter=',', skip_header=1)

                dim = 0
                while dim < 2:
                    if dim == 0:
                        part_0 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                        dim = dim +1
                    else:
                        part_1 = per_vec_dim_nocount_overlap(pers_diagram, dim)
                        dim = dim +1

                blah = put_together(part_0, part_1)
                info = [pat_id, lead]
                blah = put_together(info, blah)
                blah.append(value)
                blah = np.array(blah)

                if lead == 'V2' or lead == 'V3':
                    writer.writerow(blah)
                    j = j+1
                else:
                    j = j+1

                # writer.writerow(blah)
                # j = j+1

            i = i+1

time.sleep(1)
end = time.time()

print(f"Total runtime of the program is {end - start} seconds")
