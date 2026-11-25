import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Visualize_EKG as vekg
import Normalize_EKG as nekg
import Spline_approx as sapp


"""
Note the first functions are strictly for plotting.
"""

def plot_3d(projected_points):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for x,y,z in projected_points:
        ax.scatter(x,y,z)

    return plt.show()

def plot_2d(projected_points):

    fig = plt.figure()

    for x,y in projected_points:
        plt.scatter(x,y)

    return plt.show()

def save_plot_3d(projected_points, naming_things):
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for x,y,z in projected_points:
        ax.scatter(x,y,z)

    # Save the plot as an image file with the specified output file name
    plt.savefig(naming_things)
    
    # Print a confirmation message that the persistence diagram was saved
    print(f"Projected points saved as {naming_things}.png")

def save_plot_2d(projected_points, naming_things):
    
    fig = plt.figure()

    for x,y in projected_points:
        plt.scatter(x,y)

    # Save the plot as an image file with the specified output file name
    plt.savefig(naming_things)
    
    # Print a confirmation message that the persistence diagram was saved
    print(f"Projected points saved as {naming_things}")


# =========================================================
# fUNCTION THAT WILL NEED TO BE IMPORTED TO main_EKG.py
# =========================================================

def SWE_w_Splines(ptf, lead, embedding_dimension):

    trimmed_signal = nekg.trim_by_CSV(ptf, lead)
    fun, n = sapp.get_cs_trimmed_signal(ptf, lead)

    i = 0
    while i <= 2:

        if i == 0:

            # =========================================================
            # SLIDING WINDOW PARAMETERS
            # =========================================================

            fixed_tau = 1.0

            mu = trimmed_signal.mean()
            acceptable_variance = 0.025

            minimum_tau = 0.001

            swe_points = []

            # =========================================================
            # HELPER FUNCTION
            # =========================================================

            def evaluate_spline(x):
                """
                Evaluate the spline and make sure the result is a scalar.
                """
                value = fun(x)

                value = np.asarray(value).squeeze()

                if value.ndim != 0:
                    raise ValueError(
                        f"fun({x}) returned an array with shape {value.shape} "
                        "instead of a scalar."
                    )

                return float(value)


            # =========================================================
            # ADAPTIVE SLIDING WINDOW EMBEDDING
            # =========================================================

            i = 0.0

            while i + (embedding_dimension -1) * fixed_tau < n:

                # Evaluate the standard sliding window

                sum_of_3 = 0
                s = 0
                while s < embedding_dimension:
                    f0 = abs(evaluate_spline(i + s*fixed_tau))

                    sum_of_3 = sum_of_3 + f0

                    s = s+1


                # -----------------------------------------------------
                # Use normal tau
                # -----------------------------------------------------

                if sum_of_3 <= (embedding_dimension) * (mu + acceptable_variance):

                    tau = fixed_tau


                # -----------------------------------------------------
                # Modify tau
                # -----------------------------------------------------

                else:

                    max_var = max(abs(trimmed_signal.min()), abs(trimmed_signal.max()))

                    diff_tau = (
                        fixed_tau
                        - abs((sum_of_3 / (embedding_dimension)) - mu - acceptable_variance) / max_var
                    )

                    if diff_tau >= minimum_tau:
                        tau = diff_tau
                    else:
                        tau = minimum_tau


                # -----------------------------------------------------
                # Make sure the adaptive window stays inside the domain
                # -----------------------------------------------------

                if i + (embedding_dimension -1) * tau >= n:
                    break


                # -----------------------------------------------------
                # Construct the 3D sliding-window point
                # -----------------------------------------------------

                multiples_of_tau = np.arange(embedding_dimension)

                point = np.zeros(embedding_dimension)

                for step in multiples_of_tau:
                    point[step] = evaluate_spline(i + step*tau)

                swe_points.append(point)

                # Move to the next window.
                #
                # IMPORTANT:
                # tau determines spacing INSIDE the window.
                # fixed_step determines how far we move the window.
                i += tau

            # =========================================================
            # CONVERT TO NUMPY ARRAY
            # =========================================================

            swe_points = np.asarray(swe_points, dtype=float)

            # print("Number of SWE points:", len(swe_points))
            # print("Shape:", swe_points.shape)

            if len(swe_points) >= 3000:
                i = 1
            else:
                i = 3

        elif i == 1:

            # =========================================================
            # SLIDING WINDOW PARAMETERS
            # =========================================================

            fixed_tau = 1.0

            mu = trimmed_signal.mean()
            acceptable_variance = 0.025

            minimum_tau = 0.01

            swe_points = []

            # =========================================================
            # HELPER FUNCTION
            # =========================================================

            def evaluate_spline(x):
                """
                Evaluate the spline and make sure the result is a scalar.
                """
                value = fun(x)

                value = np.asarray(value).squeeze()

                if value.ndim != 0:
                    raise ValueError(
                        f"fun({x}) returned an array with shape {value.shape} "
                        "instead of a scalar."
                    )

                return float(value)


            # =========================================================
            # ADAPTIVE SLIDING WINDOW EMBEDDING
            # =========================================================

            i = 0.0

            while i + (embedding_dimension -1) * fixed_tau < n:

                # Evaluate the standard sliding window

                sum_of_3 = 0
                s = 0
                while s < embedding_dimension:
                    f0 = abs(evaluate_spline(i + s*fixed_tau))

                    sum_of_3 = sum_of_3 + f0

                    s = s+1


                # -----------------------------------------------------
                # Use normal tau
                # -----------------------------------------------------

                if sum_of_3 <= (embedding_dimension) * (mu + acceptable_variance):

                    tau = fixed_tau


                # -----------------------------------------------------
                # Modify tau
                # -----------------------------------------------------

                else:

                    max_var = max(abs(trimmed_signal.min()), abs(trimmed_signal.max()))

                    diff_tau = (
                        fixed_tau
                        - abs((sum_of_3 / (embedding_dimension)) - mu - acceptable_variance) / max_var
                    )

                    if diff_tau >= minimum_tau:
                        tau = diff_tau
                    else:
                        tau = minimum_tau


                # -----------------------------------------------------
                # Make sure the adaptive window stays inside the domain
                # -----------------------------------------------------

                if i + (embedding_dimension -1) * tau >= n:
                    break


                # -----------------------------------------------------
                # Construct the 3D sliding-window point
                # -----------------------------------------------------

                multiples_of_tau = np.arange(embedding_dimension)

                point = np.zeros(embedding_dimension)

                for step in multiples_of_tau:
                    point[step] = evaluate_spline(i + step*tau)

                swe_points.append(point)

                # Move to the next window.
                #
                # IMPORTANT:
                # tau determines spacing INSIDE the window.
                # fixed_step determines how far we move the window.
                i += tau

            # =========================================================
            # CONVERT TO NUMPY ARRAY
            # =========================================================

            swe_points = np.asarray(swe_points, dtype=float)

            # print("Number of SWE points:", len(swe_points))
            # print("Shape:", swe_points.shape)

            if len(swe_points) >= 3000:
                print('still bigger than 3000.')
                i = 2
            else:
                i = 3

        elif i == 2:

            # =========================================================
            # SLIDING WINDOW PARAMETERS
            # =========================================================

            fixed_tau = 1.0

            mu = trimmed_signal.mean()
            acceptable_variance = 0.025

            minimum_tau = 0.05

            swe_points = []

            # =========================================================
            # HELPER FUNCTION
            # =========================================================

            def evaluate_spline(x):
                """
                Evaluate the spline and make sure the result is a scalar.
                """
                value = fun(x)

                value = np.asarray(value).squeeze()

                if value.ndim != 0:
                    raise ValueError(
                        f"fun({x}) returned an array with shape {value.shape} "
                        "instead of a scalar."
                    )

                return float(value)


            # =========================================================
            # ADAPTIVE SLIDING WINDOW EMBEDDING
            # =========================================================

            i = 0.0

            while i + (embedding_dimension -1) * fixed_tau < n:

                # Evaluate the standard sliding window

                sum_of_3 = 0
                s = 0
                while s < embedding_dimension:
                    f0 = abs(evaluate_spline(i + s*fixed_tau))

                    sum_of_3 = sum_of_3 + f0

                    s = s+1


                # -----------------------------------------------------
                # Use normal tau
                # -----------------------------------------------------

                if sum_of_3 <= (embedding_dimension) * (mu + acceptable_variance):

                    tau = fixed_tau


                # -----------------------------------------------------
                # Modify tau
                # -----------------------------------------------------

                else:

                    max_var = max(abs(trimmed_signal.min()), abs(trimmed_signal.max()))

                    diff_tau = (
                        fixed_tau
                        - abs((sum_of_3 / (embedding_dimension)) - mu - acceptable_variance) / max_var
                    )

                    if diff_tau >= minimum_tau:
                        tau = diff_tau
                    else:
                        tau = minimum_tau


                # -----------------------------------------------------
                # Make sure the adaptive window stays inside the domain
                # -----------------------------------------------------

                if i + (embedding_dimension -1) * tau >= n:
                    break


                # -----------------------------------------------------
                # Construct the 3D sliding-window point
                # -----------------------------------------------------

                multiples_of_tau = np.arange(embedding_dimension)

                point = np.zeros(embedding_dimension)

                for step in multiples_of_tau:
                    point[step] = evaluate_spline(i + step*tau)

                swe_points.append(point)

                # Move to the next window.
                #
                # IMPORTANT:
                # tau determines spacing INSIDE the window.
                # fixed_step determines how far we move the window.
                i += tau

            # =========================================================
            # CONVERT TO NUMPY ARRAY
            # =========================================================

            swe_points = np.asarray(swe_points, dtype=float)

            # print("Number of SWE points:", len(swe_points))
            # print("Shape:", swe_points.shape)

            if len(swe_points) >= 3000:
                print('Something big is wrong that you changed minimum tau and still bigger than 3000.')
            else:
                print('you are all good now')

        
    return swe_points


# =========================================================
# FROM CONFERENCE!!!!
# =========================================================


# # =========================================================
# # LOAD DATA
# # =========================================================

# # ptf = 'Brugada_dataset/files/188981/188981'

# # lead = ["V3"]

# # lb = 121
# # ub = 923

# trimmed_signal = nekg.trim_by_CSV(ptf, lead)

# # Get cubic spline approximation
# fun, n = sapp.get_cs_trimmed_signal(ptf, lead)

# print("n =", n)


# # =========================================================
# # SLIDING WINDOW PARAMETERS
# # =========================================================

# fixed_tau = 1.0
# fixed_step = 1.0

# mu = trimmed_signal.mean()
# acceptable_variance = 0.025

# minimum_tau = 0.01

# embedding_dimension = 3

# swe_points = []


# # =========================================================
# # HELPER FUNCTION
# # =========================================================

# def evaluate_spline(x):
#     """
#     Evaluate the spline and make sure the result is a scalar.
#     """
#     value = fun(x)

#     value = np.asarray(value).squeeze()

#     if value.ndim != 0:
#         raise ValueError(
#             f"fun({x}) returned an array with shape {value.shape} "
#             "instead of a scalar."
#         )

#     return float(value)


# # =========================================================
# # ADAPTIVE SLIDING WINDOW EMBEDDING
# # =========================================================

# i = 0.0

# while i + (embedding_dimension - 1) * fixed_tau < n:

#     print("i =", i)

#     # Evaluate the standard sliding window

#     sum_of_3 = 0
#     s = 0
#     while s < embedding_dimension:
#         f0 = abs(evaluate_spline(i + s*fixed_tau))

#         sum_of_3 = sum_of_3 + f0

#         s = s+1


#     # -----------------------------------------------------
#     # Use normal tau
#     # -----------------------------------------------------

#     if sum_of_3 <= 3 * (mu + acceptable_variance):

#         tau = fixed_tau


#     # -----------------------------------------------------
#     # Modify tau
#     # -----------------------------------------------------

#     else:

#         max_var = max(abs(trimmed_signal.min()), abs(trimmed_signal.max()))

#         diff_tau = (
#             fixed_tau
#             - abs((sum_of_3 / embedding_dimension) - mu - acceptable_variance) / max_var
#         )

#         if diff_tau != 0:
#             tau = diff_tau
#         else:
#             tau = minimum_tau


#     # -----------------------------------------------------
#     # Make sure the adaptive window stays inside the domain
#     # -----------------------------------------------------

#     if i + (embedding_dimension -1) * tau >= n:
#         break


#     # -----------------------------------------------------
#     # Construct the 3D sliding-window point
#     # -----------------------------------------------------

#     multiples_of_tau = np.arange(embedding_dimension)

#     point = np.zeros(embedding_dimension)

#     for step in multiples_of_tau:
#         point[step] = evaluate_spline(i + step*tau)

#     swe_points.append(point)

#     # Move to the next window.
#     #
#     # IMPORTANT:
#     # tau determines spacing INSIDE the window.
#     # fixed_step determines how far we move the window.
#     i += tau


# # =========================================================
# # CONVERT TO NUMPY ARRAY
# # =========================================================

# swe_points = np.asarray(swe_points, dtype=float)

# print("Number of SWE points:", len(swe_points))
# print("Shape:", swe_points.shape)


# # # =========================================================
# # # PLOT
# # # =========================================================

# # fig = plt.figure()

# # ax = fig.add_subplot(111, projection='3d')

# # ax.scatter(
# #     swe_points[:, 0],
# #     swe_points[:, 1],
# #     swe_points[:, 2],
# #     s=10
# # )

# # ax.set_xlabel(r"$f(t)$")
# # ax.set_ylabel(r"$f(t+\tau)$")
# # ax.set_zlabel(r"$f(t+2\tau)$")

# # ax.set_title("Adaptive Sliding Window Embedding")

# # plt.tight_layout()

# # # Save BEFORE plt.show()
# # plt.savefig(
# #     "testing_differing_tau.png",
# #     dpi=300,
# #     bbox_inches="tight"
# # )

# # plt.show()

# # print("Projected points saved as testing_differing_tau.png")