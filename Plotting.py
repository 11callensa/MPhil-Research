import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_crystal(positions, edge_indices):
    """
        Plots a system of atoms and bonds in 3D space.

        :param positions: 3D coordinates of the system.
        :param edge_indices: Bonds between pairs of atoms.
    """

    atom_coordinates = []                                                                                               # Parse positions into atom coordinates

    for position in positions:
        parts = position.split()                                                                                        # Split each line into element and coordinates
        atom_coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    atom_coordinates = np.array(atom_coordinates)                                                                       # Convert atom_coordinates to a numpy array for easier manipulation.

    fig = plt.figure()                                                                                                  # Set up the plot.
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c='b', marker='o', label='Atoms')# Plot atoms as scatter points (different colors for clarity).

    for bond in edge_indices:                                                                                           # Plot bonds as lines between the atoms defined by edge_indices.
        i, j = bond
        ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                [atom_coordinates[i, 2], atom_coordinates[j, 2]], c='r', linestyle='-', linewidth=1)

    ax.set_xlabel('X')                                                                                                  # Labels and title.
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Atoms and Bonds')

    plt.show()


def plot_external_surfaces(external_faces):
    """
        Plots the external planes found by `surface_finder` in 3D.

        :param external_faces: List of external surfaces, where each surface is a set of 3D points.
    """

    fig = plt.figure()                                                                                                  # Create 3D figure.
    ax = fig.add_subplot(111, projection="3d")

    for face in external_faces:                                                                                         # Iterate over each external face and plot it.
        face = np.array(face)
        ax.add_collection3d(Poly3DCollection([face], alpha=0.5, edgecolor="k"))

    all_points = np.vstack(external_faces)                                                                              # Stack all face points together.
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c="r", marker="o")                                 # Scatter plot of the points.

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("External Surfaces of the Crystal")

    plt.show()