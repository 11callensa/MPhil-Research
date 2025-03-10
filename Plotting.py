import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_crystal(positions, edge_indices):
    # Parse positions into atom coordinates
    atom_coordinates = []
    for position in positions:
        parts = position.split()  # Split each line into element and coordinates
        atom_coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))  # 3D coordinates

    # Convert atom_coordinates to a numpy array for easier manipulation
    atom_coordinates = np.array(atom_coordinates)

    # Set up the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms as scatter points (different colors for clarity)
    ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c='b', marker='o', label='Atoms')

    # Plot bonds as lines between the atoms defined by edge_indices
    for bond in edge_indices:
        i, j = bond
        ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                [atom_coordinates[i, 2], atom_coordinates[j, 2]], c='r', linestyle='-', linewidth=1)

    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Atoms and Bonds')

    # Show plot
    plt.show()


def plot_external_surfaces(external_faces):
    """
    Plots the external planes found by `surface_finder` in 3D.

    :param external_faces: List of external surfaces, where each surface is a set of 3D points.
    """

    # Create 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Iterate over each external face and plot it
    for face in external_faces:
        face = np.array(face)  # Convert to NumPy array for easy manipulation
        ax.add_collection3d(Poly3DCollection([face], alpha=0.5, edgecolor="k"))

    # Scatter plot of the points (for reference)
    all_points = np.vstack(external_faces)  # Stack all face points together
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c="r", marker="o")

    # Labels and view adjustments
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("External Surfaces of the Crystal")

    plt.show()