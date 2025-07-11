import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_crystal(positions, edge_indices):
    """
        Plots a system of atoms and bonds in 3D space with a legend.

        :param positions: 3D coordinates of the system.
        :param edge_indices: Bonds between pairs of atoms.
    """

    atom_coordinates = []

    for position in positions:
        parts = position.split()
        atom_coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    atom_coordinates = np.array(atom_coordinates)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot atoms and assign the scatter object to a variable
    scatter = ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2],
                         c='r', marker='o', label='Atoms')

    # Plot bonds
    for bond in edge_indices:
        i, j = bond
        ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                [atom_coordinates[i, 2], atom_coordinates[j, 2]],
                c='b', linestyle='-', linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Atoms and Bonds')

    # Create custom legend handles
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Hydrogen Atoms',
               markerfacecolor='r', markeredgecolor='k', markersize=8),
        Line2D([0], [0], color='b', lw=2, label='Hydrogen Bonds')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

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


def plot_adsorption_sites(layer_atoms, edge_indices, adsorption_sites_dict):
    """
    Plots the compound surface layer with identified adsorption sites.

    Parameters:
    - layer_atoms: list of strings like ['Ti x y z', ...]
    - edge_indices: list of tuples of bonded atom indices
    - adsorption_sites_dict: output from identify_adsorption_sites(...)
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Parse atom coordinates
    coords = []
    elements = []
    for line in layer_atoms:
        parts = line.split()
        elements.append(parts[0])
        coords.append(np.array(list(map(float, parts[1:]))))

    coords = np.array(coords)

    # Plot atoms
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c='blue', label='Bulk Atoms', s=50)

    # Plot bonds
    for i, j in edge_indices:
        bond = np.array([coords[i], coords[j]])
        ax.plot(bond[:, 0], bond[:, 1], bond[:, 2], color='gray', alpha=0.6)

    # Plot adsorption sites
    colors = {'top': 'green', 'bridge': 'orange', 'hollow': 'red'}
    markers = {'top': 'o', 'bridge': '^', 'hollow': 's'}

    for site_type, site_coords in adsorption_sites_dict.items():
        site_coords = np.array(site_coords)
        if len(site_coords) > 0:
            ax.scatter(site_coords[:, 0], site_coords[:, 1], site_coords[:, 2],
                       c=colors[site_type], marker=markers[site_type],
                       label=f'{site_type.title()} Sites', s=80, edgecolor='k')

    ax.legend()
    ax.set_title("Adsorption Sites on Surface")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def plot_adsorbed_atoms(atom_list, title="Surface + Adsorbed H₂"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = [], [], []
    colors = []
    for entry in atom_list:
        element, x, y, z = entry.split()
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
        colors.append('red' if element == 'H' else 'dodgerblue')

    ax.scatter(xs, ys, zs, c=colors, s=60, edgecolor='k', depthshade=True)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 0.3])
    plt.tight_layout()
    plt.show()
