import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def load_existing_edges(name):
    """
        Load existing edges from the CSV file.

        :return: The edge indices from the file.
    """

    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    if not os.path.exists(CSV_FILE):
        print("Does not exist")
        return []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        return [list(map(int, row)) for row in reader]


def save_edges_to_csv(edges, name):
    """
        Save the updated edge list to the CSV file.
    """

    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(edges)


def build_connections(positions, num_fixed, name, save=1):
    """
        Build connections for a crystal structure. It keeps the original compound edges unchanged while connecting
        each hydrogen atom to n nearest hydrogen atoms.It also connects each hydrogen atom to at least m nearest
        compound atoms.

        :param positions: List of atomic positions as strings.
        :param num_fixed: Number of atoms in the compound (fixed structure).
        :param name: Name used to load/save edge indices.
        :return: Updated edge indices.
    """

    atom_coordinates = np.array([list(map(float, pos.split()[1:])) for pos in positions])                               # Parse positions into numpy array.

    edge_indices = load_existing_edges(name)                                                                            # Load existing edges.

    compound_coords = atom_coordinates[:num_fixed]                                                                      # Separate compound atoms and hydrogen atoms.
    hydrogen_coords = atom_coordinates[num_fixed:]

    tree_compound = KDTree(compound_coords)                                                                             # Create KDTree for nearest neighbor search.
    tree_hydrogen = KDTree(hydrogen_coords)

    for i in range(len(hydrogen_coords)):                                                                               # Find n nearest hydrogen neighbors for each hydrogen atom.
        global_index = num_fixed + i                                                                                    # Convert hydrogen index to global index.
        k_h = min(7, len(hydrogen_coords))
        if k_h > 1:
            _, neighbor_indices = tree_hydrogen.query(hydrogen_coords[i], k=k_h)
            if isinstance(neighbor_indices, int):  # In case only one neighbor is returned
                neighbor_indices = [neighbor_indices]
            for neighbor in neighbor_indices:
                if neighbor == i: continue  # Skip self
                if neighbor < 0 or neighbor >= len(hydrogen_coords):
                    continue  # Skip invalid neighbor
                global_neighbor = num_fixed + neighbor
                if global_neighbor >= len(atom_coordinates):  # Safety check
                    continue
                edge = tuple(sorted([global_index, global_neighbor]))
                if edge not in edge_indices:
                    edge_indices.append(edge)

    for i in range(len(hydrogen_coords)):                                                                               # Find at least m nearest compound atoms for each hydrogen atom.
        global_index = num_fixed + i
        k_c = min(7, len(compound_coords))
        _, neighbor_indices = tree_compound.query(hydrogen_coords[i], k=k_c)
        if isinstance(neighbor_indices, int):
            neighbor_indices = [neighbor_indices]
        for neighbor in neighbor_indices:
            if neighbor < 0 or neighbor >= len(compound_coords):
                continue
            global_neighbor = neighbor
            if global_index >= len(atom_coordinates) or global_neighbor >= len(atom_coordinates):
                continue
            edge = tuple(sorted([global_index, global_neighbor]))
            if edge not in edge_indices:
                edge_indices.append(edge)

    def draw_plot():                                                                                                    # Function to visualize connections.
        ax.clear()

        colors = ['black' if i < num_fixed else 'red' for i in range(len(atom_coordinates))]                            # Color compound atoms black, hydrogen atoms red.
        ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c=colors, marker='o')

        for i, (x, y, z) in enumerate(atom_coordinates):                                                                # Label atoms.
            ax.text(x, y, z, str(i), color='black')

        for bond in edge_indices:                                                                                       # Draw bonds.
            i, j = bond
            bond_color = 'g' if i < num_fixed and j < num_fixed else 'b'                                                # Green = compound bonds, Blue = hydrogen bonds
            ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                    [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                    [atom_coordinates[i, 2], atom_coordinates[j, 2]], c=bond_color)

        plt.draw()

    fig = plt.figure()                                                                                                  # Plot the updated structure.
    ax = fig.add_subplot(111, projection='3d')

    # Filter invalid edges
    edge_indices = [edge for edge in edge_indices if
                    edge[0] < len(atom_coordinates) and edge[1] < len(atom_coordinates)]

    if save == 1:
        draw_plot()
        plt.show()
    else:
        pass

    if save == 1:
        save_edges_to_csv(edge_indices, name)                                                                               # Save updated edges.
        return edge_indices
    else:
        return edge_indices
