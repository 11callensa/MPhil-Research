from DFT_New import calculate_energy
from Compound_Properties import node_edge_features
from Mol_Geometry import centre_coords
from Plotting import plot_crystal

import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


filename = 'H_training_supplement.csv'

file_data = [(filename, ['Compounds', 'Node Features', 'Edge Features',
                               'Edge Indices', 'Energy Input Features',
                               'Energy Output Features', 'Uncertain Features', 'Num. Fixed Atoms',
                               'Num. Placed H Atoms'])]


for filename, header in file_data:
    try:
        with open(filename, mode='x', newline='') as file:  # Try creating the file in exclusive ('x') mode
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header if the file is new
            print(f"{filename} setup complete (new file created).")

    except FileExistsError:
        print(f"File '{filename}' already exists. Appended new row.")


def build_connections(positions, save=1):
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

    tree_hydrogen = KDTree(atom_coordinates)

    edge_indices = []

    for i in range(len(atom_coordinates)):                                                                              # Find n nearest hydrogen neighbors for each hydrogen atom.
        k_h = min(7, len(atom_coordinates))
        _, neighbor_indices = tree_hydrogen.query(atom_coordinates[i], k=k_h)
        # Get n (includes self).
        for neighbor in neighbor_indices[1:]:                                                                           # Skip self (first index).
            edge = tuple(sorted([i, neighbor]))                                                  # Undirected.
            if edge not in edge_indices:
                edge_indices.append(edge)

    def draw_plot():                                                                                                    # Function to visualize connections.
        ax.clear()

        colors = ['black' for i in range(len(atom_coordinates))]                            # Color compound atoms black, hydrogen atoms red.
        ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c=colors, marker='o')

        for i, (x, y, z) in enumerate(atom_coordinates):                                                                # Label atoms.
            ax.text(x, y, z, str(i), color='black')

        for bond in edge_indices:                                                                                       # Draw bonds.
            i, j = bond
            bond_color = 'b'                                                # Green = compound bonds, Blue = hydrogen bonds
            ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                    [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                    [atom_coordinates[i, 2], atom_coordinates[j, 2]], c=bond_color)

        plt.draw()

    fig = plt.figure()                                                                                                  # Plot the updated structure.
    ax = fig.add_subplot(111, projection='3d')

    if save == 1:
        draw_plot()
        plt.show()
    else:
        pass

    return edge_indices


def generate_h2_coordinates(n_molecules, bond_length=0.708, box_size=10, seed=None, max_attempts=1000):
    if seed is not None:
        np.random.seed(seed)

    all_atoms = []
    atom_count = 1
    min_distance = 1 * bond_length  # 1.416 Å

    def is_valid(new_atoms, existing_atoms):
        for na in new_atoms:
            for ea in existing_atoms:
                if np.linalg.norm(np.array(na) - np.array(ea)) < min_distance:
                    return False
        return True

    for _ in range(n_molecules):
        for attempt in range(max_attempts):
            # Generate random origin
            origin = np.random.uniform(0, box_size, size=3)

            # Random direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)

            partner = origin + bond_length * direction

            new_atoms = [origin, partner]

            if is_valid(new_atoms, all_atoms):
                all_atoms.extend(new_atoms)
                break
        else:
            raise RuntimeError(f"Failed to place molecule {_+1} without violating distance constraints.")

    # Format output
    formatted = []
    for i, coords in enumerate(all_atoms, start=1):
        formatted.append(f'H {coords[0]:.4f} {coords[1]:.4f} {coords[2]:.4f}')

    return formatted


num_sims = 1

for i in range(num_sims):

    gen = 2
    xyz_coords = generate_h2_coordinates(gen)

    edge_indices = build_connections(xyz_coords, 0)

    centered_xyz, _ = centre_coords(xyz_coords, gen*2)
    energy = calculate_energy(centered_xyz)

    node_features, edge_features = node_edge_features(centered_xyz, edge_indices, {'H': 0}, gen*2, 1)

    plot_crystal(centered_xyz, edge_indices)

    with open(filename, mode='a', newline='') as file1:  # Append mode.
        writer = csv.writer(file1)
        writer.writerow(
            [f'H_sim_{40}_by_{40}_{gen}_{i}', str([node_features]), str([edge_features]), str(edge_indices), str([0]),
             str([energy]), str([0]), str([0]), str(gen*2)])
