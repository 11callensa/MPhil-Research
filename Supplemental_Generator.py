from DFT_New import calculate_energy
from Compound_Properties import node_edge_features
from Mol_Geometry import centre_coords
from Build_Connections import build_connections
from Plotting import plot_crystal

import numpy as np
import csv


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


def generate_h2_coordinates(n_molecules, bond_length=0.708, box_size=10.0, seed=None, max_attempts=1000):
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


num_sims = 2

for i in range(num_sims):

    gen = 20
    xyz_coords = generate_h2_coordinates(gen)

    edge_indices = build_connections(xyz_coords, gen*2, 'H', 0)
    centered_xyz, _ = centre_coords(xyz_coords, gen*2)
    energy = calculate_energy(centered_xyz)

    node_features, edge_features = node_edge_features(centered_xyz, edge_indices, {'H': 0}, gen*2, 1)

    with open(filename, mode='a', newline='') as file1:  # Append mode.
        writer = csv.writer(file1)
        writer.writerow(
            [f'H_sim_{10}_by_{10}_{gen}_{i}', str([node_features]), str([edge_features]), str(edge_indices), str([0]),
             str([energy]), str([0]), str([0]), str(gen*2)])
