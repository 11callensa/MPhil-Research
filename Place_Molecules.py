import math
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from DFT import calculate_energy


def combine_matrices(compound_matrix, hydrogen_matrix):
    """
        Combines the reoriented compound matrix and the locations of the hydrogen molecules
        to make one large matrix ready for DFT.

        :param compound_matrix: 3D compound coordinates.
        :param hydrogen_matrix: 3D hydrogen molecule coordinates.
        :return: A merged matrix of the compound and hydrogen coordinates.
    """

    combined_matrix = compound_matrix.copy()
    combined_matrix.extend(hydrogen_matrix)

    elements = []
    coordinates = []

    for line in combined_matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return combined_matrix


def placement_z(local_minima, binding_sites, energies, compound_matrix, surface_points, hydrogen_bond):
    """
    Places hydrogen molecules on local minima with one atom directed toward the nearest binding site with the next
    lowest energy.

    :param local_minima: List of local minimum positions.
    :param binding_sites: Array of binding site positions.
    :param energies: Array of energies corresponding to the binding sites.
    :param compound_matrix: 3D coordinates of the base compound.
    :param surface_points: Array of surface points of the material.
    :param hydrogen_bond: Hydrogen bond length to define the molecule.
    :return: List of 3D coordinates of the placed hydrogen molecules.
    """

    z_surface = surface_points[:, 2]
    z_max = np.max(z_surface)
    z_positions = z_max + np.arange(0.1, 15, 0.1)

    # placed_molecules = []
    hydrogen_final = []

    for i, position in enumerate(local_minima):
        cog_x, cog_y = position[0], position[1]

        # Calculate distances to all binding sites
        distances = np.linalg.norm(binding_sites[:, :2] - position[:2], axis=1)

        # Find the binding site with the lowest energy (excluding current local minimum)
        valid_indices = distances > 0  # Exclude the current site itself
        if not np.any(valid_indices):
            continue  # Skip if no valid binding sites
        next_site_index = np.argmin(energies[valid_indices])
        nearest_site = binding_sites[valid_indices][next_site_index]

        # Calculate the angle toward the nearest binding site
        dx = nearest_site[0] - cog_x
        dy = nearest_site[1] - cog_y
        angle = math.atan2(dy, dx)

        # Calculate the positions of the two hydrogen atoms
        atom1_x = cog_x + (hydrogen_bond / 2) * math.cos(angle)
        atom1_y = cog_y + (hydrogen_bond / 2) * math.sin(angle)
        atom2_x = cog_x - (hydrogen_bond / 2) * math.cos(angle)
        atom2_y = cog_y - (hydrogen_bond / 2) * math.sin(angle)

        energies_optimised = []
        z_list = []
        hydrogen_atoms = []

        for z_coord in tqdm(z_positions, desc=f"Computing energies (Z Placement) - for point {i + 1} out of "
                                              f"{len(local_minima)}", unit="point offset"):

            print("Z coordinate: ", z_coord)

            try:
                hydrogen_atoms.append(f"H {atom1_x:.10f} {atom1_y:.10f} {z_coord:.10f}")
                hydrogen_atoms.append(f"H {atom2_x:.10f} {atom2_y:.10f} {z_coord:.10f}")

                combined_matrix = combine_matrices(compound_matrix, hydrogen_atoms)
                energy = calculate_energy(combined_matrix)

                energies_optimised.append(energy)
                z_list.append(z_coord)

                hydrogen_atoms = []

            except Exception as e:
                print(f"Error at z = {z_coord}: {e}")
                break  # Stop processing further z-coordinates for this point

        print("Z list: ", z_list)
        print("Optimised Energies: ", energies_optimised)

        plt.figure(figsize=(8, 6))  # Set the figure size
        plt.plot(z_list, energies_optimised, marker='o', linestyle='-', color='b', label="Total Energy")
        plt.xlabel("z-coordinate (Å)", fontsize=12)  # X-axis label
        plt.ylabel("Total Energy (eV)", fontsize=12)  # Y-axis label
        plt.title("Total Energy vs. z-coordinate", fontsize=14)  # Title of the plot
        plt.grid(True)  # Add a grid for better readability
        plt.legend(fontsize=10)  # Add legend
        plt.show()  # Display the plot

        # Use available data to find the minimum energy
        if energies_optimised:
            min_position, min_energy = min(zip(z_list, energies_optimised), key=lambda x: x[1])
            print(f"Min position for point {i + 1}: z = {min_position}, energy = {min_energy}")

            hydrogen_final.append(f"H {atom1_x:.10f} {atom1_y:.10f} {min_position:.10f}")
            hydrogen_final.append(f"H {atom2_x:.10f} {atom2_y:.10f} {min_position:.10f}")

    print("Hydrogen Final: ", hydrogen_final)

    symbols = [line.split()[0] for line in hydrogen_final]
    num_molecules = len(symbols)/2

    coordinates = [
        [round(float(coord), 4) for coord in atom.split()[1:]]
        for atom in hydrogen_final
    ]

    # Group coordinates into pairs
    paired_coordinates = [coordinates[i:i + 2] for i in range(0, len(coordinates), 2)]

    return hydrogen_final, num_molecules, paired_coordinates
