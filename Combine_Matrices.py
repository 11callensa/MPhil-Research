import random
import math
import numpy as np
import matplotlib.pyplot as plt


def place_hydrogen_molecules(local_minima, surface_points, hydrogen_bond):
    """
        Creates and places hydrogen molecules comprised of a fixed centre of gravity on a local minimum
        and a random direction defined by an angle. The molecules are placed a fixed distance above the surface.

        :param local_minima: All mesh points where the energy was a local minimum.
        :param surface_points: Uppermost surface points.
        :param hydrogen_bond: Hydrogen bond length which helps define the molecule.
        :return: 3D coordinates of the placed hydrogen molecules above the surface of the compound.
    """

    # Extract z coordinates and find the maximum z-coordinate
    z_coordinates = surface_points[:, 2]
    z_max = np.max(z_coordinates) + 4                                                                                   # Could paramaterise the z-coordinate gap

    # Extract x and y coordinates of the surface points
    xy_coordinates = surface_points[:, :2]

    placed_molecules = []

    for position in local_minima:
        angle = random.uniform(0, 2 * math.pi)  # Generate a random direction (angle in radians)

        cog_x = position[0]
        cog_y = position[1]

        # Calculate the positions of the two hydrogen atoms
        atom1_x = cog_x + (hydrogen_bond / 2) * math.cos(angle)
        atom1_y = cog_y + (hydrogen_bond / 2) * math.sin(angle)
        atom2_x = cog_x - (hydrogen_bond / 2) * math.cos(angle)
        atom2_y = cog_y - (hydrogen_bond / 2) * math.sin(angle)

        placed_molecules.append((atom1_x, atom1_y, atom2_x, atom2_y))

    print("No. of molecules: ", len(placed_molecules))

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract surface points as separate arrays
    surface_x = xy_coordinates[:, 0]
    surface_y = xy_coordinates[:, 1]
    surface_z = [z_max] * len(surface_points)

    # Plot the surface points
    ax.scatter(surface_x, surface_y, surface_z, c='blue', label='Surface Points')

    # Add a plane joining the surface points
    ax.plot_trisurf(surface_x, surface_y, surface_z, color='cyan', alpha=0.5, label='Surface Plane')

    # Plot hydrogen molecule placements
    for atom1_x, atom1_y, atom2_x, atom2_y in placed_molecules:
        ax.scatter([atom1_x, atom2_x], [atom1_y, atom2_y], [z_max, z_max], c='red')
        ax.plot([atom1_x, atom2_x], [atom1_y, atom2_y], [z_max, z_max], c='red', linewidth=0.8)

    # Add labels and legend
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()
    plt.title("Hydrogen Placement Above Surface with Plane")
    plt.show()

    hydrogen_atoms = []

    for atom1_x, atom1_y, atom2_x, atom2_y in placed_molecules:
        hydrogen_atoms.append(f"H {atom1_x:.10f} {atom1_y:.10f} {z_max:.10f}")
        hydrogen_atoms.append(f"H {atom2_x:.10f} {atom2_y:.10f} {z_max:.10f}")

    return hydrogen_atoms


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