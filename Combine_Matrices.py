import random
import math
import matplotlib.pyplot as plt
import pandas as pd


def place_hydrogen_molecules(dataframe):

    compound_row = dataframe.loc[0]
    compound_bond = compound_row['Bond Lengths']  # Li4H4 bond length
    base_matrix = compound_row['XYZ']

    # Extract z coordinates and find the maximum z-coordinate
    z_coordinates = [float(line.split()[-1]) for line in base_matrix]
    z_max = max(z_coordinates)

    # Get surface points (those with the maximum z-coordinate)
    surface_points = [line for line in base_matrix if float(line.split()[-1]) == z_max]

    # Extract x and y coordinates of the surface points
    xy_coordinates = [(float(point.split()[1]), float(point.split()[2])) for point in surface_points]

    # Determine the boundaries for x and y
    x_values = [coord[0] for coord in xy_coordinates]
    y_values = [coord[1] for coord in xy_coordinates]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Define the z-coordinate for hydrogen atoms (just above the surface)
    h2_z = z_max + 1.2*(compound_bond[0])

    hydrogen_row = dataframe.loc[1]
    hydrogen_bond = hydrogen_row['Bond Lengths']  # Hydrogen bond length

    placed_molecules = []

    # Initialize a counter for the consistency check
    consistent_count = 0
    max_retries = 1000  # Add a maximum retry count to avoid infinite loops

    while consistent_count < 15:
        retries = 0  # Counter to track retries for this placement attempt

        while retries < max_retries:
            # Generate random center of gravity (COG) within bounds
            cog_x = random.uniform(x_min + hydrogen_bond / 2, x_max - hydrogen_bond / 2)
            cog_y = random.uniform(y_min + hydrogen_bond / 2, y_max - hydrogen_bond / 2)

            # Generate a random direction (angle in radians)
            angle = random.uniform(0, 2 * math.pi)

            # Calculate the positions of the two hydrogen atoms
            atom1_x = cog_x + (hydrogen_bond / 2) * math.cos(angle)
            atom1_y = cog_y + (hydrogen_bond / 2) * math.sin(angle)
            atom2_x = cog_x - (hydrogen_bond / 2) * math.cos(angle)
            atom2_y = cog_y - (hydrogen_bond / 2) * math.sin(angle)

            # Check if both atoms are within bounds
            if not (x_min <= atom1_x <= x_max and y_min <= atom1_y <= y_max and
                    x_min <= atom2_x <= x_max and y_min <= atom2_y <= y_max):
                retries += 1
                continue

            # Check if the molecule satisfies the distance constraint with all existing molecules
            valid = True
            for molecule in placed_molecules:
                atom1_x_existing, atom1_y_existing, atom2_x_existing, atom2_y_existing = molecule

                # Calculate distances from both atoms of the new molecule to existing molecules
                distances = [
                    math.sqrt((atom1_x - atom1_x_existing) ** 2 + (atom1_y - atom1_y_existing) ** 2),
                    math.sqrt((atom1_x - atom2_x_existing) ** 2 + (atom1_y - atom2_y_existing) ** 2),
                    math.sqrt((atom2_x - atom1_x_existing) ** 2 + (atom2_y - atom1_y_existing) ** 2),
                    math.sqrt((atom2_x - atom2_x_existing) ** 2 + (atom2_y - atom2_y_existing) ** 2),
                ]

                if any(d < hydrogen_bond for d in distances):
                    valid = False
                    break

            # If valid, add the molecule's atom coordinates to the list
            if valid:
                placed_molecules.append((atom1_x, atom1_y, atom2_x, atom2_y))
                break

            retries += 1

        # If too many retries were done without a successful placement, break the loop
        if retries >= max_retries:
            print("Maximum retries reached, stopping.")
            break

        # Check if the number of molecules remains consistent
        if len(placed_molecules) > consistent_count:
            consistent_count = 0
        consistent_count += 1

    print("No. of molecules: ", len(placed_molecules))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the surface points
    # surface_x = [point[0] for point in xy_coordinates]
    # surface_y = [point[1] for point in xy_coordinates]
    # surface_z = [z_max] * len(surface_points)
    # ax.scatter(surface_x, surface_y, surface_z, c='blue', label='Surface')
    #
    # # Plot the hydrogen molecules
    # for atom1_x, atom1_y, atom2_x, atom2_y in placed_molecules:
    #     # Plot the individual atoms
    #     ax.scatter([atom1_x, atom2_x], [atom1_y, atom2_y], [h2_z, h2_z], c='red')
    #     # Plot a line connecting the two atoms of the molecule
    #     ax.plot([atom1_x, atom2_x], [atom1_y, atom2_y], [h2_z, h2_z], c='red', linewidth=0.8)
    #
    # # Labeling and legend
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.legend()
    # plt.show()

    # Format the output
    hydrogen_atoms = []
    for atom1_x, atom1_y, atom2_x, atom2_y in placed_molecules:
        hydrogen_atoms.append(f"H {atom1_x:.10f} {atom1_y:.10f} {h2_z:.10f}")
        hydrogen_atoms.append(f"H {atom2_x:.10f} {atom2_y:.10f} {h2_z:.10f}")

    return hydrogen_atoms, len(placed_molecules)


def combine_matrices(compound_matrix, hydrogen_matrix):

    combined_matrix = compound_matrix.copy()
    combined_matrix.extend(hydrogen_matrix)

    elements = []
    coordinates = []

    for line in combined_matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return combined_matrix