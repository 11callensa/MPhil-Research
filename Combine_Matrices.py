import random
import math
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point


def place_hydrogen_molecules(compound_bond, compound_xyz, hydrogen_bond, hydrogen_spacing):

    z_coordinates = [float(line.split()[-1]) for line in compound_xyz]                                                  # Extract z coordinates and find the maximum z-coordinate
    z_max = max(z_coordinates)

    surface_points = [line for line in compound_xyz if float(line.split()[-1]) == z_max]                                # Get surface points (those with the maximum z-coordinate)

    xy_coordinates = [(float(point.split()[1]), float(point.split()[2])) for point in surface_points]                   # Extract x and y coordinates of the surface points

    boundary_polygon = Polygon(xy_coordinates)                                                                          # Define the polygon boundary

    h2_z = z_max + hydrogen_spacing if hydrogen_spacing >= compound_bond[0] else compound_bond[0]                       # Define the z-coordinate for hydrogen atoms (just above the surface)

    H2_space = hydrogen_spacing if hydrogen_spacing >= hydrogen_bond else hydrogen_bond

    placed_molecules = []

    consistent_count = 0
    max_retries = 1000                                                                                                  # Add a maximum retry count to avoid infinite loops

    while consistent_count < 15:
        retries = 0                                                                                                     # Counter to track retries for this placement attempt

        while retries < max_retries:                                                                                    # Generate random center of gravity (COG) within bounds
            cog_x, cog_y = random.uniform(boundary_polygon.bounds[0], boundary_polygon.bounds[2]), \
                           random.uniform(boundary_polygon.bounds[1], boundary_polygon.bounds[3])
            center = Point(cog_x, cog_y)

            if not boundary_polygon.contains(center):
                retries += 1
                continue

            angle = random.uniform(0, 2 * math.pi)                                                                   # Generate a random direction (angle in radians)

            atom1_x = cog_x + (hydrogen_bond / 2) * math.cos(angle)                                                     # Calculate the positions of the two hydrogen atoms
            atom1_y = cog_y + (hydrogen_bond / 2) * math.sin(angle)
            atom2_x = cog_x - (hydrogen_bond / 2) * math.cos(angle)
            atom2_y = cog_y - (hydrogen_bond / 2) * math.sin(angle)

            if not (boundary_polygon.contains(Point(atom1_x, atom1_y)) and                                        # Check if both atoms are within the polygon boundary
                    boundary_polygon.contains(Point(atom2_x, atom2_y))):
                retries += 1
                continue

            valid = True                                                                                                # Check if the molecule satisfies the distance constraint with all existing molecules
            for molecule in placed_molecules:
                atom1_x_existing, atom1_y_existing, atom2_x_existing, atom2_y_existing = molecule

                distances = [                                                                                           # Calculate distances from both atoms of the new molecule to existing molecules
                    math.sqrt((atom1_x - atom1_x_existing) ** 2 + (atom1_y - atom1_y_existing) ** 2),
                    math.sqrt((atom1_x - atom2_x_existing) ** 2 + (atom1_y - atom2_y_existing) ** 2),
                    math.sqrt((atom2_x - atom1_x_existing) ** 2 + (atom2_y - atom1_y_existing) ** 2),
                    math.sqrt((atom2_x - atom2_x_existing) ** 2 + (atom2_y - atom2_y_existing) ** 2),
                ]

                if any(d < H2_space for d in distances):
                    valid = False
                    break

            if valid:                                                                                                   # If valid, add the molecule's atom coordinates to the list
                placed_molecules.append((atom1_x, atom1_y, atom2_x, atom2_y))
                break

            retries += 1

        if retries >= max_retries:                                                                                      # If too many retries were done without a successful placement, break the loop
            print("Maximum retries reached, stopping.")
            break

        if len(placed_molecules) > consistent_count:                                                                    # Check if the number of molecules remains consistent
            consistent_count = 0
        consistent_count += 1

    print("No. of molecules: ", len(placed_molecules))

    # fig = plt.figure()                                                                                                  # Plotting
    # ax = fig.add_subplot(111, projection='3d')
    #
    # surface_x = [point[0] for point in xy_coordinates]                                                                  # Plot the surface points
    # surface_y = [point[1] for point in xy_coordinates]
    # surface_z = [z_max] * len(surface_points)
    # ax.scatter(surface_x, surface_y, surface_z, c='blue', label='Surface')
    #
    # for atom1_x, atom1_y, atom2_x, atom2_y in placed_molecules:
    #     ax.scatter([atom1_x, atom2_x], [atom1_y, atom2_y], [h2_z, h2_z], c='red')                              # Plot the individual atoms
    #     ax.plot([atom1_x, atom2_x], [atom1_y, atom2_y], [h2_z, h2_z], c='red', linewidth=0.8)                     # Plot a line connecting the two atoms of the molecule
    #
    # ax.set_xlabel('X Coordinate')
    # ax.set_ylabel('Y Coordinate')
    # ax.set_zlabel('Z Coordinate')
    # ax.legend()
    # plt.title("Hydrogen Placement Above Surface")
    # plt.show()

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