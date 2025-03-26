import numpy as np

from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def surface_finder(matrix):
    """
        Finds all external planes of the compound.

        :param matrix: The 3D coordinates of the compound.
        :return: A list of 3D coordinates defining each external plane.
    """

    elements = []
    coordinates = []

    for line in matrix:                                                                                                 # Extract coordinates.
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coordinates = np.array(coordinates)

    def find_all_outer_surfaces(coordinates):
        try:
            hull = ConvexHull(coordinates)                                                                              # Extract the convex hull.
            external_faces = []

            for simplex in hull.simplices:                                                                              # Extracting all external faces.
                face_points = coordinates[simplex]
                external_faces.append(face_points)

            return external_faces
        except Exception as e:
            print("Convex Hull failed, falling back to direct surface detection.")
            return []

    external_faces = find_all_outer_surfaces(coordinates)                                                               # Get all external faces.

    return external_faces


def compute_volume(matrix):
    """
        Computes the volume encapsulated by a set of 3D points using the convex hull.

        :param matrix: 3D coordinates of the system.
        :return: Volume encapsulated by the atoms.
    """

    elements = []
    coordinates = []

    for line in matrix:                                                                                                 # Extract the coordinates.
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    points = np.array(coordinates)
    hull = ConvexHull(points)                                                                                           # Compute convex hull.
    return hull.volume


def centre_coords(base_matrix, num_center):
    """
        Centres the system of atoms according to the center of the compound.

        :param base_matrix: Original 3D coordinates of all atoms in the system.
        :param num_center: The first num_center atoms from which to find the centroid.
        :return: Centered 3D coordinates of all atoms in the system.
    """

    coordinates = []                                                                                                    # Extract coordinates from input
    atom_labels = []
    for atom in base_matrix:
        parts = atom.split()
        atom_labels.append(parts[0])
        coordinates.append([float(x) for x in parts[1:]])

    coordinates = np.array(coordinates)

    center = np.mean(coordinates[:num_center], axis=0)                                                                  # Compute the geometric center (mean position).

    centered_coordinates = coordinates - center                                                                         # Subtract the center from each coordinate to center the crystal.

    centered_atom_list = []                                                                                             # Format the centered atoms back into strings.
    for i, coords in enumerate(centered_coordinates):
        centered_atom_list.append(f"{atom_labels[i]} {' '.join(f'{x:.10f}' for x in coords)}")

    return centered_atom_list, center


def place_hydrogen(matrix, surfaces, bond_length, offset1, offset2):
    """
        Places hydrogen molecules around the compound surfaces, at a certain offset distance from the surfaces.

        :param matrix: 3D coordinates of the compound.
        :param surfaces: Coordinates of the compound surfaces.
        :param bond_length: Bond length of H2.
        :param offset1: Surface offset distance 1.
        :param offset2: Surface offset distance 2.
        :return: 3D coordinates of the compound surrounded by H2 molecules.
    """

    fig = plt.figure(figsize=(10, 10))                                                                                  # Setup a 3D figure.
    ax = fig.add_subplot(111, projection='3d')

    all_points = np.vstack(surfaces)                                                                                    # Compute global centroid to determine outward direction.
    global_centroid = np.mean(all_points, axis=0)

    for i, surface in enumerate(surfaces):
        surface = np.array(surface)                                                                                     # Extract the surface information.
        ax.add_collection3d(Poly3DCollection([surface], alpha=0.5, edgecolor='k'))

        A, B, C = surface                                                                                               # Compute the edge lengths.
        edge1 = np.linalg.norm(A - B)
        edge2 = np.linalg.norm(B - C)
        edge3 = np.linalg.norm(C - A)
        avg_edge_length = (edge1 + edge2 + edge3) / 3

        if avg_edge_length < bond_length:                                                                               # Skip molecule placement if avg edge length is smaller than bond length.
            continue

        v1, v2 = B - A, C - A                                                                                           # Compute normal vector.
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        centroid = np.mean(surface, axis=0)                                                                             # Ensure normal points outward.
        if np.dot(normal, centroid - global_centroid) < 0:
            normal = -normal                                                                                            # Flip normal if pointing inward.

        offset_distance = offset2 if i % 2 == 0 else offset1

        P1 = centroid - normal * offset_distance                                                                        # Place the molecule's midpoint at the centroid of the surface.
        P2 = centroid + normal * offset_distance                                                                        # Molecule's center of gravity (midpoint of the bond) should be at centroid.

        P1_offset, P2_offset = P1, P2                                                                                   # Ensure the correct bond length.
        current_bond_length = np.linalg.norm(P2_offset - P1_offset)                                                     # Make sure the distance between P1 and P2 is exactly the bond length.

        if current_bond_length == 0:                                                                                    # Avoid division by zero.
            print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
            continue                                                                                                    # Skip this iteration if the bond length is zero.

        scale_factor = bond_length / current_bond_length
        P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor

        def is_within_surface(p, surface):                                                                              # Check if both atoms are within the surface's bounds.
            A, B, C = surface                                                                                           # Check if a point is within the triangle surface using barycentric coordinates.
            v0, v1, v2 = B - A, C - A, p - A                                                                            # Vectors from point to vertices.
            d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v0, v2), np.dot(v1, v2)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            return u >= 0 and v >= 0 and w >= 0

        while not is_within_surface(P1_offset, surface) or not is_within_surface(P2_offset, surface):                   # Retry until both atoms are within the surface.
            P1 = centroid - normal * offset_distance                                                                    # If atoms are out of bounds, adjust position.
            P2 = centroid + normal * offset_distance
            P1_offset, P2_offset = P1, P2
            current_bond_length = np.linalg.norm(P2_offset - P1_offset)

            if current_bond_length == 0:                                                                                # Avoid division by zero.
                print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
                continue                                                                                                # Skip this iteration if the bond length is zero.

            scale_factor = bond_length / current_bond_length
            P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor

        tangent = np.cross(normal, v1)                                                                                  # Compute a direction parallel to the surface (tangent direction).
        if np.linalg.norm(tangent) == 0:
            tangent = np.cross(normal, v2)                                                                              # If the first cross product is zero, use the other edge.
        tangent = tangent / np.linalg.norm(tangent) * bond_length                                                       # Scale to bond length.

        P1_offset = centroid - tangent / 2 + normal * offset_distance / 2                                               # Apply offset distance to ensure molecules are outside the surface.
        P2_offset = centroid + tangent / 2 + normal * offset_distance / 2

        ax.scatter(*P1_offset, color='r', s=50)                                                                         # Plot points and lines.
        ax.scatter(*P2_offset, color='b', s=50)
        ax.add_collection3d(Line3DCollection([[P1_offset, P2_offset]], colors='black', linewidths=2))

        matrix.append(f'H {P1_offset[0]:.10f} {P1_offset[1]:.10f} {P1_offset[2]:.10f}')
        matrix.append(f'H {P2_offset[0]:.10f} {P2_offset[1]:.10f} {P2_offset[2]:.10f}')

    ax.set_xticks(np.arange(int(np.min(all_points[:, 0])) - 2, int(np.max(all_points[:, 0])) + 2, 2))
    ax.set_yticks(np.arange(int(np.min(all_points[:, 1])) - 2, int(np.max(all_points[:, 1])) + 2, 2))
    ax.set_zticks(np.arange(int(np.min(all_points[:, 2])) - 2, int(np.max(all_points[:, 2])) + 2, 2))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    return matrix


def find_centroid(coords):
    return np.mean(coords, axis=0)


def find_direction(coords):
    """
        Finds the direction between the first two atoms in the compound.

        :param coords: 3D coordinates of the compound.
        :return: The direction vector between the first two compound atoms.
    """

    return coords[1] - coords[0]                                                                                        # Vector between first two atoms


def find_distances(coords, centroid):
    return np.linalg.norm(coords - centroid, axis=1)


def find_translation(new_coords, prev_centroid):
    """
        Finds the translation vector between the centroid of the previous compound and the current
        compound coordinates.

        :param new_coords: Compound atom coordinates.
        :param prev_centroid: Previous centroid of the compound in the last optimisation iteration.
        :return: Translation vector.
    """

    current_centroid = find_centroid(new_coords)
    return current_centroid - prev_centroid


def find_rotation(new_coords, prev_direction):
    """
        Finds the rotation axis and angle when compared with the previous ones.

        :param new_coords: 3D coordinates of the optimised compound.
        :param prev_direction: Previous direction vector between the two first atoms.
        :return: Rotation axis and angle.
    """

    new_direction = find_direction(new_coords)                                                                          # Find the new direction vector between the first two atoms.
    rotation_axis = np.cross(prev_direction, new_direction)                                                             # Calculate the rotation axis.

    if np.linalg.norm(rotation_axis) < 1e-6:                                                                            # Check if the axis is very small, implying collinearity.
        return None, 0                                                                                                  # No rotation needed if the vectors are co-linear.

    rotation_axis /= np.linalg.norm(rotation_axis)                                                                      # Normalize the rotation axis.
    angle = np.arccos(np.dot(prev_direction, new_direction) /
                      (np.linalg.norm(prev_direction) * np.linalg.norm(new_direction)))                                 # Calculate the angle of rotation.

    return rotation_axis, angle


def apply_translation(coords, trans_vec):
    """
        Translate the compound back to its original position.

        :param coords: 3D coordinates of the compound.
        :param trans_vec: Translation vector between previous and current compound coordinates.
        :return: Translated compound coordinates.
    """

    return coords + trans_vec


def apply_rotation(coords, rotation_axis, angle):
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(angle * rotation_axis)
    return rot.apply(coords)
