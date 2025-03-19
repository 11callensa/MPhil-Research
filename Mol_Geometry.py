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

    for line in matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coordinates = np.array(coordinates)

    def find_all_outer_surfaces(coordinates):
        try:
            hull = ConvexHull(coordinates)
            external_faces = []

            # Extracting all external faces
            for simplex in hull.simplices:
                face_points = coordinates[simplex]
                external_faces.append(face_points)

            return external_faces
        except Exception as e:
            print("Convex Hull failed, falling back to direct surface detection.")
            return []

    # Get all external faces
    external_faces = find_all_outer_surfaces(coordinates)

    return external_faces


def compute_volume(matrix):
    """
    Computes the volume encapsulated by a set of 3D points using the convex hull.

    Parameters:
    points (list of lists or np.ndarray): Nx3 array of points in 3D space.

    Returns:
    float: Volume of the convex hull.
    """
    elements = []
    coordinates = []

    for line in matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    points = np.array(coordinates)  # Ensure it's a NumPy array
    hull = ConvexHull(points)  # Compute convex hull
    return hull.volume  # Return the volume


def centre_coords(base_matrix, num_center):

    """
    Centers a crystal's atoms around (0, 0, 0).

    Parameters:
        base_matrix (list of str): Each element is a string representing an atom
                                with its element type and 3D coordinates.

    Returns:
        list of str: A list of strings representing the centered atoms.
    """
    # Extract coordinates from input
    coordinates = []
    atom_labels = []
    for atom in base_matrix:
        parts = atom.split()
        atom_labels.append(parts[0])
        coordinates.append([float(x) for x in parts[1:]])

    # Convert to a NumPy array
    coordinates = np.array(coordinates)

    # Compute the geometric center (mean position)
    center = np.mean(coordinates[:num_center], axis=0)

    # Subtract the center from each coordinate to center the crystal
    centered_coordinates = coordinates - center

    # Format the centered atoms back into strings
    centered_atom_list = []
    for i, coords in enumerate(centered_coordinates):
        centered_atom_list.append(f"{atom_labels[i]} {' '.join(f'{x:.10f}' for x in coords)}")

    return centered_atom_list, center


def place_hydrogen(matrix, surfaces, bond_length, offset1=6, offset2=5):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute global centroid to determine outward direction
    all_points = np.vstack(surfaces)
    global_centroid = np.mean(all_points, axis=0)

    for i, surface in enumerate(surfaces):
        surface = np.array(surface)
        ax.add_collection3d(Poly3DCollection([surface], alpha=0.5, edgecolor='k'))

        # Compute the edge lengths
        A, B, C = surface
        edge1 = np.linalg.norm(A - B)
        edge2 = np.linalg.norm(B - C)
        edge3 = np.linalg.norm(C - A)
        avg_edge_length = (edge1 + edge2 + edge3) / 3

        # Skip molecule placement if avg edge length is smaller than bond length
        if avg_edge_length < bond_length:
            continue

        # Compute normal vector
        v1, v2 = B - A, C - A
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # Ensure normal points outward
        centroid = np.mean(surface, axis=0)
        if np.dot(normal, centroid - global_centroid) < 0:
            normal = -normal  # Flip normal if pointing inward

        offset_distance = offset2 if i % 2 == 0 else offset1

        # Place the molecule's midpoint at the centroid of the surface
        # Molecule's center of gravity (midpoint of the bond) should be at centroid
        P1 = centroid - normal * offset_distance
        P2 = centroid + normal * offset_distance

        # Ensure the correct bond length
        # Make sure the distance between P1 and P2 is exactly the bond length
        P1_offset, P2_offset = P1, P2
        current_bond_length = np.linalg.norm(P2_offset - P1_offset)

        if current_bond_length == 0:  # Avoid division by zero
            print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
            continue  # Skip this iteration if the bond length is zero

        scale_factor = bond_length / current_bond_length
        P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor

        # Now we need to check if both atoms are within the surface's bounds.
        def is_within_surface(p, surface):
            # Check if a point is within the triangle surface using barycentric coordinates
            A, B, C = surface
            # Vectors from point to vertices
            v0, v1, v2 = B - A, C - A, p - A
            d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v0, v2), np.dot(v1, v2)
            denom = d00 * d11 - d01 * d01
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            return u >= 0 and v >= 0 and w >= 0

        # Retry until both atoms are within the surface
        while not is_within_surface(P1_offset, surface) or not is_within_surface(P2_offset, surface):
            # If atoms are out of bounds, adjust position
            P1 = centroid - normal * offset_distance
            P2 = centroid + normal * offset_distance
            P1_offset, P2_offset = P1, P2
            current_bond_length = np.linalg.norm(P2_offset - P1_offset)

            if current_bond_length == 0:  # Avoid division by zero
                print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
                continue  # Skip this iteration if the bond length is zero

            scale_factor = bond_length / current_bond_length
            P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor

        # Compute a direction parallel to the surface (tangent direction)
        tangent = np.cross(normal, v1)  # Tangent to the surface
        if np.linalg.norm(tangent) == 0:
            tangent = np.cross(normal, v2)  # If the first cross product is zero, use the other edge
        tangent = tangent / np.linalg.norm(tangent) * bond_length  # Scale to bond length

        # Apply offset distance to ensure molecules are outside the surface
        P1_offset = centroid - tangent / 2 + normal * offset_distance / 2
        P2_offset = centroid + tangent / 2 + normal * offset_distance / 2

        # Plot points and lines
        ax.scatter(*P1_offset, color='r', s=50)
        ax.scatter(*P2_offset, color='b', s=50)
        ax.add_collection3d(Line3DCollection([[P1_offset, P2_offset]], colors='black', linewidths=2))

        matrix.append(f'H {P1_offset[0]:.10f} {P1_offset[1]:.10f} {P1_offset[2]:.10f}')
        matrix.append(f'H {P2_offset[0]:.10f} {P2_offset[1]:.10f} {P2_offset[2]:.10f}')

    # Set tick marks every 2 units
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
    return coords[1] - coords[0]  # Vector between first two atoms


def find_distances(coords, centroid):
    return np.linalg.norm(coords - centroid, axis=1)


def find_translation(new_coords, prev_centroid):
    current_centroid = find_centroid(new_coords)
    return current_centroid - prev_centroid


def find_rotation(new_coords, prev_direction):
    new_direction = find_direction(new_coords)
    rotation_axis = np.cross(prev_direction, new_direction)

    # Check if the axis is very small, implying collinearity
    if np.linalg.norm(rotation_axis) < 1e-6:
        return None, 0  # No rotation needed if the vectors are collinear

    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize
    angle = np.arccos(np.dot(prev_direction, new_direction) /
                      (np.linalg.norm(prev_direction) * np.linalg.norm(new_direction)))
    return rotation_axis, angle


def apply_translation(coords, trans_vec):
    return coords + trans_vec


def apply_rotation(coords, rotation_axis, angle):
    from scipy.spatial.transform import Rotation as R
    rot = R.from_rotvec(angle * rotation_axis)
    return rot.apply(coords)
