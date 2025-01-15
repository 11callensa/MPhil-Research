import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot(matrix):
    elements = []
    coordinates = []

    # Parse the matrix to extract elements and coordinates
    for line in matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coordinates = np.array(coordinates)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get unique elements
    unique_elements = set(elements)

    # Use a colormap to assign a unique color to each element
    cmap = cm.get_cmap('tab20', len(unique_elements))  # Choose a colormap, tab20 for distinct colors
    colors = {elem: cmap(i) for i, elem in enumerate(unique_elements)}

    # Scatter plot for each element type
    for elem in unique_elements:
        idx = [i for i, e in enumerate(elements) if e == elem]
        coords_elem = coordinates[idx]
        ax.scatter(coords_elem[:, 0], coords_elem[:, 1], coords_elem[:, 2],
                   color=colors[elem], label=elem, s=50)

    # Compute the convex hull to find the outer surfaces
    hull = ConvexHull(coordinates)

    # Identify the uppermost surface
    uppermost_surface_points = None
    max_avg_z = -np.inf

    for simplex in hull.simplices:
        # Get the points for the face
        face_points = coordinates[simplex]
        avg_z = np.mean(face_points[:, 2])

        # Update if this face has the highest average Z-coordinate
        if avg_z > max_avg_z:
            max_avg_z = avg_z
            uppermost_surface_points = face_points

    # Plot the plane for the uppermost surface
    if uppermost_surface_points is not None:
        verts = [uppermost_surface_points]
        ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.5, edgecolor='k'))

    # Labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)

    plt.legend()
    plt.show()


def surface_finder(matrix):
    """
        Finds the largest surface of the compound. This will be assumed as the surface
        the molecules will adsorb to in ideal conditions.

        :param matrix: The 3D coordinates of the compound.
        :return: The 3D coordinates of 3 points that bound the largest surface.
    """

    elements = []
    coordinates = []

    for line in matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coordinates = np.array(coordinates)

    def find_largest_outer_surface(coordinates):
        try:
            # Attempt to compute the convex hull of all points
            hull = ConvexHull(coordinates)
            max_area = 0
            largest_face = None
            normal_vector = None
            point_on_plane = None

            # Iterate over all faces (simplices) in the convex hull
            for simplex in hull.simplices:
                face_points = coordinates[simplex]

                # Calculate the area of the face (using the first 3 points of the simplex)
                v1 = face_points[1] - face_points[0]
                v2 = face_points[2] - face_points[0]
                area = 0.5 * np.linalg.norm(np.cross(v1, v2))

                # Calculate the average Z-coordinate for the face
                avg_z = np.mean(face_points[:, 2])

                # Check if this face has the largest area
                if area > max_area:
                    max_area = area
                    largest_face = face_points

                    # Calculate normal vector and point on plane for the largest face
                    normal_vector = np.cross(v1, v2)
                    point_on_plane = face_points[0]
            if largest_face is not None:
                d = -np.dot(normal_vector, point_on_plane)
                return largest_face, normal_vector, d
        except Exception as e:
            # If convex hull fails (e.g., flat geometry), use fallback approach
            print("Convex Hull failed, falling back to direct surface detection.")

            # Fallback logic: identify uppermost surface by maximum z-coordinate
            max_z = np.max(coordinates[:, 2])
            largest_face = coordinates[coordinates[:, 2] == max_z]

            # Plane equation for flat surfaces: normal is perpendicular to the z-axis
            normal_vector = np.array([0, 0, 1])
            d = -max_z
            return largest_face, normal_vector, d

        return np.array([]), None, None

    # Get the largest surface and plane equation
    largest_surface_points, normal_vector, d = find_largest_outer_surface(coordinates)

    return largest_surface_points


def rotation_matrix_calculator(normal_vector):
    """
    Rotate the surface so that the normal vector aligns with the Z-axis.
    """
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize the normal vector
    z_axis = np.array([0, 0, 1])  # Target direction (Z-axis)

    # Compute the axis of rotation: cross product of the normal vector and the Z-axis
    rotation_axis = np.cross(normal_vector, z_axis)

    # print("Rotation axis: ", rotation_axis)

    # Compute the angle of rotation: dot product between the normal vector and the Z-axis
    cos_angle = np.dot(normal_vector, z_axis)
    sin_angle = np.linalg.norm(rotation_axis)

    # print("Cos angle: ", cos_angle)
    # print("Sin angle: ", sin_angle) #GOOD

    if sin_angle == 0:  # No rotation needed if the normal is already aligned with Z-axis
        return np.eye(3)

    # Normalize the rotation axis
    rotation_axis = rotation_axis / sin_angle

    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])

    K_squared = np.dot(K, K)

    rotation_matrix = np.eye(3) + K * sin_angle + K_squared * (1 - cos_angle)

    return rotation_matrix


def reorient_coordinates(base_matrix, largest_surface_points):
    """
        Reorients the coordinates of the shape based on the identified largest
        surface plane, ensuring the largest surface is the uppermost parallel to the XY plane.

        :param base_matrix: The current 3D coordinates of the compound.
        :param largest_surface_points: The current 3D coordinates of the largest surface.
        :return new_base_matrix: The reoriented 3D coordinates of the compound with the largest surface at the top.
    """

    if len(largest_surface_points) != 3:
        raise ValueError("Exactly three points are required to define a plane.")

    # Compute the normal vector using the surface points
    P1 = largest_surface_points[0]
    P2 = largest_surface_points[1]
    P3 = largest_surface_points[2]

    # Create vectors from the first point to the other two
    v1 = P2 - P1
    v2 = P3 - P1

    # Compute the normal vector to the surface (cross product of v1 and v2)
    normal_vector = np.cross(v1, v2)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Rotate the shape so the plane becomes parallel to the XY plane (normal vector to Z axis)
    rotation_matrix = rotation_matrix_calculator(normal_vector)

    # Apply the rotation to all coordinates in base_matrix
    coordinates = []
    for line in base_matrix:
        parts = line.split()
        x, y, z = map(float, parts[1:])
        rotated_point = np.dot(rotation_matrix, np.array([x, y, z]))
        coordinates.append(rotated_point)

    coordinates = np.array(coordinates)

    # Check Z-coordinates of the largest surface points after rotation
    rotated_surface_points = [np.dot(rotation_matrix, point) for point in largest_surface_points]
    surface_z_avg = np.mean([point[2] for point in rotated_surface_points])

    # Compare the Z-coordinates of the largest surface to ensure it is at the top
    rotated_surface_points_set = {tuple(point) for point in rotated_surface_points}
    other_points_z_avg = np.mean([coord[2] for coord in coordinates if tuple(coord) not in rotated_surface_points_set])

    if surface_z_avg < other_points_z_avg:
        # Flip the shape by inverting the Z-axis if the surface is below
        inversion_matrix = np.diag([1, 1, -1])
        coordinates = np.dot(coordinates, inversion_matrix)
        rotated_surface_points = np.dot(rotated_surface_points, inversion_matrix)

    # Translate to ensure the surface is at the top
    max_surface_z = np.max([point[2] for point in rotated_surface_points])
    translation = np.array([0, 0, -max_surface_z])

    # Apply the translation to all coordinates
    translated_coordinates = coordinates + translation

    # Process the base_matrix to translate coordinates and create new_base_matrix
    new_base_matrix = []
    for i, line in enumerate(base_matrix):
        parts = line.split()
        new_x, new_y, new_z = translated_coordinates[i]
        new_base_matrix.append(f'{parts[0]} {new_x:.10f} {new_y:.10f} {new_z:.10f}')

    # Extract coordinates from new_base_matrix
    coordinates = np.array([[float(x) for x in line.split()[1:]] for line in new_base_matrix])

    # Find the maximum z-coordinate
    z_max = np.max(coordinates[:, 2])

    # Get surface points, including all three coordinates (x, y, z)
    surface_points = coordinates[coordinates[:, 2] == z_max]

    return new_base_matrix, surface_points, rotation_matrix, translation


def centre_coords(base_matrix):

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
    center = np.mean(coordinates, axis=0)

    # Subtract the center from each coordinate to center the crystal
    centered_coordinates = coordinates - center

    # Format the centered atoms back into strings
    centered_atom_list = []
    for i, coords in enumerate(centered_coordinates):
        centered_atom_list.append(f"{atom_labels[i]} {' '.join(f'{x:.10f}' for x in coords)}")

    return centered_atom_list, center


def unorient_minima(local_minima_positions, rotation_matrix, translation_vector, center):
    """
    Finds the equivalent coordinates of the local minima on the un-oriented shape.

    :param local_minima_positions: np.array, coordinates in the oriented shape.
    :param rotation_matrix: np.array, 3x3 matrix used to rotate the shape during orientation.
    :param translation_vector: np.array, 1x3 vector used to translate the shape during orientation.
    :param center: np.array, center to be subtracted after reorientation.
    :return: np.array, equivalent coordinates in the un-oriented shape.
    """
    # Reverse the translation: Subtract the translation vector
    translated_back = local_minima_positions - translation_vector

    # Reverse the rotation: Apply the inverse rotation matrix
    rotation_matrix_inverse = np.linalg.inv(rotation_matrix)
    original_coordinates = np.dot(translated_back, rotation_matrix_inverse.T)

    # Subtract the center from each set of coordinates
    original_coordinates_centered = original_coordinates - center

    return original_coordinates_centered


def unorient_mol(paired_coordinates, rotation_matrix, translation_vector, center):
    transformed_coordinates = []

    for pair in paired_coordinates:
        transformed_pair = []
        for coord in pair:
            # Apply rotation matrix to the coordinates and then apply translation vector
            transformed_coord = np.dot(rotation_matrix, coord) + translation_vector

            # Subtract the center from each transformed coordinate
            centered_coord = transformed_coord - center

            transformed_pair.append(centered_coord.tolist())

        transformed_coordinates.append(transformed_pair)

    return transformed_coordinates
