import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Correctly formatted base matrix with commas
# base_matrix = [
#     'Li 0.0000000000 0.0000000000 0.0000000000',
#     'Li -0.0000000000 2.0086193919 2.0086193919',
#     'Li 2.0086193919 0.0000000000 2.0086193919',
#     'Li 2.0086193919 2.0086193919 0.0000000000',
#     'H 0.0000000000 0.0000000000 2.0086193919',
#     'H -0.0000000000 2.0086193919 0.0000000000',
#     'H 2.0086193919 0.0000000000 0.0000000000',
#     'H 2.0086193919 2.0086193919 2.0086193919'
# ]


# base_matrix = ['La 0.0000000000 0.0000000000 3.7649854168',
#                'Ni 3.8956769237 -2.2491701206 2.0984875349',
#                'Ni 2.6506802448 -0.0927726175 2.0984875349',
#                'Ni 1.4056835660 -2.2491701206 2.0984875349',
#                'Ni 2.6506802448 1.5303709529 0.2077287805',
#                'Ni 2.6506802448 -1.5303709529 0.0683256443',
#                'H 1.3576333598 2.2769119200 3.7263083042',
#                'H 2.6506802448 0.0372890186 3.7263083042',
#                'H 3.9437271298 2.2769119200 3.7263083042',
#                'H 3.8936743347 0.8127279805 1.5189022224',
#                'H 2.6506802448 2.9656568977 1.5189022224',
#                'H 1.4076861549 0.8127279805 1.5189022224']


# base_matrix = ['Mg 2.2383470000 2.2383470000 1.4962685000',
#                'Mg 0.0000000000 0.0000000000 0.0000000000',
#                'H 3.1138629990 3.1138629990 0.0000000000',
#                'H 0.8755159990 3.6011780010 1.4962685000',
#                'H 1.3628310010 1.3628310010 0.0000000000',
#                'H 3.6011780010 0.8755159990 1.4962685000']


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

    # Compute the convex hull to find outer surfaces (planes)
    hull = ConvexHull(coordinates)

    # Loop over each face (simplices) of the convex hull
    for simplex in hull.simplices:
        # Get the points for the face
        face_points = coordinates[simplex]

        # Calculate the normal vector and plane equation for the face
        v1 = face_points[1] - face_points[0]
        v2 = face_points[2] - face_points[0]
        normal_vector = np.cross(v1, v2)
        point_on_plane = face_points[0]

        # Create the plane for this face
        plane_points = []
        tolerance = 2*10^-15  # Increase this tolerance if needed
        for point in coordinates:
            # Check if the point lies on the plane using the plane equation
            # Calculate the distance from the point to the plane
            distance = np.abs(np.dot(normal_vector, point - point_on_plane))
            if distance < tolerance:
                plane_points.append(point)

        plane_points = np.array(plane_points)

        # Plot the plane as a translucent surface if there are sufficient points
        if len(plane_points) >= 3:
            verts = [plane_points]  # Points on the plane
            ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.3, edgecolor='k'))

    # Labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)

    plt.legend()
    plt.show()


def surface_finder(matrix):
    elements = []
    coordinates = []

    for line in matrix:
        parts = line.split()
        elements.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coordinates = np.array(coordinates)

    def find_largest_outer_surface(coordinates):
        # Compute the convex hull of all points
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

        # Find the equation of the plane for the largest face
        if largest_face is not None:
            d = -np.dot(normal_vector, point_on_plane)
            return largest_face, normal_vector, d  # Return the largest face and its plane parameters
        else:
            return np.array([]), None, None

    # Get the largest surface and plane equation
    largest_surface_points, normal_vector, d = find_largest_outer_surface(coordinates)

    # Find all points lying on the plane
    def points_on_plane(coordinates, normal_vector, d, tolerance=1e-3):
        plane_points = []
        for point in coordinates:
            distance = abs(np.dot(normal_vector, point) + d) / np.linalg.norm(normal_vector)
            if distance < tolerance:
                plane_points.append(point)
        return np.array(plane_points)

    plane_points = points_on_plane(coordinates, normal_vector, d)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color='blue', s=50, label="All Points")

    # Highlight the points on the largest surface plane
    if plane_points.size > 0:
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2],
                   color='red', s=80, label="Points on Largest Surface")

        # Optionally, draw the convex hull for the points on the plane
        if len(plane_points) >= 3:
            hull_plane = ConvexHull(plane_points[:, :2])  # Convex hull in the x-y plane
            verts = [plane_points[hull_plane.vertices, :3]]  # Extract vertices for plotting
            ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.3, edgecolor='k'))

    # Labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)

    plt.title("Largest Surface Plotter")
    plt.legend()
    plt.show()

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
    Reorient the coordinates of the shape based on the identified surface plane,
    ensuring the largest surface is the uppermost parallel to the XY plane.
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

    # Format the new base_matrix with translated coordinates
    new_base_matrix = []
    for i, line in enumerate(base_matrix):
        parts = line.split()
        new_x, new_y, new_z = translated_coordinates[i]
        new_base_matrix.append(f'{parts[0]} {new_x:.10f} {new_y:.10f} {new_z:.10f}')

    return new_base_matrix

# plot(base_matrix)
# surface_points = surface(base_matrix)  # This should return the largest surface points
#
# new_base_matrix = reorient_coordinates(base_matrix, surface_points)
# plot(new_base_matrix)



# surface_points = np.array([[3, 2, 3], [4, 1, 6], [1, 3, 5]])
#
# test_matrix = ['Hi 3 2 3',
#                 'Hi 4 1 6',
#                 'Hi 1 3 5',
#                 'Hi 5 7 9']
#
# plot(test_matrix)
#
# print("Surface points: ", surface_points)
# new_base_matrix = reorient_coordinates(test_matrix, surface_points)
# plot(new_base_matrix)
