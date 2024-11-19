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
        selected_face = None
        selected_z_avg = -np.inf  # To track the highest average Z value

        # Iterate over all faces (simplices) in the convex hull
        for simplex in hull.simplices:
            face_points = coordinates[simplex]

            # Calculate the area of the face (using the first 3 points of the simplex)
            v1 = face_points[1] - face_points[0]
            v2 = face_points[2] - face_points[0]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))

            # Calculate the average Z-coordinate for the face
            avg_z = np.mean(face_points[:, 2])

            # Check if this face has the largest area or if it's the same area but uppermost
            if area > max_area or (area == max_area and avg_z > selected_z_avg):
                max_area = area
                largest_face = face_points
                selected_z_avg = avg_z
                selected_face = simplex

                # Calculate normal vector and point on plane for the largest face
                normal_vector = np.cross(v1, v2)
                point_on_plane = face_points[0]

        # Find the 3 points on the largest face (select only 3 points)
        if largest_face is not None:
            return largest_face[:3]  # Return the first 3 points on the largest face
        else:
            return np.array([])  # In case no face is found

    largest_surface_points = find_largest_outer_surface(coordinates)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all points
    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color='blue', s=50, label="All Points")

    # Highlight the points on the largest surface plane
    if largest_surface_points.shape[0] == 3:
        ax.scatter(largest_surface_points[:, 0], largest_surface_points[:, 1], largest_surface_points[:, 2],
                   color='red', s=80, label="Largest Surface Points")

        # Create and plot the plane for the largest surface using just the 3 points
        verts = [largest_surface_points]  # Points on the largest surface
        ax.add_collection3d(Poly3DCollection(verts, color='cyan', alpha=0.3, edgecolor='k'))

    # Labels and viewing angle
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=30)

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

    # print("Normalised rotation axis: ", rotation_axis)

    # Compute the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])

    # print("K: ", K) #GOOD

    K_squared = np.dot(K, K)

    # print("K squared: ", K_squared)
    # print("Previous K squared: ", np.outer(rotation_axis, rotation_axis))

    rotation_matrix = np.eye(3) + K * sin_angle + K_squared * (1 - cos_angle)

    # print("Rotation matrix: ", rotation_matrix)

    return rotation_matrix


def reorient_coordinates(base_matrix, largest_surface_points):
    """
    Reorient the coordinates of the shape based on the identified surface plane.
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

    # print("Vector 1: ", v1)
    # print("Vector 2: ", v2)

    # Compute the normal vector to the surface (cross product of v1 and v2)
    normal_vector = np.cross(v1, v2)

    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # print("Normalised normal vector: ", normal_vector)

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

    # Translate to ensure the surface is at the top
    # Find the highest point of the surface in the Z-direction
    max_z = np.max(coordinates[:, 2])
    translation = np.array([0, 0, -max_z])

    # Apply the translation to all coordinates
    translated_coordinates = coordinates + translation

    # Format the new base_matrix with translated coordinates
    new_base_matrix = []
    for i, line in enumerate(base_matrix):
        parts = line.split()
        new_x, new_y, new_z = translated_coordinates[i]
        new_base_matrix.append(f'{parts[0]} {new_x:.10f} {new_y:.10f} {new_z:.10f}')

    # print("New Base Matrix: ", new_base_matrix)

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
