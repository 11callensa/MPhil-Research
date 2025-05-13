import numpy as np
from itertools import combinations

from scipy.spatial import ConvexHull


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


# def place_hydrogen(matrix, surfaces, bond_length, offset1, offset2):
#     """
#         Places hydrogen molecules around the compound surfaces, at a certain offset distance from the surfaces.
#
#         :param matrix: 3D coordinates of the compound.
#         :param surfaces: Coordinates of the compound surfaces.
#         :param bond_length: Bond length of H2.
#         :param offset1: Surface offset distance 1.
#         :param offset2: Surface offset distance 2.
#         :return: 3D coordinates of the compound surrounded by H2 molecules.
#     """
#
#     fig = plt.figure(figsize=(10, 10))                                                                                  # Setup a 3D figure.
#     ax = fig.add_subplot(111, projection='3d')
#
#     all_points = np.vstack(surfaces)                                                                                    # Compute global centroid to determine outward direction.
#     global_centroid = np.mean(all_points, axis=0)
#
#     for i, surface in enumerate(surfaces):
#         surface = np.array(surface)                                                                                     # Extract the surface information.
#         ax.add_collection3d(Poly3DCollection([surface], alpha=0.5, edgecolor='k'))
#
#         A, B, C = surface                                                                                               # Compute the edge lengths.
#         edge1 = np.linalg.norm(A - B)
#         edge2 = np.linalg.norm(B - C)
#         edge3 = np.linalg.norm(C - A)
#         avg_edge_length = (edge1 + edge2 + edge3) / 3
#
#         if avg_edge_length < bond_length:                                                                               # Skip molecule placement if avg edge length is smaller than bond length.
#             continue
#
#         v1, v2 = B - A, C - A                                                                                           # Compute normal vector.
#         normal = np.cross(v1, v2)
#         normal = normal / np.linalg.norm(normal)
#
#         centroid = np.mean(surface, axis=0)                                                                             # Ensure normal points outward.
#         if np.dot(normal, centroid - global_centroid) < 0:
#             normal = -normal                                                                                            # Flip normal if pointing inward.
#
#         offset_distance = offset2 if i % 2 == 0 else offset1
#
#         P1 = centroid - normal * offset_distance                                                                        # Place the molecule's midpoint at the centroid of the surface.
#         P2 = centroid + normal * offset_distance                                                                        # Molecule's center of gravity (midpoint of the bond) should be at centroid.
#
#         P1_offset, P2_offset = P1, P2                                                                                   # Ensure the correct bond length.
#         current_bond_length = np.linalg.norm(P2_offset - P1_offset)                                                     # Make sure the distance between P1 and P2 is exactly the bond length.
#
#         if current_bond_length == 0:                                                                                    # Avoid division by zero.
#             print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
#             continue                                                                                                    # Skip this iteration if the bond length is zero.
#
#         scale_factor = bond_length / current_bond_length
#         P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor
#
#         def is_within_surface(p, surface):                                                                              # Check if both atoms are within the surface's bounds.
#             A, B, C = surface                                                                                           # Check if a point is within the triangle surface using barycentric coordinates.
#             v0, v1, v2 = B - A, C - A, p - A                                                                            # Vectors from point to vertices.
#             d00, d01, d11, d20, d21 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1), np.dot(v0, v2), np.dot(v1, v2)
#             denom = d00 * d11 - d01 * d01
#             v = (d11 * d20 - d01 * d21) / denom
#             w = (d00 * d21 - d01 * d20) / denom
#             u = 1 - v - w
#             return u >= 0 and v >= 0 and w >= 0
#
#         while not is_within_surface(P1_offset, surface) or not is_within_surface(P2_offset, surface):                   # Retry until both atoms are within the surface.
#             P1 = centroid - normal * offset_distance                                                                    # If atoms are out of bounds, adjust position.
#             P2 = centroid + normal * offset_distance
#             P1_offset, P2_offset = P1, P2
#             current_bond_length = np.linalg.norm(P2_offset - P1_offset)
#
#             if current_bond_length == 0:                                                                                # Avoid division by zero.
#                 print(f"Warning: current_bond_length is zero between {P1_offset} and {P2_offset}")
#                 continue                                                                                                # Skip this iteration if the bond length is zero.
#
#             scale_factor = bond_length / current_bond_length
#             P2_offset = P1_offset + (P2_offset - P1_offset) * scale_factor
#
#         tangent = np.cross(normal, v1)                                                                                  # Compute a direction parallel to the surface (tangent direction).
#         if np.linalg.norm(tangent) == 0:
#             tangent = np.cross(normal, v2)                                                                              # If the first cross product is zero, use the other edge.
#         tangent = tangent / np.linalg.norm(tangent) * bond_length                                                       # Scale to bond length.
#
#         P1_offset = centroid - tangent / 2 + normal * offset_distance / 2                                               # Apply offset distance to ensure molecules are outside the surface.
#         P2_offset = centroid + tangent / 2 + normal * offset_distance / 2
#
#         ax.scatter(*P1_offset, color='r', s=50)                                                                         # Plot points and lines.
#         ax.scatter(*P2_offset, color='b', s=50)
#         ax.add_collection3d(Line3DCollection([[P1_offset, P2_offset]], colors='black', linewidths=2))
#
#         matrix.append(f'H {P1_offset[0]:.10f} {P1_offset[1]:.10f} {P1_offset[2]:.10f}')
#         matrix.append(f'H {P2_offset[0]:.10f} {P2_offset[1]:.10f} {P2_offset[2]:.10f}')
#
#     ax.set_xticks(np.arange(int(np.min(all_points[:, 0])) - 2, int(np.max(all_points[:, 0])) + 2, 2))
#     ax.set_yticks(np.arange(int(np.min(all_points[:, 1])) - 2, int(np.max(all_points[:, 1])) + 2, 2))
#     ax.set_zticks(np.arange(int(np.min(all_points[:, 2])) - 2, int(np.max(all_points[:, 2])) + 2, 2))
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()
#
#     return matrix


def layer1_extractor(compound_xyz, surface_points, miller_index, tolerance=0.1):
    """
    Returns atoms on the outermost surface layer that is parallel to a Miller-index-defined plane.

    Parameters:
    - compound_xyz: list of strings ['Ti x y z', ...]
    - surface_points: list of arrays of triangle vertex coordinates from surface_finder
    - miller_index: list like [1, 0, 0]
    - tolerance: float, numerical tolerance when comparing normal vectors and layer values

    Returns:
    - List of strings for atom coordinates on top surface parallel to Miller index
    """

    # Convert Miller index into a normal vector (it's already a normal to a plane)
    miller_normal = np.array(miller_index, dtype=np.float64)
    miller_normal = miller_normal / np.linalg.norm(miller_normal)

    # Determine slicing axis (where miller index has a 1)
    axis = np.argmax(np.abs(miller_index))

    # Step 1: Parse atom coordinates
    atoms = []
    coords = []
    for line in compound_xyz:
        parts = line.split()
        symbol = parts[0]
        coord = np.array(list(map(float, parts[1:])))
        atoms.append((symbol, coord))
        coords.append(coord)
    coords = np.array(coords)

    # Step 2: Find all surface triangles parallel to Miller plane
    aligned_triangles = []
    for tri in surface_points:
        v1, v2, v3 = tri
        normal = np.cross(v2 - v1, v3 - v1)
        if np.linalg.norm(normal) == 0:
            continue
        normal = normal / np.linalg.norm(normal)
        # If normal is perpendicular to Miller normal (i.e. plane is parallel), dot = 0
        if np.abs(np.dot(normal, miller_normal)) < tolerance:
            aligned_triangles.append(tri)

    if not aligned_triangles:
        print("No surface planes aligned with Miller index.")
        return []

    # Step 3: Extract all unique points on these aligned triangles
    surface_coords = set()
    for tri in aligned_triangles:
        for pt in tri:
            surface_coords.add(tuple(pt.round(decimals=5)))  # rounding to avoid floating point noise

    surface_coords = np.array(list(surface_coords))

    # Step 4: Get the max axis value (e.g. max z if Miller is [1, 0, 0])
    max_val = np.max(surface_coords[:, axis])

    # Step 5: Get all atoms from compound_xyz whose coordinate along axis is near max_val
    selected_atoms = []
    for symbol, coord in atoms:
        if abs(coord[axis] - max_val) < tolerance:
            selected_atoms.append(f"{symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}")

    return selected_atoms


def layer2_extractor(compound_xyz, miller_index, first_layer_atoms, distance, tolerance=0.05):
    """
    Extracts atoms lying within a specified distance below a plane formed by the first layer.

    Parameters:
    - compound_xyz: list of strings like ['Ti x y z', ...]
    - miller_index: list of 3 ints, e.g., [1, 0, 0]
    - first_layer_atoms: list of strings from first layer
    - distance: float, depth below the surface to extract next layer
    - tolerance: float, optional numerical fuzziness for edge inclusion

    Returns:
    - List of atoms in string format, like ['Ti x y z', ...]
    """

    # Parse Miller index and normal vector
    miller_normal = np.array(miller_index, dtype=np.float64)
    miller_normal = miller_normal / np.linalg.norm(miller_normal)

    # Step 1: Get a point on the plane (use first atom from first_layer)
    first_layer_coords = []
    for line in first_layer_atoms:
        parts = line.split()
        coord = np.array(list(map(float, parts[1:])))
        first_layer_coords.append(coord)
    first_layer_coords = np.array(first_layer_coords)
    plane_point = np.mean(first_layer_coords, axis=0)  # Plane passes through centroid of first layer

    # Step 2: Parse full compound coordinates
    atoms = []
    for line in compound_xyz:
        parts = line.split()
        symbol = parts[0]
        coord = np.array(list(map(float, parts[1:])))
        atoms.append((symbol, coord))

    # Step 3: Compute signed distance from each atom to the first plane
    selected_atoms = []
    for symbol, coord in atoms:
        vec_to_atom = coord - plane_point
        signed_dist = np.dot(vec_to_atom, miller_normal)
        if -distance - tolerance <= signed_dist <= -tolerance:
            selected_atoms.append(f"{symbol} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}")

    return selected_atoms


def rotater(xyz):

    # Step 1: Parse positions
    elements = []
    coords = []

    for entry in xyz:
        parts = entry.split()
        elements.append(parts[0])
        coords.append([float(x) for x in parts[1:]])

    coords = np.array(coords)

    # Step 2: Compute normal vector of the plane
    p0, p1, p2 = coords[:3]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Step 3: Compute rotation matrix to align normal with Z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)

    print('Normal: ', normal)

    if s == 0:  # Already aligned
        print('Aligned')
        rot_matrix = np.eye(3)
    else:
        vx = np.array([[    0, -v[2],  v[1]],
                       [ v[2],     0, -v[0]],
                       [-v[1],  v[0],     0]])
        rot_matrix = np.eye(3) + vx + (vx @ vx) * ((1 - c) / (s**2))

    rotated_coords = (rot_matrix @ coords.T).T

    rotated_coords = [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in zip(elements, rotated_coords)]

    return rotated_coords


def tiler(atoms_list):
    # Parse atomic entries into element list and position array
    elements = []
    coords = []
    for entry in atoms_list:
        parts = entry.split()
        elements.append(parts[0])
        coords.append([float(x) for x in parts[1:]])

    coords = np.array(coords)

    # Determine unit cell size along X and Y
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    dx = max_coords[0] - min_coords[0]
    dy = max_coords[1] - min_coords[1]

    # Tiling offsets for 2x2 layout: [1] [2], [3] [4]
    shift_vectors = [
        np.array([0, 0, 0]),         # [1]
        np.array([dx, 0, 0]),        # [2]
        np.array([0, dy, 0]),        # [3]
        np.array([dx, dy, 0])        # [4]
    ]

    # Generate tiled atoms
    tiled_atoms = []
    for shift in shift_vectors:
        shifted_coords = coords + shift
        for el, (x, y, z) in zip(elements, shifted_coords):
            tiled_atoms.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")

    return tiled_atoms


def site_finder(layer_atoms, min_bond=1.0, max_bond=3.0, z_tol=1e-3):
    """
    Identifies top, bridge, and hollow adsorption sites on the topmost atomic layer.

    Parameters:
    - layer_atoms: list of strings like ['Ti x y z', ...]
    - min_bond: float, lower cutoff for bond length
    - max_bond: float, upper cutoff for bond length
    - z_tol: float, tolerance for comparing z-coordinates (to handle numerical precision)

    Returns:
    - Dictionary with site categories:
        {'top': [coords], 'bridge': [coords], 'hollow': [coords]}
    """
    coords = []
    for line in layer_atoms:
        parts = line.split()
        coords.append(np.array(list(map(float, parts[1:]))))

    coords = np.array(coords)

    # Step 1: Find max Z
    max_z = np.max(coords[:, 2])

    # Step 2: Filter atoms that lie on this topmost plane (within tolerance)
    top_layer_coords = np.array([c for c in coords if abs(c[2] - max_z) < z_tol])

    top_sites = top_layer_coords.tolist()

    bridge_sites = []
    hollow_sites = []

    # Step 3: Bridge sites (midpoints of bonded pairs)
    for i, j in combinations(range(len(top_layer_coords)), 2):
        dist = np.linalg.norm(top_layer_coords[i] - top_layer_coords[j])
        if min_bond < dist < max_bond:
            midpoint = (top_layer_coords[i] + top_layer_coords[j]) / 2
            bridge_sites.append(midpoint)

    # Step 4: Hollow sites (centroids of bonded triangles)
    for i, j, k in combinations(range(len(top_layer_coords)), 3):
        d1 = np.linalg.norm(top_layer_coords[i] - top_layer_coords[j])
        d2 = np.linalg.norm(top_layer_coords[j] - top_layer_coords[k])
        d3 = np.linalg.norm(top_layer_coords[i] - top_layer_coords[k])
        if all(min_bond < d < max_bond for d in [d1, d2, d3]):
            centroid = (top_layer_coords[i] + top_layer_coords[j] + top_layer_coords[k]) / 3
            hollow_sites.append(centroid)

    return {
        'top': top_sites,
        'bridge': bridge_sites,
        'hollow': hollow_sites
    }


def place_hydrogen(tiled_xyz, adsorption_sites, coverage, hydrogen_bond, height_above=6.0, min_distance=3.0):
    """
    Places H2 molecules evenly across the top surface with spacing constraint.

    Parameters:
    - tiled_xyz: list of strings ['Ti x y z', ...]
    - adsorption_sites: dict of site categories with coordinates
    - coverage: float, number of H2 molecules per top-layer atom (1.0 = 1 ML)
    - hydrogen_bond: float, bond length between two H atoms in Angstrom
    - height_above: float, height above adsorption site to place H2
    - min_distance: minimum allowed distance between any two H atoms

    Returns:
    - new_structure: list of atoms including original + placed H2
    """
    # Get top surface atoms by z
    non_h_atoms = [line for line in tiled_xyz if not line.startswith('H')]
    z_coords = [float(line.split()[3]) for line in non_h_atoms]
    max_z = max(z_coords)
    tolerance = 0.1
    top_surface_atoms = [line for line in non_h_atoms if abs(float(line.split()[3]) - max_z) < tolerance]
    num_surface_atoms = len(top_surface_atoms)
    num_H2 = int(num_surface_atoms * coverage)

    print('Max z (top surface):', max_z)
    print('Number of top surface atoms:', num_surface_atoms)
    print('Requested H₂ molecules:', num_H2)

    # Prioritised adsorption sites
    all_sites = adsorption_sites['top'] + adsorption_sites['bridge'] + adsorption_sites['hollow']
    if len(all_sites) < num_H2:
        print(f"⚠️ Warning: Only {len(all_sites)} adsorption sites available. Reducing H₂ count.")
        num_H2 = len(all_sites)

    step = len(all_sites) / num_H2
    candidate_indices = [int(i * step) for i in range(len(all_sites))]

    placed_H_coords = []
    used_indices = set()
    new_structure = tiled_xyz.copy()

    i = 0  # Index in candidate_indices
    placed = 0
    while placed < num_H2 and i < len(candidate_indices):
        idx = candidate_indices[i]
        if idx in used_indices or idx >= len(all_sites):
            i += 1
            continue

        site = all_sites[idx]
        base_pos = np.array(site) + np.array([0, 0, height_above])
        offset = np.array([hydrogen_bond, 0, 0])
        h1 = base_pos - offset / 2
        h2 = base_pos + offset / 2

        # Check minimum distance from all previously placed H atoms
        too_close = any(
            np.linalg.norm(np.array(existing) - h) < min_distance
            for h in [h1, h2]
            for existing in placed_H_coords
        )

        if not too_close:
            # Accept placement
            new_structure.append(f"H {h1[0]:.6f} {h1[1]:.6f} {h1[2]:.6f}")
            new_structure.append(f"H {h2[0]:.6f} {h2[1]:.6f} {h2[2]:.6f}")
            placed_H_coords.extend([h1, h2])
            used_indices.add(idx)
            placed += 1

        i += 1

    if placed < num_H2:
        print(f"⚠️ Only placed {placed} H₂ molecules due to spacing constraint.")

    return new_structure

