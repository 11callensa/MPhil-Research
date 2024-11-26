from scipy.spatial import Delaunay, ConvexHull
from matplotlib.path import Path
from pyscf import gto, dft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from DFT import get_spin


def mesh(compound_xyz, surface_points):
    """
        Creates a mesh of points on the surface for hydrogen atoms to be placed at each.

        :param compound_xyz: The 3D coordinates of the reoriented compound.
        :param surface_points: The 3D coordinates of the uppermost surface.
        :return: The coordinates of the mesh points.
    """

    coordinates = np.array([[float(x) for x in line.split()[1:]] for line in compound_xyz])

    z_max = np.max(coordinates[:, 2])

    # Keep only the x and y coordinates
    surface_points = surface_points[:, :2]

    hull = ConvexHull(surface_points)
    hull_path = Path(surface_points[hull.vertices])

    x_min, x_max = np.min(surface_points[:, 0]), np.max(surface_points[:, 0])
    y_min, y_max = np.min(surface_points[:, 1]), np.max(surface_points[:, 1])

    resolution = 7
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]

    inside_mask = hull_path.contains_points(grid_points)
    filtered_points = grid_points[inside_mask]

    filtered_points_3d = np.c_[filtered_points, np.full(filtered_points.shape[0], z_max)]
    mesh_points = np.vstack((filtered_points_3d, coordinates[coordinates[:, 2] == z_max]))

    tri = Delaunay(mesh_points[:, :2])

    triangles = tri.simplices
    triangle_faces = [[list(mesh_points[vertex]) for vertex in triangle] for triangle in triangles]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(Poly3DCollection(triangle_faces, alpha=0.5, edgecolor='k'))
    ax.scatter(mesh_points[:, 0], mesh_points[:, 1], mesh_points[:, 2], color='r', s=5, label="Grid Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Refined 3D Meshed Surface")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_max - 0.1, z_max + 0.1)

    plt.legend()
    plt.show()

    return mesh_points


def energy_profile(compound_xyz, mesh_points):
    """
        Places a hydrogen atom at each mesh point and computes the total energy, creating an
        energy profile.
        
        :param compound_xyz: 3D coordinates of the compound.
        :param mesh_points: Locations to conduct energy calculation.
        :return: Each mesh point and its associated energy.
    """
    
    energies = []
    positions = []

    for point in tqdm(mesh_points, desc="Computing energies", unit="point"):
        hydrogen_position = point + [0, 0, 1.0]
        all_atoms = compound_xyz + [f'H {hydrogen_position[0]} {hydrogen_position[1]} {hydrogen_position[2]}']

        symbols = [line.split()[0] for line in all_atoms]
        element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

        mol = gto.M(
            verbose=0,
            atom=all_atoms,
            basis='def2-svp',
            unit='Angstrom',
            spin=get_spin(element_count)
        )

        mf = dft.RKS(mol)
        mf.xc = 'b3lyp'
        energy = mf.kernel() * 27.2114

        energies.append(energy)
        positions.append(point)

    return np.array(positions), np.array(energies)


def find_local_minima(positions, energies, distance_threshold=1.0):
    """
        Compares each mesh point's energy with the energies of the surrounding mesh points
        to identify local minima energy locations.

        :param positions: Mesh points.
        :param energies: Associated energies of all mesh points.
        :param distance_threshold: Nearest points means any points within a 1 angstrom radius.
        :return:
    """

    local_minima_indices = []

    # Convert positions to a 2D array of [x, y] (ignoring z)
    xy_positions = positions[:, :2]

    # Loop through all positions
    for i, (pos, energy) in enumerate(zip(xy_positions, energies)):
        # Calculate distances between the current point and all other points
        distances = cdist([pos], xy_positions)[0]

        # Find indices of points within the threshold distance
        nearby_indices = np.where(distances <= distance_threshold)[0]

        # Compare energy with its neighbors
        # If the energy at this position is less than all nearby positions, it's a local minimum
        is_local_minimum = True
        for j in nearby_indices:
            if energies[j] < energy:
                is_local_minimum = False
                break

        # If it's a local minimum, add the index
        if is_local_minimum:
            # Check if this energy is already in the list of minima
            if energy not in [energies[idx] for idx in local_minima_indices]:
                local_minima_indices.append(i)
            else:
                # If the energy is the same as a previously found local minimum, include this as well
                # To prevent duplication, ensure the same index isn't added again
                if i not in local_minima_indices:
                    local_minima_indices.append(i)

    # Extract positions of local minima based on the identified indices
    local_minima_positions = positions[local_minima_indices]

    return local_minima_positions


def plot_local_minima(positions, energies, local_minima_positions, mesh_points):
    # Prepare heatmap of energies
    x = positions[:, 0]
    y = positions[:, 1]
    z = energies

    grid_x, grid_y = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    # Create a 2D plot
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_x, grid_y, grid_z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(label="Energy")

    plt.scatter(mesh_points[:, 0], mesh_points[:, 1],
                color='green', s=20, label="Mesh Points")

    # Overlay local minima
    plt.scatter(local_minima_positions[:, 0], local_minima_positions[:, 1],
                color='red', s=50, label="Local Minima", marker='x')

    # Formatting
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Local Minima and Energy Heatmap')
    plt.legend()
    plt.show()
