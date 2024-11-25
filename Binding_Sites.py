from scipy.spatial import Delaunay, ConvexHull
from matplotlib.path import Path
from pyscf import gto, scf, dft
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.signal import argrelextrema

from Mol_Geometry import plot
from DFT import get_spin


def mesh(compound_xyz):
    coordinates = np.array([[float(x) for x in line.split()[1:]] for line in compound_xyz])

    z_max = np.max(coordinates[:, 2])
    surface_points = coordinates[coordinates[:, 2] == z_max][:, :2]  # Only x, y

    hull = ConvexHull(surface_points)
    hull_path = Path(surface_points[hull.vertices])

    x_min, x_max = np.min(surface_points[:, 0]), np.max(surface_points[:, 0])
    y_min, y_max = np.min(surface_points[:, 1]), np.max(surface_points[:, 1])

    resolution = 8
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]

    inside_mask = hull_path.contains_points(grid_points)
    filtered_points = grid_points[inside_mask]

    filtered_points_3d = np.c_[filtered_points, np.full(filtered_points.shape[0], z_max)]
    all_points = np.vstack((filtered_points_3d, coordinates[coordinates[:, 2] == z_max]))

    tri = Delaunay(all_points[:, :2])

    triangles = tri.simplices
    triangle_faces = [[list(all_points[vertex]) for vertex in triangle] for triangle in triangles]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.add_collection3d(Poly3DCollection(triangle_faces, alpha=0.5, edgecolor='k'))
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], color='r', s=5, label="Grid Points")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Refined 3D Meshed Surface")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_max - 0.1, z_max + 0.1)

    plt.legend()
    plt.show()

    return all_points, surface_points


# Compute energy with added hydrogen atom
def compute_energy_with_hydrogen(compound_xyz, mesh_points):
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


# Find the local minima
def find_local_minima(positions, energies):
    local_minima_indices = argrelextrema(energies, np.less)[0]
    local_minima_positions = positions[local_minima_indices]

    return local_minima_positions


# Plot the local minima on top of the heatmap
def plot_local_minima(positions, energies, local_minima_positions, mesh_points, surface_points_2d):
    """
    Plots the heatmap of energies in 2D and overlays the local minima and surface points.

    Args:
        positions (ndarray): The 2D positions of the mesh points.
        energies (ndarray): The energies at each mesh point.
        local_minima_positions (ndarray): The 2D positions of the local minima.
        surface_points_2d (ndarray): The 2D coordinates of the surface mesh points.
    """
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

