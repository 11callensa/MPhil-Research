import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv
import os


def load_existing_edges(name):
    """ Load existing edges from the CSV file. """
    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    if not os.path.exists(CSV_FILE):
        print("Does not exist")
        return []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        return [list(map(int, row)) for row in reader]


def save_edges_to_csv(edges, name):
    """ Save the updated edge list to the CSV file. """
    CSV_FILE = f"Edge Indices/edge_indices_{name}.csv"  # CSV file storing connections
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(edges)


# def build_connections(positions, num_fixed, name):
#     # Parse positions into atom coordinates
#     atom_coordinates = np.array([list(map(float, pos.split()[1:])) for pos in positions])
#
#     # Load existing edges from CSV
#     edge_indices = load_existing_edges(name)
#
#     # Function to plot everything
#     def draw_plot():
#         ax.clear()
#
#         # Plot atoms: Orange for first num_fixed, Red for others
#         colors = ['black' if i < num_fixed else 'red' for i in range(len(atom_coordinates))]
#         ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c=colors, marker='o')
#
#         # Label atoms
#         for i, (x, y, z) in enumerate(atom_coordinates):
#             ax.text(x, y, z, str(i), color='black')
#
#         # Draw bonds
#         for bond in edge_indices:
#             i, j = bond
#             if i < num_fixed and j < num_fixed:
#                 bond_color = 'g'  # Green for connections within num_fixed atoms
#             else:
#                 bond_color = 'b'  # Blue if at least one atom is outside num_fixed
#
#             ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
#                     [atom_coordinates[i, 1], atom_coordinates[j, 1]],
#                     [atom_coordinates[i, 2], atom_coordinates[j, 2]], c=bond_color)
#
#         plt.draw()
#
#     # Create figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Show initial plot
#     draw_plot()
#     plt.show()
#
#     while True:
#         # Ask user to add or delete edges
#         print("\nCurrent edges:", edge_indices)
#         action = input("Enter 'add' to add edges, 'del' to delete edges, 'done' to finish: ").strip().lower()
#
#         if action == 'add':
#             user_input = input("Enter new edges (e.g., [[0, 1], [2, 3]]): ").strip()
#             try:
#                 new_edges = eval(user_input)  # Convert string input to list
#                 if not all(isinstance(edge, list) and len(edge) == 2 for edge in new_edges):
#                     print("Invalid format! Use [[0,1],[2,3]]")
#                     continue
#
#                 for edge in new_edges:
#                     edge = sorted(edge)  # Ensure undirected consistency
#                     if edge not in edge_indices:
#                         edge_indices.append(edge)
#                     else:
#                         print(f"Edge {edge} already exists!")
#
#                 draw_plot()
#
#             except Exception as e:
#                 print(f"Invalid input: {e}")
#
#         elif action == 'del':
#             user_input = input("Enter edges to delete (e.g., [[0, 1], [2, 3]]): ").strip()
#             try:
#                 edges_to_delete = eval(user_input)
#                 for edge in edges_to_delete:
#                     edge = sorted(edge)  # Normalize
#                     if edge in edge_indices:
#                         edge_indices.remove(edge)
#                     else:
#                         print(f"Edge {edge} not found!")
#
#                 draw_plot()
#
#             except Exception as e:
#                 print(f"Invalid input: {e}")
#
#         elif action == 'done':
#             break
#
#         else:
#             print("Invalid choice, enter 'add', 'del', or 'done'.")
#
#     # Save updated edges to CSV
#     save_edges_to_csv(edge_indices, name)
#
#     return edge_indices


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def build_connections(positions, num_fixed, name):
    """
    Build connections for a crystal structure.

    - Keeps the original compound edges unchanged.
    - Connects each hydrogen atom to 4 nearest hydrogen atoms.
    - Connects each hydrogen atom to at least 4 nearest compound atoms.

    :param positions: List of atomic positions as strings.
    :param num_fixed: Number of atoms in the compound (fixed structure).
    :param name: Name used to load/save edge indices.
    :return: Updated edge indices.
    """

    # Parse positions into numpy array
    atom_coordinates = np.array([list(map(float, pos.split()[1:])) for pos in positions])

    # Load existing edges (these must remain unchanged)
    edge_indices = load_existing_edges(name)

    # Separate compound atoms and hydrogen atoms
    compound_coords = atom_coordinates[:num_fixed]  # First 63 atoms
    hydrogen_coords = atom_coordinates[num_fixed:]  # Remaining atoms

    # Create KDTree for nearest neighbor search
    tree_compound = KDTree(compound_coords)
    tree_hydrogen = KDTree(hydrogen_coords)

    # Find 4 nearest hydrogen neighbors for each hydrogen atom
    for i in range(len(hydrogen_coords)):
        global_index = num_fixed + i  # Convert hydrogen index to global index
        _, neighbor_indices = tree_hydrogen.query(hydrogen_coords[i], k=7)  # Get 5 (includes self)
        for neighbor in neighbor_indices[1:]:  # Skip self (first index)
            edge = tuple(sorted([global_index, num_fixed + neighbor]))  # Undirected
            if edge not in edge_indices:
                edge_indices.append(edge)

    # Find at least 4 nearest compound atoms for each hydrogen atom
    for i in range(len(hydrogen_coords)):
        global_index = num_fixed + i
        _, neighbor_indices = tree_compound.query(hydrogen_coords[i], k=2)  # Get 4 closest compound atoms
        for neighbor in neighbor_indices:
            edge = tuple(sorted([global_index, neighbor]))  # Undirected
            if edge not in edge_indices:
                edge_indices.append(edge)

    # Function to visualize connections
    def draw_plot():
        ax.clear()

        # Color compound atoms black, hydrogen atoms red
        colors = ['black' if i < num_fixed else 'red' for i in range(len(atom_coordinates))]
        ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c=colors, marker='o')

        # Label atoms
        for i, (x, y, z) in enumerate(atom_coordinates):
            ax.text(x, y, z, str(i), color='black')

        # Draw bonds
        for bond in edge_indices:
            i, j = bond
            bond_color = 'g' if i < num_fixed and j < num_fixed else 'b'  # Green = compound bonds, Blue = hydrogen bonds
            ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                    [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                    [atom_coordinates[i, 2], atom_coordinates[j, 2]], c=bond_color)

        plt.draw()

    # Plot the updated structure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_plot()
    plt.show()

    # Save updated edges
    save_edges_to_csv(edge_indices, name)

    return edge_indices

name = 'LiH-Trial'
num_fixed = 8
coordinates = ['Li -1.0043096959 1.0043096959 1.0043096959', 'Li -1.0043096959 -1.0043096959 -1.0043096959', 'Li 1.0043096959 -1.0043096959 1.0043096959', 'Li 1.0043096959 1.0043096959 -1.0043096959', 'H -1.0043096959 -1.0043096959 1.0043096959', 'H -1.0043096959 1.0043096959 -1.0043096959', 'H 1.0043096959 1.0043096959 1.0043096959', 'H 1.0043096959 -1.0043096959 -1.0043096959', 'H -0.5850856992 -1.7543096959 -0.5850856992', 'H -0.0844540981 -1.7543096959 -0.0844540981', 'H 0.0844540981 -2.3793096959 0.0844540981', 'H 0.5850856992 -2.3793096959 0.5850856992', 'H 0.0844540981 -0.0844540981 1.7543096959', 'H 0.5850856992 -0.5850856992 1.7543096959', 'H -0.5850856992 0.5850856992 2.3793096959', 'H -0.0844540981 0.0844540981 2.3793096959', 'H 1.7543096959 -0.5850856992 0.5850856992', 'H 1.7543096959 -0.0844540981 0.0844540981', 'H 2.3793096959 0.0844540981 -0.0844540981', 'H 2.3793096959 0.5850856992 -0.5850856992', 'H -1.7543096959 -0.5850856992 -0.5850856992', 'H -1.7543096959 -0.0844540981 -0.0844540981', 'H -2.3793096959 0.0844540981 0.0844540981', 'H -2.3793096959 0.5850856992 0.5850856992', 'H -0.5850856992 1.7543096959 0.5850856992', 'H -0.0844540981 1.7543096959 0.0844540981', 'H 0.0844540981 2.3793096959 -0.0844540981', 'H 0.5850856992 2.3793096959 -0.5850856992', 'H -0.0844540981 -0.0844540981 -1.7543096959', 'H -0.5850856992 -0.5850856992 -1.7543096959', 'H 0.5850856992 0.5850856992 -2.3793096959', 'H 0.0844540981 0.0844540981 -2.3793096959']

# name = 'TiO2-A'
# num_fixed = 63

# coordinates = ['Ti 0.0000000000 0.0000000000 0.0000000002', 'Ti 0.0000000000 -1.8912698030 2.4037554262', 'Ti 0.0000000000 1.8912698030 2.4037554262', 'Ti -1.8912698030 -1.8912698030 -4.8075108528', 'Ti -1.8912698030 -1.8912698030 4.8075108532', 'Ti -1.8912698030 1.8912698030 -4.8075108528', 'Ti -1.8912698030 1.8912698030 4.8075108532', 'Ti 1.8912698030 -1.8912698030 -4.8075108528', 'Ti 1.8912698030 -1.8912698030 4.8075108532', 'Ti 1.8912698030 1.8912698030 -4.8075108528', 'Ti 1.8912698030 1.8912698030 4.8075108532', 'Ti -1.8912698030 0.0000000000 -2.4037554268', 'Ti 1.8912698030 0.0000000000 -2.4037554268', 'O -1.8912698030 0.0000000000 -0.4119832958', 'O 1.8912698030 0.0000000000 -0.4119832958', 'O 0.0000000000 0.0000000000 1.9917721302', 'O 0.0000000000 -1.8912698030 0.4119832962', 'O 0.0000000000 1.8912698030 0.4119832962', 'O -1.8912698030 -1.8912698030 2.8157387222', 'O -1.8912698030 1.8912698030 2.8157387222', 'O 1.8912698030 -1.8912698030 2.8157387222', 'O 1.8912698030 1.8912698030 2.8157387222', 'O 0.0000000000 -1.8912698030 4.3955275562', 'O 0.0000000000 1.8912698030 4.3955275562', 'O -1.8912698030 -1.8912698030 -2.8157387228', 'O -1.8912698030 1.8912698030 -2.8157387228', 'O 1.8912698030 -1.8912698030 -2.8157387228', 'O 1.8912698030 1.8912698030 -2.8157387228', 'O -1.8912698030 0.0000000000 -4.3955275568', 'O 1.8912698030 0.0000000000 -4.3955275568', 'O 0.0000000000 0.0000000000 -1.9917721298', 'O 0.0000000000 -3.7825396060 1.9917721302', 'O 0.0000000000 3.7825396060 1.9917721302', 'O -3.7825396060 -1.8912698030 -5.2194941488', 'O -1.8912698030 -3.7825396060 -4.3955275568', 'O -1.8912698030 -1.8912698030 -6.7992829828', 'O 0.0000000000 -1.8912698030 -5.2194941488', 'O -3.7825396060 -1.8912698030 4.3955275562', 'O -1.8912698030 -3.7825396060 5.2194941492', 'O -1.8912698030 -1.8912698030 6.7992829832', 'O -1.8912698030 0.0000000000 5.2194941492', 'O -3.7825396060 1.8912698030 -5.2194941488', 'O -1.8912698030 1.8912698030 -6.7992829828', 'O -1.8912698030 3.7825396060 -4.3955275568', 'O 0.0000000000 1.8912698030 -5.2194941488', 'O -3.7825396060 1.8912698030 4.3955275562', 'O -1.8912698030 1.8912698030 6.7992829832', 'O -1.8912698030 3.7825396060 5.2194941492', 'O 1.8912698030 -3.7825396060 -4.3955275568', 'O 1.8912698030 -1.8912698030 -6.7992829828', 'O 3.7825396060 -1.8912698030 -5.2194941488', 'O 1.8912698030 -3.7825396060 5.2194941492', 'O 1.8912698030 -1.8912698030 6.7992829832', 'O 1.8912698030 0.0000000000 5.2194941492', 'O 3.7825396060 -1.8912698030 4.3955275562', 'O 1.8912698030 1.8912698030 -6.7992829828', 'O 1.8912698030 3.7825396060 -4.3955275568', 'O 3.7825396060 1.8912698030 -5.2194941488', 'O 1.8912698030 1.8912698030 6.7992829832', 'O 1.8912698030 3.7825396060 5.2194941492', 'O 3.7825396060 1.8912698030 4.3955275562', 'O -3.7825396060 0.0000000000 -1.9917721298', 'O 3.7825396060 0.0000000000 -1.9917721298', 'H 3.2987979595 2.7074108767 5.6937699773', 'H 2.7919819949 3.0243402422 6.0731871030', 'H -2.9942296175 3.7352120481 -6.0371396807', 'H -3.3111589830 3.2283960834 -6.4165568063', 'H -3.1289880900 -3.0700756895 5.6066579079', 'H -2.9617918644 -2.6616754295 6.1602991724', 'H -3.4672171994 -3.3541306416 -6.1263950571', 'H -2.8381714011 -3.6094774899 -6.3273014299', 'H -2.7919819949 3.0243402422 6.0731871030', 'H -3.2987979595 2.7074108767 5.6937699773', 'H 2.9942296175 3.7352120481 -6.0371396807', 'H 3.3111589830 3.2283960834 -6.4165568063', 'H 3.1803984587 -2.9177165529 -5.7830253538', 'H 2.5513526604 -3.1730634013 -5.9839317265', 'H 3.3541306418 -3.4672171993 6.1263950571', 'H 3.6094774901 -2.8381714009 6.3273014297', 'H -2.7748619468 0.8236588835 -7.0384422251', 'H -3.2301403199 0.4371876519 -6.6581455118', 'H -3.8059586000 -0.4371876519 -6.9915185835', 'H -4.2612369731 -0.8236588835 -6.6112218702', 'H -2.8095400884 3.9249294924 -1.5530389461', 'H -3.2945062248 3.4399633560 -1.3773127582', 'H -3.8819050943 3.7364479630 1.3773127583', 'H -4.3668712307 3.2514818266 1.5530389461', 'H -0.3009980212 4.5325396060 -1.0609247698', 'H -0.9598485141 4.5325396060 -1.3201158731', 'H 0.9598485141 5.1575396060 2.1440824655', 'H 0.3009980212 5.1575396060 1.8848913622', 'H 2.9783275735 3.7561420073 -1.8034862452', 'H 3.1257187396 3.6087508412 -1.1268654590', 'H 3.8819050943 3.7364479630 1.3773127583', 'H 4.3668712307 3.2514818266 1.5530389461', 'H -4.5325396060 -0.3009980212 -1.8848913622', 'H -4.5325396060 -0.9598485141 -2.1440824654', 'H -5.1575396060 0.9598485141 1.3201158728', 'H -5.1575396060 0.3009980212 1.0609247696', 'H -3.4429691664 -3.2915004144 -1.6367911677', 'H -3.9219236820 -2.8125458987 -1.8428715985', 'H -3.2544876370 -4.3638654203 1.8428715983', 'H -3.7334421526 -3.8849109047 1.6367911675', 'H -0.9598485141 -4.5325396060 2.1440824655', 'H -0.3009980212 -4.5325396060 1.8848913622', 'H 0.3009980212 -5.1575396060 -1.0609247698', 'H 0.9598485141 -5.1575396060 -1.3201158731', 'H -3.9117561788 0.8530068119 5.4442058766', 'H -3.5713331432 0.4078397234 5.8768748264', 'H -3.7725231801 -0.4078397234 6.6319251742', 'H -3.4321001444 -0.8530068119 7.0645941240', 'H -0.2764232677 3.6329244010 6.3216975909', 'H -0.9844232677 3.6329244010 6.3216975909', 'H 0.6304232677 3.1314879376 7.5549079109', 'H 0.6304232677 3.6748611001 7.1010250997', 'H 0.2764232677 3.7415446609 -5.6605403519', 'H 0.9844232677 3.7415446609 -5.6605403519', 'H -0.6304232677 3.3834164765 -7.1264698175', 'H -0.6304232677 3.8212068477 -6.5700494810', 'H 4.5325396060 0.3009980212 1.0609247696', 'H 4.5325396060 0.9598485141 1.3201158728', 'H 5.1575396060 -0.9598485141 -2.1440824654', 'H 5.1575396060 -0.3009980212 -1.8848913622', 'H 3.1257187397 -3.6087508411 2.0781417759', 'H 2.9783275734 -3.7561420073 1.4015209898', 'H 3.8849109047 -3.7334421526 -1.6367911677', 'H 4.3638654203 -3.2544876370 -1.8428715985', 'H 3.4052852145 -0.4371876519 -6.5118459471', 'H 3.8605635875 -0.8236588835 -6.1315492338', 'H 3.1314879376 0.6304232677 -7.5549079105', 'H 3.6748611001 0.6304232677 -7.1010250993', 'H -0.6304232677 -0.9844232677 -7.5492829828', 'H -0.6304232677 -0.2764232677 -7.5492829828', 'H 0.2764232677 0.6304232677 -8.1742829828', 'H 0.9844232677 0.6304232677 -8.1742829828', 'H 0.8530068119 -3.9117561788 -5.4442058770', 'H 0.4078397234 -3.5713331430 -5.8768748268', 'H -0.4078397234 -3.7725231799 -6.6319251744', 'H -0.8530068119 -3.4321001442 -7.0645941241', 'H 3.7415446610 -0.9844232677 5.6605403515', 'H 3.7415446610 -0.2764232677 5.6605403515', 'H 3.8212068478 0.6304232677 6.5700494808', 'H 3.3834164767 0.6304232677 7.1264698174', 'H -0.3801074671 -0.3801074671 7.5492829832', 'H -0.8807390682 -0.8807390682 7.5492829832', 'H 0.8807390682 0.8807390682 8.1742829832', 'H 0.3801074671 0.3801074671 8.1742829832', 'H 0.2764232677 -3.6329244010 6.3216975909', 'H 0.9844232677 -3.6329244010 6.3216975909', 'H -0.8236588835 -3.1755353324 7.5181148619', 'H -0.4371876519 -3.6308137054 7.1378181486']

build_connections(coordinates, num_fixed, name)