import numpy as np
import matplotlib.pyplot as plt
import csv
import os


CSV_FILE = "edge_indices.csv"  # CSV file storing connections


def load_existing_edges():
    """ Load existing edges from the CSV file. """
    if not os.path.exists(CSV_FILE):
        print("Does not exist")
        return []

    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        return [list(map(int, row)) for row in reader]


def save_edges_to_csv(edges):
    """ Save the updated edge list to the CSV file. """
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(edges)


def build_connections(positions):
    # Parse positions into atom coordinates
    atom_coordinates = np.array([list(map(float, pos.split()[1:])) for pos in positions])

    # Load existing edges from CSV
    edge_indices = load_existing_edges()

    # Function to plot everything
    def draw_plot():
        ax.clear()
        ax.scatter(atom_coordinates[:, 0], atom_coordinates[:, 1], atom_coordinates[:, 2], c='r', marker='o')

        # Label atoms
        for i, (x, y, z) in enumerate(atom_coordinates):
            ax.text(x, y, z, str(i), color='black')

        # Draw bonds
        for bond in edge_indices:
            i, j = bond
            ax.plot([atom_coordinates[i, 0], atom_coordinates[j, 0]],
                    [atom_coordinates[i, 1], atom_coordinates[j, 1]],
                    [atom_coordinates[i, 2], atom_coordinates[j, 2]], c='b')

        plt.draw()

    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Show initial plot
    draw_plot()
    plt.show()

    while True:
        # Ask user to add or delete edges
        print("\nCurrent edges:", edge_indices)
        action = input("Enter 'add' to add edges, 'del' to delete edges, 'done' to finish: ").strip().lower()

        if action == 'add':
            user_input = input("Enter new edges (e.g., [[0, 1], [2, 3]]): ").strip()
            try:
                new_edges = eval(user_input)  # Convert string input to list
                if not all(isinstance(edge, list) and len(edge) == 2 for edge in new_edges):
                    print("Invalid format! Use [[0,1],[2,3]]")
                    continue

                for edge in new_edges:
                    edge = sorted(edge)  # Ensure undirected consistency
                    if edge not in edge_indices:
                        edge_indices.append(edge)
                    else:
                        print(f"Edge {edge} already exists!")

                draw_plot()

            except Exception as e:
                print(f"Invalid input: {e}")

        elif action == 'del':
            user_input = input("Enter edges to delete (e.g., [[0, 1], [2, 3]]): ").strip()
            try:
                edges_to_delete = eval(user_input)
                for edge in edges_to_delete:
                    edge = sorted(edge)  # Normalize
                    if edge in edge_indices:
                        edge_indices.remove(edge)
                    else:
                        print(f"Edge {edge} not found!")

            except Exception as e:
                print(f"Invalid input: {e}")

        elif action == 'done':
            break

        else:
            print("Invalid choice, enter 'add', 'del', or 'done'.")

    # Save updated edges to CSV
    save_edges_to_csv(edge_indices)

    return edge_indices
