import ast
import os
import random
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog
from torch_geometric.utils import to_undirected, degree

#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")

print("Device:", device)



class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        Computes the mean and std for a single feature (e.g., x, y, etc.) across all graphs.
        """
        self.mean = np.mean(np.array(data))
        self.std = np.std(np.array(data))

    def transform(self, data):
        """
        Normalize a single feature based on the mean and std calculated during fitting.
        """
        normalized_data = [(f - self.mean) / self.std for f in data]
        return normalized_data

    def fit_transform(self, data):
        """
        Fit and transform the data in one step.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return [(f * self.std) + self.mean for f in data]


class PreProcess:
    def __init__(self):
        self.normalizers = []  # Initialize normalizers list here
        self.num_feats = None

    def process(self, data, num_feats):
        # This method processes the data and normalizes it
        self.normalizers = []  # Reinitialize normalizers every time we process new data
        normalized_list = []

        for i in range(num_feats):  # 6 features (x, y, z, mass, charge, spin)
            normalizer = Normalizer()  # Create a new normalizer for each feature
            self.normalizers.append(normalizer)  # Add to the normalizers list
            feature_list = [data[j][i] for j in range(len(data))]

            normalized_feature = normalizer.fit_transform(feature_list)  # Fit and transform
            normalized_list.append(normalized_feature)

        # Recombine the normalized features into the correct format
        normalised_features = list(zip(*normalized_list))
        normalised_features = [list(tup) for tup in normalised_features]

        return normalised_features

    def transform(self, data, num_feats):
        # This method calls process and returns the normalized data
        return self.process(data, num_feats)

    def inverse_process(self, normalized_data):
        # Inverse normalization process
        transposed = list(zip(*normalized_data))  # Get feature-wise lists
        denormalized_list = []

        for i, norm_feat in enumerate(transposed):
            # Make sure normalizers list is populated correctly
            if i >= len(self.normalizers):
                raise IndexError(f"Index {i} is out of range for normalizers.")
            original_feat = self.normalizers[i].inverse_transform(list(norm_feat))
            denormalized_list.append(original_feat)

        # Recombine the denormalized features
        return [list(tup) for tup in zip(*denormalized_list)]


class FCNN_FeatureCombiner(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(FCNN_FeatureCombiner, self).__init__()

        self.input = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)

        return x


class GNN(nn.Module):
    def __init__(self, node_dim, hidden_node_dim, output_node_dim,
                 edge_dim, hidden_edge_dim, output_edge_dim,
                 hidden_gnn_dim1, hidden_gnn_dim2, output_gnn_dim=3):
        super(GNN, self).__init__()

        # Node feature combiner
        self.node_feature_combiner = FCNN_FeatureCombiner(input_dim=node_dim,
                                                          hidden_size=hidden_node_dim,
                                                          output_dim=output_node_dim)

        self.edge_feature_combiner = FCNN_FeatureCombiner(input_dim=edge_dim,
                                                          hidden_size=hidden_edge_dim,
                                                          output_dim=1)  # Output size = 1 (weight per edge)

        self.gcn1 = GCNConv(output_node_dim, hidden_gnn_dim1)
        self.gcn2 = GCNConv(hidden_gnn_dim1, hidden_gnn_dim2)

        self.fc_out = nn.Linear(hidden_gnn_dim2, output_gnn_dim)

    def forward(self, node_features, edge_index, edge_features):
        """
        Forward pass through the GNN.

        node_features: Tensor of shape (num_nodes, feature_dim).
        edge_index: Tensor of shape (2, num_edges).
        edge_features: Tensor of shape (num_edges, edge_feature_dim).

        Returns:
        output: Predicted output features for each node.
        """

        node_features_combined = self.node_feature_combiner(node_features)

        edge_weights = self.edge_feature_combiner(edge_features).squeeze(-1)  # Ensure shape is (num_edges,)
        edge_weights = torch.cat([edge_weights, edge_weights], dim=0)
        edge_weights = torch.sigmoid(edge_weights)

        edge_index = edge_index.to(torch.long)
        edge_index = to_undirected(edge_index, num_nodes=node_features_combined.shape[0])

        x = self.gcn1(node_features_combined, edge_index, edge_weights)

        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_weights)
        x = F.relu(x)

        output = self.fc_out(x)

        return output


def load_training_data(csv_path):
    df = pd.read_csv(csv_path)

    def parse_column(col_name):
        num_rows = len(df)
        all_graphs = []

        for i in range(num_rows):
            value = df[col_name].iloc[i]
            try:
                parsed_value = ast.literal_eval(value)  # Parses the string as list
                all_graphs.append(parsed_value)  # One entry per graph
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {col_name} in row {i}: {e}")
                return None

        return all_graphs  # List of [graph_1_data, graph_2_data, ...]

    system_names = df[df.columns[0]].tolist()  # First column

    node_features = parse_column('Node Features Initial Combined')
    edge_features = parse_column('Edge Features Initial Combined')
    edge_indices = parse_column('Edge Indices Combined')
    system_features = parse_column('Diffusion Input Features')
    initial_coords = parse_column('Diffusion Initial Coords')
    output_coords = parse_column('Diffusion Output Coords')

    num_fixed_atoms = df.iloc[:, 10].astype(int).tolist()

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "initial_coords": initial_coords,
        "output_coords": output_coords,
        "num_fixed_atoms": num_fixed_atoms
    }


def load_testing_data(csv_path):
    df = pd.read_csv(csv_path)

    def parse_column(col_name):
        num_rows = len(df)
        all_graphs = []

        for i in range(num_rows):
            value = df[col_name].iloc[i]
            try:
                parsed_value = ast.literal_eval(value)  # Parses the string as list
                all_graphs.append(parsed_value)  # One entry per graph
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {col_name} in row {i}: {e}")
                return None

        return all_graphs  # List of [graph_1_data, graph_2_data, ...]

    system_names = df[df.columns[0]].tolist()  # First column

    node_features = parse_column('Node Features Initial Combined')
    edge_features = parse_column('Edge Features Initial Combined')
    edge_indices = parse_column('Edge Indices Combined')
    system_features = parse_column('Diffusion Input Features')
    initial_coords = parse_column('Diffusion Initial Coords')

    num_fixed_atoms = df.iloc[:, 10].astype(int).tolist()

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "initial_coords": initial_coords,
        "num_fixed_atoms": num_fixed_atoms
    }


def flatten_graph_data(graph_data):
    """Flattens [[...], [...], ...] into a single list and keeps lengths for re-splitting."""
    flat = []
    sizes = []
    for graph in graph_data:
        sizes.append(len(graph))
        flat.extend(graph)
    return flat, sizes


def split_back(flat_data, sizes):
    """Splits flat_data back into original per-graph groups using sizes."""
    result = []
    index = 0
    for size in sizes:
        result.append(flat_data[index:index + size])
        index += size
    return result


def run_training():
    """
        Opens the training file, extracts and normalises data and runs training on the GNN.
        The GNN full model parameters are then saved for use in testing later.
    """

    # root = tk.Tk()
    # root.withdraw()                                                                                                     # Hide the root window.

    # file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])                 # Open file dialog to select training CSV file.
    # if not file_path:
    #     print("No file selected. Exiting...")
    #     return

    file_path = "diffusion_training.csv"
    data = load_training_data(file_path)                                                                                # Load the diffusion data from the selected file.

    element_lists = [[line.split()[0] for line in group] for group in data['output_coords']]

    for i in range(len(data['output_coords'])):
        data['output_coords'][i] = [[float(x) for x in line.split()[1:]] for line in data['output_coords'][i]]

    flat_nodes, node_sizes = flatten_graph_data(data['node_features'])
    flat_edges, edge_sizes = flatten_graph_data(data['edge_features'])
    flat_coords, coord_sizes = flatten_graph_data(data['output_coords'])

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    coord_normaliser = PreProcess()

    node_features_norm_flat = node_normaliser.transform(flat_nodes, 6)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges, len(flat_edges[0]))
    output_coords_norm_flat = coord_normaliser.transform(flat_coords, len(flat_coords[0]))

    node_features_norm = split_back(node_features_norm_flat, node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, edge_sizes)
    output_coords_norm = split_back(output_coords_norm_flat, coord_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],  # Already on device
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            y=torch.tensor(output_coords_norm[i], dtype=torch.float).to(device),
        )

        data_obj.system_name = data['system_names'][i]  # Attach the name
        data_obj.elements = element_lists[i]  # <- Add this
        data_obj.num_fixed = data['num_fixed_atoms'][i]
        graph_list.append(data_obj)

    # random.seed(int(time.time()))  # Seed based on current time
    random.seed(42)
    random.shuffle(graph_list)

    num_train = 6
    train_graphs = graph_list[:num_train]
    test_graphs = graph_list[num_train:]

    print("Train systems:")
    for g in train_graphs:
        print(" -", g.system_name)

    print("\nTest systems:")
    for g in test_graphs:
        print(" -", g.system_name)

    epochs = 2000

    node_size = 6
    node_hidden_size = 16
    node_output_size = 2

    edge_size = 2
    edge_hidden_size = 8
    edge_output_size = 1

    hidden_size1 = 64
    hidden_size2 = 32
    gnn_output_size = 3

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_train_list = []
    loss_test_list = []

    loss_func = nn.SmoothL1Loss(beta=2.0)

    # loss_func = nn.MSELoss()

    batch_size_train = 2
    batch_size_test = 1

    train_loader = DataLoader(train_graphs, batch_size_train, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size_test, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            # print(batch.system_name)

            # Split metadata per individual graph in the batch
            batch_elements = batch.elements  # List[List[str]], length = batch_size
            batch_num_fixed = batch.num_fixed.to(device)

            predicted_coords = model(batch.x, batch.edge_index, batch.edge_attr)

            # Setup to build loss over all graphs in the batch
            loss = 0.0
            start_idx = 0

            for i in range(len(batch_elements)):

                elements = batch_elements[i]
                N_fixed = batch_num_fixed[i].item()
                num_atoms = len(elements)

                end_idx = start_idx + num_atoms
                pred = predicted_coords[start_idx:end_idx]
                true = batch.y[start_idx:end_idx]

                movable_mask = torch.tensor(
                    [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                    dtype=torch.bool,
                    device=pred.device
                )
                fixed_mask = torch.tensor(
                    [idx < N_fixed for idx in range(len(elements))],
                    dtype=torch.bool,
                    device=pred.device
                )

                loss_movable = loss_func(pred[movable_mask], true[movable_mask])
                loss_fixed = loss_func(pred[fixed_mask], true[fixed_mask])
                loss += loss_movable + loss_fixed

                start_idx = end_idx  # move to the next sample in the batch

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        all_predicted_coords = []
        all_elements = []
        all_names = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                batch_elements = batch.elements  # List[List[str]]
                batch_num_fixed = batch.num_fixed.to(device)

                predicted_coords = model(batch.x, batch.edge_index, batch.edge_attr)

                loss = 0.0
                start_idx = 0

                for i in range(len(batch_elements)):
                    elements = batch_elements[i]
                    N_fixed = batch_num_fixed[i].item()
                    num_atoms = len(elements)

                    end_idx = start_idx + num_atoms
                    pred = predicted_coords[start_idx:end_idx]
                    true = batch.y[start_idx:end_idx]

                    movable_mask = torch.tensor(
                        [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                        dtype=torch.bool,
                        device=pred.device
                    )
                    fixed_mask = torch.tensor(
                        [idx < N_fixed for idx in range(len(elements))],
                        dtype=torch.bool,
                        device=pred.device
                    )

                    loss_movable = loss_func(pred[movable_mask], true[movable_mask])
                    loss_fixed = loss_func(pred[fixed_mask], true[fixed_mask])
                    loss += loss_movable + loss_fixed

                    pred = coord_normaliser.inverse_process(pred)

                    all_predicted_coords.append(pred)
                    all_elements.append(elements)
                    all_names.append(batch.system_name[i])  # string per sample

                    start_idx = end_idx

                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_train_loss = train_loss / len(train_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

        loss_train_list.append(avg_train_loss)
        loss_test_list.append(avg_test_loss)

    for i, (name, elements, coords) in enumerate(zip(all_names, all_elements, all_predicted_coords)):
        filename = f"Predicted Coords/{name}_predicted.xyz"
        with open(filename, "w") as f:
            f.write(f"{len(elements)}\n")
            f.write("0 1\n")  # Optional comment line
            for elem, coord in zip(elements, coords):
                f.write(f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.plot(loss_test_list, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'Train Loss')
    plt.show()

    input()

    save_option = input('Do you want to save this model?: ')

    if save_option == 'y':

        hyperparameters = {
            "node_dim": node_size,
            "hidden_node_dim": node_hidden_size,
            "output_node_dim": node_output_size,
            "edge_dim": edge_size,
            "hidden_edge_dim": edge_hidden_size,
            "output_edge_dim": edge_output_size,
            "hidden_gnn_dim1": hidden_size1,
            "hidden_gnn_dim2": hidden_size2,
            "output_gnn_dim": gnn_output_size
        }                                                                                                                   # Store the hyperparameters in a dictionary for saving.

        root = tk.Tk()                                                                                                      # Request user input for model name.
        root.withdraw()                                                                                                     # Hide the root window.

        model_name = simpledialog.askstring("Input", "Enter the model name:")
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        model_dir = "Coordinate GNN Models"                                                                                 # Save the trained model in a folder along with its hyperparameters.
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"coordinate_model_{model_name}.pt")

        model_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters
        }                                                                                                                   # Create a dictionary to store both the state_dict and hyperparameters.

        torch.save(model_data, model_save_path)                                                                             # Save the model data.
        print(f"Model and hyperparameters saved at: {model_save_path}")

    else:
        return None


def run_testing():
    """
        Opens the testing file, extracts and normalises data and runs training on the GNN.
    """

    # root = tk.Tk()
    # root.withdraw()                                                                                                     # Hide the root window.

    # file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])                 # Open file dialog to select training CSV file.
    # if not file_path:
    #     print("No file selected. Exiting...")
    #     return

    file_path = "diffusion_testing.csv"

    data = load_testing_data(file_path)

    element_lists = [[line.split()[0] for line in group] for group in data['output_coords']]

    model_dir = "Coordinate GNN Models"                                                                                 # Select the model to test.
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = torch.load(model_file_path)                                                                            # Load the model and hyperparameters.

    hyperparameters = model_data["hyperparameters"]                                                                     # Retrieve the hyperparameters and use them to initialize the model.

    model = GNN(
        node_dim=hyperparameters["node_dim"],
        hidden_node_dim=hyperparameters["hidden_node_dim"],
        output_node_dim=hyperparameters["output_node_dim"],
        edge_dim=hyperparameters["edge_dim"],
        hidden_edge_dim=hyperparameters["hidden_edge_dim"],
        output_edge_dim=hyperparameters["output_edge_dim"],
        hidden_gnn_dim1=hyperparameters["hidden_gnn_dim1"],
        hidden_gnn_dim2=hyperparameters["hidden_gnn_dim2"],
        output_gnn_dim=hyperparameters["output_gnn_dim"]
    )

    model.load_state_dict(model_data["model_state_dict"])                                                               # Load the state_dict (weights).
    model.eval()                                                                                                        # Set the model to evaluation mode.

    with torch.no_grad():
        node_features = data["node_features"]
        edge_index = data["edge_indices"]
        edge_attr = data["edge_features"]

        predicted_coords = model(node_features, edge_index, edge_attr)

        print("Predicted Optimised Coordinates: ", predicted_coords)

    print("Testing completed.")


run_training()
# run_testing()
