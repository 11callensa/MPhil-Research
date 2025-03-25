import pandas as pd
import torch
import ast

import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool  # Import pooling function
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog
import os


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
                 hidden_gnn_dim1, hidden_gnn_dim2, output_gnn_dim):
        super(GNN, self).__init__()

        # Node feature combiner
        self.node_feature_combiner = FCNN_FeatureCombiner(input_dim=node_dim,
                                                          hidden_size=hidden_node_dim,
                                                          output_dim=output_node_dim)

        # Edge feature combiner (to scalar weight)
        self.edge_feature_combiner = FCNN_FeatureCombiner(input_dim=edge_dim,
                                                          hidden_size=hidden_edge_dim,
                                                          output_dim=output_edge_dim)

        # GCN layers
        self.gcn1 = GCNConv(output_node_dim, hidden_gnn_dim1)
        self.gcn2 = GCNConv(hidden_gnn_dim1, hidden_gnn_dim2)

        # Final output layer
        self.fc_out = nn.Linear(hidden_gnn_dim2, output_gnn_dim)

    def forward(self, node_features, edge_index, edge_features, batch):
        """
        Forward pass through the GNN.

        node_features: (num_nodes, feature_dim)
        edge_index: (2, num_edges)
        edge_features: (num_edges, edge_feature_dim)
        batch: (num_nodes,) tensor mapping each node to its graph

        Returns:
        output: (batch_size, 1) tensor (one value per graph)
        """

        # Process node and edge features
        node_features_combined = self.node_feature_combiner(node_features)
        edge_weights = self.edge_feature_combiner(edge_features).squeeze(-1)  # Ensure shape is (num_edges,)

        # Convert edge indices to int64
        edge_index = edge_index.to(torch.long)

        # Pass through GCN layers
        x = self.gcn1(node_features_combined, edge_index, edge_weights)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_weights)
        x = F.relu(x)

        # **Graph-level pooling**
        x = global_mean_pool(x, batch)  # Now x shape is (batch_size, hidden_gnn_dim2)

        # print("x shape final: ", x.shape)

        # Final output layer
        output = self.fc_out(x)  # Now shape is (batch_size, 1)
        return output


class DataNormalizer(object):
    def __init__(self, epsilon=1e-8):                                                                                   # Epsilon is a small value to prevent division by small values/zero.
        """
            Gives the object mean, standard deviation and small epsilon attributes.

            :param epsilon: Small stability factor.
        """
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def fit(self, data):
        """
            Computes the mean and standard deviation for each feature-set in the data.

            :param data: Load data.
        """
        self.mean = data.mean(dim=(0))
        self.std = data.std(dim=(0)) + self.epsilon  # Avoid division by zero

    def transform(self, data):
        """
            Standardizes the data using the computed mean and standard deviation.

            :param data: Load data.
            :return Standardized data.
        """
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        """
            Extracts the mean and standard deviation using fit(), and then standardizes the data using transform().

            :param data: Load data.
            :return Standardized data.
        """
        self.fit(data)                                                                                                  # Fit the data first
        return self.transform(data)

    def inverse_transform(self, data):
        """
            Reverses the standardization, scaling back to the original data.

            :param data; Standardized load data.
            :return Original data.
        """
        return data * self.std + self.mean


def load_training_data(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Function to safely parse string representations of nested lists
    def parse_column(col_name, dtype=torch.float32):
        num_rows = len(df)
        all_values = []

        for i in range(num_rows):
            value = df[col_name].iloc[i]  # Extract row data

            try:
                parsed_value = ast.literal_eval(value)  # Convert string to Python list

                # if col_name == 'Edge Indices (Triple)':
                #     parsed_value = list(zip(*parsed_value))

                all_values.extend(parsed_value)  # Append all sets from the row

            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {col_name} in row {i}: {e}")
                return None  # Return None if there's an issue

        return all_values

    # Convert each column
    node_features = parse_column('Node Features (Triple)')
    edge_features = parse_column('Edge Features (Triple)')
    edge_indices = parse_column('Edge Indices (Triple)')
    system_features = parse_column('Energy Input Features (Triple)')
    energy_output = parse_column('Energy Output Features (Triple)')

    return {"node_features": node_features, "edge_features": edge_features,
            "edge_indices": edge_indices, "system_features": system_features,
            "energy_output": energy_output}


def preprocess_node_features(node_features):
    """
    Extracts the first 7 features from each atom's feature list while preserving the list structure.

    Args:
        node_features (list of list of lists):
        [
            [ [atom1_feats], [atom2_feats], ... ],  # System 1
            [ [atom1_feats], [atom2_feats], ... ],  # System 2
            ...
        ]

    Returns:
        list of list of lists: Same structure, but each atom now only has its first 7 features.
    """
    return [[atom[:7] for atom in system] for system in node_features]


def compute_normalization_stats(node_features_list, edge_features_list, energy_output_list):
    """
    Computes mean and standard deviation for:
        - Node features across all nodes in all graphs.
        - Edge features across all edges in all graphs.
        - Energy outputs across all graphs.

    Returns:
        node_mean, node_std, edge_mean, edge_std, energy_mean, energy_std
    """
    all_node_features = []
    all_edge_features = []
    all_energy_outputs = []

    for system in node_features_list:
        all_node_features.extend(system)  # Flatten node features

    for system in edge_features_list:
        all_edge_features.extend(system)  # Flatten edge features

    all_energy_outputs.extend(energy_output_list)  # Energy is per system

    # print(len(all_energy_outputs))
    # input()

    # Convert to tensors
    all_node_features = torch.tensor(all_node_features, dtype=torch.float32)
    all_edge_features = torch.tensor(all_edge_features, dtype=torch.float32)
    all_energy_outputs = torch.tensor(all_energy_outputs, dtype=torch.float32)

    # Compute means and stds
    node_mean, node_std = all_node_features.mean(dim=0), all_node_features.std(dim=0)
    # print(node_mean)
    # print(node_std)
    # input()
    edge_mean, edge_std = all_edge_features.mean(dim=0), all_edge_features.std(dim=0)
    energy_mean, energy_std = all_energy_outputs.mean(dim=0), all_energy_outputs.std(dim=0)

    # Prevent division by zero
    node_std[node_std == 0] = 1.0
    edge_std[edge_std == 0] = 1.0
    energy_std[energy_std == 0] = 1.0

    return node_mean, node_std, edge_mean, edge_std, energy_mean, energy_std


def create_graph_data(node_features, edge_indices, edge_features, energy_output,
                      node_mean, node_std, edge_mean, edge_std, energy_mean, energy_std):
    """
    Converts a single system's data into a PyTorch Geometric Data object, normalizing:
        - Node features
        - Edge features
        - Energy outputs
    """
    node_features = torch.tensor(node_features, dtype=torch.float32)
    node_features = (node_features - node_mean) / node_std  # Normalize nodes

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).T
    edge_features = torch.tensor(edge_features, dtype=torch.float32)
    edge_features = (edge_features - edge_mean) / edge_std  # Normalize edges

    energy_output = torch.tensor(energy_output, dtype=torch.float32)
    energy_output = (energy_output - energy_mean) / energy_std  # Normalize energy

    # print(energy_output)
    # input()

    return Data(
        x=node_features,
        edge_index=edge_indices,
        edge_attr=edge_features,
        y=energy_output.unsqueeze(0),  # Ensure shape is correct for PyG
    )


def create_dataset(node_features_list, edge_indices_list, edge_features_list, energy_output_list):
    """
    Converts all system data into a list of PyG Data objects with normalization.
    """
    node_mean, node_std, edge_mean, edge_std, energy_mean, energy_std = compute_normalization_stats(
        node_features_list, edge_features_list, energy_output_list
    )

    dataset = []
    for i in range(len(node_features_list)):
        data = create_graph_data(
            node_features_list[i], edge_indices_list[i], edge_features_list[i], energy_output_list[i],
            node_mean, node_std, edge_mean, edge_std, energy_mean, energy_std
        )
        dataset.append(data)

    return dataset


def data_preprocess(batch_size):
    # # Initialize Tkinter root for file selection
    # root = tk.Tk()
    # root.withdraw()  # Hide the root window
    #
    # # Open file dialog to select training CSV file
    # file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV Files", "*.csv")])
    # if not file_path:
    #     print("No file selected. Exiting...")
    #     return

    file_path = 'energy_training.csv'

    # Load the diffusion data from the selected file
    data = load_training_data(file_path)

    data["node_features"] = preprocess_node_features(data["node_features"])

    # print(data["node_features"])
    # print(len(data["node_features"]))
    # input()

    # Example: Assume node_features_list, edge_indices_list, edge_features_list, etc., are loaded from CSV
    dataset = create_dataset(data["node_features"], data["edge_indices"], data["edge_features"], data["energy_output"])

    # Create a DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def run_training():

    batch_size = 3  # Adjust as needed

    dataset = data_preprocess(batch_size)

    epochs = 200

    node_size = 7
    node_hidden_size = 2
    node_output_size = 1

    edge_size = 2
    edge_hidden_size = 2
    edge_output_size = 1

    hidden_size1 = 8
    hidden_size2 = 8
    gnn_output_size = 1

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_train_list = []

    loss_func = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in dataset:  # Fix: Iterate over DataLoader
            node_features = batch.x  # All nodes across batched graphs
            edge_indices = batch.edge_index  # Connectivity across graphs
            edge_features = batch.edge_attr  # Edge attributes
            targets = batch.y  # Energy output labels

            output = model(node_features, edge_indices, edge_features, batch.batch)  # Pass batch index

            # print("Output: ", output)

            targets = targets.unsqueeze(1)
            # print("targets: ", targets)
            # input()

            # Compute loss (using MSE loss)
            loss = loss_func(output, targets)  # Fix: Proper loss calculation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_train_list.append(loss.item())  # Fix: Append individual loss, not accumulated loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {train_loss / len(dataset)}")  # Fix: Average loss

        # Fix: Plot loss after training, not during every epoch
    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Train Loss')
    plt.show()


run_training()
