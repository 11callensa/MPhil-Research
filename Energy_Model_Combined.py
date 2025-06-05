import ast
import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_add_pool

import matplotlib.pyplot as plt
from tkinter import filedialog

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")

device = torch.device("cpu")

print("Device:", device)


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
        x = self.relu(x)

        return x


class MyCustomGNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, aggr='add'):
        super().__init__(aggr=aggr)  # "add", "mean", or "max"

        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_in_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, node_in_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_dim]
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j: [num_edges, node_in_dim] = features of neighbor nodes
        # edge_attr: [num_edges, edge_in_dim]
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        # aggr_out: [num_nodes, hidden_dim]
        # x: [num_nodes, node_in_dim]
        return self.update_mlp(torch.cat([aggr_out, x], dim=-1))


class GNN(nn.Module):
    def __init__(self, node_dim, hidden_node_dim, output_node_dim,
                 edge_dim, hidden_edge_dim, output_edge_dim,
                 hidden_gnn_dim1, hidden_gnn_dim2, output_gnn_dim):
        super(GNN, self).__init__()

        self.node_feature_combiner = FCNN_FeatureCombiner(input_dim=node_dim,
                                                          hidden_size=hidden_node_dim,
                                                          output_dim=output_node_dim)

        self.edge_feature_combiner = FCNN_FeatureCombiner(input_dim=edge_dim,
                                                          hidden_size=hidden_edge_dim,
                                                          output_dim=output_edge_dim)

        self.gcn1 = MyCustomGNNLayer(output_node_dim, output_edge_dim, hidden_gnn_dim1)
        self.gcn2 = MyCustomGNNLayer(hidden_gnn_dim1, output_edge_dim, hidden_gnn_dim2)
        self.gcn3 = MyCustomGNNLayer(hidden_gnn_dim2, output_edge_dim, hidden_gnn_dim2)

        self.fc_out = nn.Linear(hidden_gnn_dim2, output_gnn_dim)

    def forward(self, node_features, edge_index, edge_features, batch):
        """
        Forward pass through the GNN.

        node_features: Tensor of shape (num_nodes, feature_dim).
        edge_index: Tensor of shape (2, num_edges).
        edge_features: Tensor of shape (num_edges, edge_feature_dim).

        Returns:
        output: Predicted output features for each node.
        """

        node_features = self.node_feature_combiner(node_features)
        edge_features = self.edge_feature_combiner(edge_features).squeeze(-1)  # Ensure shape is (num_edges,)

        edge_index = edge_index.to(torch.long)

        def make_undirected(edge_index, edge_attr):
            edge_index_reversed = edge_index.flip(0)
            edge_index_full = torch.cat([edge_index, edge_index_reversed], dim=1)
            edge_attr_full = torch.cat([edge_attr, edge_attr], dim=0)
            return edge_index_full, edge_attr_full

        edge_index, edge_features = make_undirected(edge_index, edge_features)

        x = self.gcn1(node_features, edge_index, edge_features)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_features)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  # shape: [num_graphs, hidden_dim]

        output = self.fc_out(x)

        return output


def load_training_data(csv_path):
    df = pd.read_csv(csv_path)

    def fix_unquoted_list_string(s):
        """Adds quotes to unquoted elements in a list-like string."""
        if not isinstance(s, str): return s
        s = s.strip("[]")  # remove brackets
        elements = [x.strip() for x in s.split(",")]
        quoted = [f"'{x}'" if not (x.startswith("'") or x.startswith('"')) else x for x in elements]
        return "[" + ", ".join(quoted) + "]"

    def parse_entry(col_name, fix_strings=False):
        extracted = []
        for i, val in enumerate(df[col_name]):
            try:
                if fix_strings:
                    val = fix_unquoted_list_string(val)
                triple = ast.literal_eval(val)
                if isinstance(triple, list) and len(triple) >= 2:
                    extracted.extend([triple[0]])  # Get first entry
                else:
                    print(f"Warning: Row {i} in column '{col_name}' does not have at least 2 elements.")
            except Exception as e:
                print(f"Error parsing row {i} in column '{col_name}': {e}")
        return extracted

    # Fix strings only in the system name column
    system_names = parse_entry(df.columns[0], fix_strings=True)

    # All other data columns (numeric, structural) are parsed raw
    node_features = parse_entry('Node Features (Triple)')
    edge_features = parse_entry('Edge Features (Triple)')
    edge_indices = parse_entry('Edge Indices (Triple)')
    system_features = parse_entry('Energy Input Features (Triple)')
    energy_output = parse_entry('Energy Output Features (Triple)')

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "energy_output": energy_output
    }


def load_testing_data(csv_path):
    df = pd.read_csv(csv_path)

    def fix_unquoted_list_string(s):
        """Adds quotes to unquoted elements in a list-like string."""
        if not isinstance(s, str): return s
        s = s.strip("[]")  # remove brackets
        elements = [x.strip() for x in s.split(",")]
        quoted = [f"'{x}'" if not (x.startswith("'") or x.startswith('"')) else x for x in elements]
        return "[" + ", ".join(quoted) + "]"

    def parse_entry(col_name, fix_strings=False):
        extracted = []
        for i, val in enumerate(df[col_name]):
            try:
                if fix_strings:
                    val = fix_unquoted_list_string(val)
                triple = ast.literal_eval(val)
                if isinstance(triple, list) and len(triple) >= 2:
                    extracted.extend([triple[0]])  # Get first and second entries
                else:
                    print(f"Warning: Row {i} in column '{col_name}' does not have at least 2 elements.")
            except Exception as e:
                print(f"Error parsing row {i} in column '{col_name}': {e}")
        return extracted

    # Fix strings only in the system name column
    system_names = parse_entry(df.columns[0], fix_strings=True)

    # All other data columns (numeric, structural) are parsed raw
    node_features = parse_entry('Node Features (Triple)')
    edge_features = parse_entry('Edge Features (Triple)')
    edge_indices = parse_entry('Edge Indices (Triple)')
    system_features = parse_entry('Energy Input Features')

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
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


class MinMaxNormalizer:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        """
        Computes the min and max for a single feature across all graphs.
        """
        data = np.array(data)
        self.min = np.min(data)
        self.max = np.max(data)

    def transform(self, data):
        """
        Normalize a single feature using min-max scaling.
        """
        return [(f - self.min) / (self.max - self.min + 1e-8) for f in data]

    def fit_transform(self, data):
        """
        Fit and transform the data in one step.
        """
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        """
        Reverse the min-max normalization.
        """
        return [f * (self.max - self.min + 1e-8) + self.min for f in data]


class PreProcess:
    def __init__(self):
        self.normalizers = []
        self.num_feats = None

    def fit(self, data, num_feats):
        self.normalizers = []
        self.num_feats = num_feats
        normalized_list = []

        for i in range(num_feats):
            normalizer = MinMaxNormalizer()
            self.normalizers.append(normalizer)
            feature_list = [data[j][i] for j in range(len(data))]
            normalizer.fit(feature_list)

    def transform(self, data):
        if self.num_feats is None:
            raise ValueError("Must call fit() before transform().")

        normalized_list = []

        for i in range(self.num_feats):
            feature_list = [data[j][i] for j in range(len(data))]
            normalized_feature = self.normalizers[i].transform(feature_list)
            normalized_list.append(normalized_feature)

        # Recombine
        normalized_features = list(zip(*normalized_list))
        return [list(tup) for tup in normalized_features]

    def inverse_process(self, normalized_data):
        transposed = list(zip(*normalized_data))
        denormalized_list = []

        for i, norm_feat in enumerate(transposed):
            if i >= len(self.normalizers):
                raise IndexError(f"Index {i} is out of range for normalizers.")
            original_feat = self.normalizers[i].inverse_transform(list(norm_feat))
            denormalized_list.append(original_feat)

        return [list(tup) for tup in zip(*denormalized_list)]


def run_training():

    file_path = "energy_training.csv"
    data = load_training_data(file_path)

    random.seed(int(time.time()))
    graph_indices = list(range(len(data['node_features'])))
    random.shuffle(graph_indices)

    num_train = len(graph_indices) - 1
    print('No. of training graphs: ', num_train)
    train_indices = graph_indices[:num_train]
    test_indices = graph_indices[num_train:]

    def extract_by_indices(data_dict, key, indices):
        return [data_dict[key][i] for i in indices]

    for i, energy in enumerate(data['energy_output']):
        data['energy_output'][i] = -1 * data['energy_output'][i]

    train_node_feats = extract_by_indices(data, 'node_features', train_indices)
    train_edge_feats = extract_by_indices(data, 'edge_features', train_indices)
    train_energies = extract_by_indices(data, 'energy_output', train_indices)

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    energy_normaliser = MinMaxNormalizer()

    flat_nodes_train, _ = flatten_graph_data(train_node_feats)
    flat_edges_train, _ = flatten_graph_data(train_edge_feats)

    node_normaliser.fit(flat_nodes_train, 6)  # Only use first 6 node features
    edge_normaliser.fit(flat_edges_train, len(flat_edges_train[0]))
    energy_normaliser.fit(train_energies)

    flat_nodes, node_sizes = flatten_graph_data(data['node_features'])
    flat_edges, edge_sizes = flatten_graph_data(data['edge_features'])

    node_features_norm_flat = node_normaliser.transform(flat_nodes)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges)
    energies_norm = energy_normaliser.transform(data['energy_output'])

    node_features_norm = split_back(node_features_norm_flat, node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, edge_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            y=torch.tensor(energies_norm[i], dtype=torch.float).to(device))

        data_obj.system_name = data['system_names'][i]
        graph_list.append(data_obj)

    train_graphs = [graph_list[i] for i in train_indices]
    test_graphs = [graph_list[i] for i in test_indices]

    print("Train systems:")
    for g in train_graphs:
        print(" -", g.system_name)

    print("\nTest systems:")
    for g in test_graphs:
        print(" -", g.system_name)

    # print("\nTrain systems with normalized energies:")
    # for i in train_indices:
    #     system_name = data['system_names'][i]
    #     energies = data['energy_output'][i]
    #     energy_norm = float(energies_norm[i])
    #     print(f"{system_name:<30} | Un-normalised Energy: {energies:.6f} | Normalised Energy: {energy_norm:.6f}")

    epochs = 700

    batch_size = 4  # Adjust as needed

    node_size = 6
    node_hidden_size = 128
    node_output_size = 256

    edge_size = 2
    edge_hidden_size = 128
    edge_output_size = 128

    hidden_size1 = 64
    hidden_size2 = 128
    gnn_output_size = 1

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8)

    loss_train_list = []
    loss_test_list = []

    # loss_func = nn.SmoothL1Loss()
    loss_func = nn.MSELoss()

    train_loader = DataLoader(train_graphs, batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:

            predicted_train_energy = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            # loss = loss_func(predicted_train_energy, batch.y.view(-1, 1))
            loss = loss_func(torch.log1p(predicted_train_energy), torch.log1p(batch.y.view(-1, 1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0
        all_predicted = []
        all_system_names = []
        all_true = []
        with torch.no_grad():
            for batch_test in test_loader:

                predicted_test_energy = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.batch)

                # loss = loss_func(predicted_test_energy, batch_test.y.view(-1, 1))
                loss = loss_func(torch.log1p(predicted_test_energy), torch.log1p(batch_test.y.view(-1, 1)))
                test_loss += loss.item()

                all_predicted.extend(predicted_test_energy.numpy())
                all_system_names.extend(batch_test.system_name)
                all_true.extend(batch_test.y.numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = test_loss / len(test_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

        loss_train_list.append(avg_train_loss)
        loss_test_list.append(avg_test_loss)

    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.plot(loss_test_list, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'Train Loss')
    plt.show()

    predicted_test_energy_denorm = energy_normaliser.inverse_transform(all_predicted)
    true_energy_denorm = energy_normaliser.inverse_transform(all_true)

    print("\n--- Predicted vs True Energies ---")
    for name, pred, true_val in zip(all_system_names, predicted_test_energy_denorm, true_energy_denorm):

        pred = -1 * pred
        true_val = -1 * true_val

        print(f"{name:<30} | Predicted: {float(pred):.6f} eV | True: {float(true_val):.6f} eV")
        print(f"Percentage Error: {(true_val - pred) * 100 / (true_val)} %")

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
        }  # Store the hyperparameters in a dictionary for saving.

        model_name = input('Input the model name: ')
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        model_dir = "Energy Combined GNN Models"  # Save the trained model in a folder along with its hyperparameters.
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"energy_combined_model_{model_name}.pt")

        model_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters
        }  # Create a dictionary to store both the state_dict and hyperparameters.

        torch.save(model_data, model_save_path)  # Save the model data.
        print(f"Model and hyperparameters saved at: {model_save_path}")

    else:
        return None


def run_testing():

    train_path = "energy_training.csv"
    train_data = load_training_data(train_path)

    train_indices = list(range(len(train_data['node_features'])))

    def extract_by_indices(data_dict, key, indices):
        return [data_dict[key][i] for i in indices]

    for i, energy in enumerate(train_data['energy_output']):
        train_data['energy_output'][i] = -1 * train_data['energy_output'][i]

    train_node_feats = extract_by_indices(train_data, 'node_features', train_indices)
    train_edge_feats = extract_by_indices(train_data, 'edge_features', train_indices)
    train_energies = extract_by_indices(train_data, 'energy_output', train_indices)

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    energy_normaliser = MinMaxNormalizer()

    flat_nodes_train, _ = flatten_graph_data(train_node_feats)
    flat_edges_train, _ = flatten_graph_data(train_edge_feats)

    node_normaliser.fit(flat_nodes_train, 6)  # Only use first 6 node features
    edge_normaliser.fit(flat_edges_train, len(flat_edges_train[0]))
    energy_normaliser.fit(train_energies)

    name = 'NiO'

    test_file_path = f"Energy Testing Data/{name}_energy_testing.csv"
    test_data = load_testing_data(test_file_path)

    test_indices = list(range(len(test_data['node_features'])))

    test_node_feats = extract_by_indices(test_data, 'node_features', test_indices)
    test_edge_feats = extract_by_indices(test_data, 'edge_features', test_indices)

    flat_nodes_test, test_node_sizes = flatten_graph_data(test_node_feats)
    flat_edges_test, test_edge_sizes = flatten_graph_data(test_edge_feats)

    node_features_norm_flat = node_normaliser.transform(flat_nodes_test)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges_test)

    node_features_norm = split_back(node_features_norm_flat, test_node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, test_edge_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in test_data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device))

        data_obj.system_name = test_data['system_names'][i]
        graph_list.append(data_obj)

    test_graphs = [graph_list[i] for i in test_indices]

    test_loader = DataLoader(test_graphs, 1, shuffle=False)

    print("\nTest system:")
    for g in test_graphs:
        print(" -", g.system_name)

    model_dir = "Energy Combined GNN Models"    # Select the model to test.
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = torch.load(model_file_path)  # Load the model and hyperparameters.

    hyperparameters = model_data[
        "hyperparameters"]  # Retrieve the hyperparameters and use them to initialize the model.

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

    model.to(device)

    model.load_state_dict(model_data["model_state_dict"])  # Load the state_dict (weights).
    model.eval()  # Set the model to evaluation mode.

    all_predicted = []
    all_system_names = []
    with torch.no_grad():
        for batch_test in test_loader:
            predicted_test_energy = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.batch)

            all_predicted.extend(predicted_test_energy.numpy())
            all_system_names.extend(batch_test.system_name)

    predicted_test_energy_denorm = energy_normaliser.inverse_transform(all_predicted)

    print("\n--- Predicted vs True Energies ---")
    for name, pred in zip(all_system_names, predicted_test_energy_denorm):
        pred = -1 * pred
        print(f"{name:<30} | Predicted Energy: {float(pred):.6f} eV |")

    return pred
