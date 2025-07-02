import ast
import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.optim as optim
from torch_optimizer import AdaBelief
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SchNet
from torch_cluster import radius_graph

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


class LogCoshLoss(nn.Module):                                                                                           # Setup the log-cosh loss function.
    def forward(self, pred, target):
        """
            Takes in predicted and target values and calculates the log-cosh loss.

            :param pred: Predicted values.
            :param target: Target (true) values.
            :return: The log-cosh loss.
        """
        return torch.mean(torch.log(torch.cosh(pred - target + 1e-12)))


class GNN_SchNet(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.schnet = SchNet(hidden_channels=hidden_channels)  # returns node embeddings

        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, z, pos, batch):
        # Get node embeddings [num_nodes, hidden_channels]
        node_embeddings = self.schnet(z, pos, batch)

        return node_embeddings


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
                if isinstance(triple, list) and len(triple) >= 3:
                    extracted.append(triple[2])  # Get only the third entry
                else:
                    print(f"Warning: Row {i} in column '{col_name}' does not have 3 elements.")
            except Exception as e:
                print(f"Error parsing row {i} in column '{col_name}': {e}")
        return extracted

    system_names = parse_entry(df.columns[0], fix_strings=True)

    num_placed = []

    node_features = parse_entry('Node Features (Triple)')
    edge_features = parse_entry('Edge Features (Triple)')
    edge_indices = parse_entry('Edge Indices (Triple)')
    system_features = parse_entry('Energy Input Features (Triple)')
    energy_output = parse_entry('Energy Output Features (Triple)')
    num_placed.extend(df['Num. Placed H Atoms'].iloc[row] for row in range(len(df)))

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "energy_output": energy_output,
        "num_placed": num_placed
    }


def load_suppl_data(csv_path):

    df = pd.read_csv(csv_path)

    def parse_column(col_name, index):
        num_rows = len(df)
        all_graphs = []

        for i in range(num_rows):
            value = df[col_name].iloc[i]
            if index == False:
                try:
                    parsed_value = ast.literal_eval(value)  # Parses the string as list
                    all_graphs.append(parsed_value[0])  # One entry per graph
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing {col_name} in row {i}: {e}")
                    return None
            else:
                try:
                    parsed_value = ast.literal_eval(value)  # Parses the string as list
                    all_graphs.append(parsed_value)  # One entry per graph
                except (ValueError, SyntaxError) as e:
                    print(f"Error parsing {col_name} in row {i}: {e}")
                    return None

        return all_graphs  # List of [graph_1_data, graph_2_data, ...]

    num_placed = []

    system_names = df[df.columns[0]].tolist()

    node_features = parse_column('Node Features', False)
    edge_features = parse_column('Edge Features', False)
    edge_indices = parse_column('Edge Indices', True)
    energy_output = parse_column('Energy Output Features', False)
    num_placed.extend(df['Num. Placed H Atoms'].iloc[row] for row in range(len(df)))

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "energy_output": energy_output,
        "num_placed": num_placed
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
                    extracted.extend([triple[2]])  # Get first and second entries
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

    file_path = "../energy_training.csv"
    orig_data = load_training_data(file_path)

    # print('Orig data: ', orig_data['node_features'])
    # print('Orig data number of graphs: ', len(orig_data['node_features']))

    data = dict()

    supplementary_path = "../H_training_supplement.csv"
    suppl_data = load_suppl_data(supplementary_path)
    # suppl_data['node_features'] = [np.array(graph)[:, 3:6].tolist() for graph in suppl_data['node_features']]

    # print('Suppl Data: ', suppl_data['node_features'])
    # print('Suppl data number of graphs: ', len(suppl_data['node_features']))
    # input()

    data['node_features'] = orig_data['node_features']
    # data['node_features'] = [np.array(graph)[:, 3:6].tolist() for graph in orig_data['node_features']]
    data['edge_features'] = orig_data['edge_features']
    data['edge_indices'] = orig_data['edge_indices']
    data['energy_output'] = orig_data['energy_output']
    data['system_names'] = orig_data['system_names']
    data['num_placed'] = orig_data['num_placed']

    # data['node_features'] = [np.array(graph)[:, 3:6].tolist() for graph in suppl_data['node_features']]
    # data['node_features'] = suppl_data['node_features']
    # data['edge_features'] = suppl_data['edge_features']
    # data['edge_indices'] = suppl_data['edge_indices']
    # data['energy_output'] = suppl_data['energy_output']
    # data['system_names'] = suppl_data['system_names']
    # data['num_placed'] = suppl_data['num_placed']

    # data['node_features'].extend(suppl_data['node_features'])
    # data['edge_features'].extend(suppl_data['edge_features'])
    # data['edge_indices'].extend(suppl_data['edge_indices'])
    # data['energy_output'].extend(suppl_data['energy_output'])
    # data['system_names'].extend(suppl_data['system_names'])
    # data['num_placed'].extend(suppl_data['num_placed'])

    for i, graph in enumerate(data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        data['node_features'][i] = graph_array.tolist()

    # print('Data energy output: ', data['energy_output'])

    random.seed(int(time.time()))
    graph_indices = list(range(len(data['node_features'])))
    random.shuffle(graph_indices)

    num_train = 26
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
    train_num_placed = extract_by_indices(data, 'num_placed', train_indices)

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    energy_normaliser = MinMaxNormalizer()
    num_placed_normaliser = MinMaxNormalizer()

    flat_nodes_train, _ = flatten_graph_data(train_node_feats)
    flat_edges_train, _ = flatten_graph_data(train_edge_feats)

    node_normaliser.fit(flat_nodes_train, 6)  # Only use first 6 node features
    edge_normaliser.fit(flat_edges_train, len(flat_edges_train[0]))
    energy_normaliser.fit(train_energies)
    num_placed_normaliser.fit(train_num_placed)

    flat_nodes, node_sizes = flatten_graph_data(data['node_features'])
    flat_edges, edge_sizes = flatten_graph_data(data['edge_features'])

    node_features_norm_flat = node_normaliser.transform(flat_nodes)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges)
    energies_norm = energy_normaliser.transform(data['energy_output'])
    num_placed_norm = num_placed_normaliser.transform(data['num_placed'])

    node_features_norm = split_back(node_features_norm_flat, node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, edge_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            num_placed=torch.tensor(num_placed_norm[i], dtype=torch.float).to(device),
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

    epochs = 500

    schnet_nodes = 760

    model = GNN_SchNet(hidden_channels=schnet_nodes).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    loss_train_list = []
    loss_test_list = []

    loss_func = nn.MSELoss()
    # loss_func = nn.SmoothL1Loss()

    batch_size_train = 26
    batch_size_test = 1

    train_loader = DataLoader(train_graphs, batch_size_train, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size_test, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            z = batch.x[:, 4].long()  # atomic numbers
            pos = batch.x[:, 0:3].float()  # 3D coordinates
            batch_idx = batch.batch  # batch indices per atom

            predicted_train_energy = model(z, pos, batch_idx)

            loss = loss_func(predicted_train_energy, batch.y.view(-1, 1))

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
                z = batch_test.x[:, 4].long()  # atomic numbers
                pos = batch_test.x[:, 0:3].float()  # 3D coordinates
                batch_idx = batch_test.batch  # batch indices per atom

                # predicted_test_energy = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr,
                #                               batch_test.num_placed, batch_test.batch)

                predicted_test_energy = model(z, pos, batch_idx)
                loss = loss_func(predicted_test_energy, batch_test.y.view(-1, 1))
                test_loss += loss.item()

                all_predicted.extend(predicted_test_energy.cpu().numpy())
                all_system_names.extend(batch_test.system_name)
                all_true.extend(batch_test.y.cpu().numpy())

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
        print(f"{name:<30} | Predicted: {float(pred):.6f} | True: {float(true_val):.6f}")
        print(f"Percentage Error: {(true_val-pred)*100/(true_val)} %")

    save_option = input('Do you want to save this model?: ')

    if save_option == 'y':

        hyperparameters = {
            "schnet_nodes": schnet_nodes,
        }  # Store the hyperparameters in a dictionary for saving.

        model_name = input('Input the model name: ')
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        model_dir = "../Energy H GNN Models"  # Save the trained model in a folder along with its hyperparameters.
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"energy_H_model_{model_name}.pt")

        model_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters
        }  # Create a dictionary to store both the state_dict and hyperparameters.

        torch.save(model_data, model_save_path)  # Save the model data.
        print(f"Model and hyperparameters saved at: {model_save_path}")

    else:
        return None


def run_testing():

    train_path = "../energy_training.csv"
    train_data = load_training_data(train_path)

    train_indices = list(range(len(train_data['node_features'])))

    def extract_by_indices(data_dict, key, indices):
        return [data_dict[key][i] for i in indices]

    for i, graph in enumerate(train_data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        train_data['node_features'][i] = graph_array.tolist()

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

    for i, graph in enumerate(test_data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        test_data['node_features'][i] = graph_array.tolist()

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

    model_dir = "../Energy H GNN Models"  # Select the model to test.
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = torch.load(model_file_path)  # Load the model and hyperparameters.

    hyperparameters = model_data[
        "hyperparameters"]  # Retrieve the hyperparameters and use them to initialize the model.

    model = GNN_SchNet(
        hidden_channels=hyperparameters["schnet_nodes"],
    )

    model.to(device)

    model.load_state_dict(model_data["model_state_dict"])  # Load the state_dict (weights).
    model.eval()  # Set the model to evaluation mode.

    all_predicted = []
    all_system_names = []
    with torch.no_grad():
        for batch_test in test_loader:
            z = batch_test.x[:, 4].long()  # atomic numbers
            pos = batch_test.x[:, 0:3].float()  # 3D coordinates
            batch_idx = batch_test.batch  # batch indices per atom

            # predicted_test_energy = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr,
            #                               batch_test.num_placed, batch_test.batch)

            predicted_test_energy = model(z, pos, batch_idx)

            all_predicted.extend(predicted_test_energy.cpu().numpy())
            all_system_names.extend(batch_test.system_name)

    predicted_test_energy_denorm = energy_normaliser.inverse_transform(all_predicted)

    print("\n--- Predicted vs True Energies ---")
    for name, pred in zip(all_system_names, predicted_test_energy_denorm):
        pred = -1 * pred
        print(f"{name:<30} | Predicted Energy: {float(pred):.6f} eV |")

    return pred


run_training()
# run_testing()