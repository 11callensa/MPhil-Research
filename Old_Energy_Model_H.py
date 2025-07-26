import ast
import os
import numpy as np
import pandas as pd
import random
import time
import inspect

import torch
from torch_optimizer import AdaBelief
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from e3nn.o3 import Irreps
from e3nn.o3 import spherical_harmonics
from e3nn.nn import Gate
from e3nn.nn.models.gate_points_2101 import FullyConnectedTensorProduct, Convolution
from e3nn import o3

from Stats_Engineering import test_residual_normality

import matplotlib.pyplot as plt
from tkinter import filedialog
from collections import defaultdict

device = torch.device("mps")

print("Device:", device)


class RBFLayer(torch.nn.Module):
    def __init__(self, num_centers=10, cutoff=5.0):
        super().__init__()
        self.centers = torch.linspace(0, cutoff, num_centers)
        self.width = (self.centers[1] - self.centers[0]) * 1.0

    def forward(self, d):
        # d: [num_edges]
        d = d.unsqueeze(-1)  # [num_edges, 1]
        return torch.exp(- ((d - self.centers.to(d.device)) ** 2) / self.width**2)


class SimpleMACE(nn.Module):
    def __init__(self,layers, nodes, neighbours, embeddings,
                 input_irreps="1x0e",
                 hidden_irreps="20x0e + 20x1o",
                 output_irreps="1x0e"):
        super().__init__()

        self.input_irreps = Irreps(input_irreps)
        self.hidden_irreps = Irreps(hidden_irreps)
        self.output_irreps = Irreps(output_irreps)

        self.embedding = nn.Embedding(num_embeddings=embeddings, embedding_dim=self.input_irreps.dim)

        # Use full spherical harmonics: "1x0e + 1x1o" (dim=4)
        self.edge_attr_irreps = o3.Irreps.spherical_harmonics(lmax=0)  # = "1x0e + 1x1o"

        # Input tensor product: input × edge_attr → hidden
        self.tp_in = FullyConnectedTensorProduct(
            self.input_irreps,
            self.edge_attr_irreps,
            self.hidden_irreps
        )

        self.rbf = RBFLayer()

        # Convolution: hidden × edge_attr → rich intermediate features
        self.conv = Convolution(
            irreps_in=self.hidden_irreps,
            irreps_node_attr=self.hidden_irreps,
            irreps_edge_attr=self.edge_attr_irreps,
            irreps_out=Irreps("40x0e + 10x1o + 6x2e"),
            number_of_basis=10,
            radial_layers=layers,
            radial_neurons=nodes,
            num_neighbors=neighbours,
        )

        # Gate activation over scalar and vector channels
        scalars_list = [mulir for mulir in self.hidden_irreps if (mulir.ir.l == 0 and mulir.ir.p == 1)]
        vectors_list = [mulir for mulir in self.hidden_irreps if (mulir.ir.l == 1 and mulir.ir.p == -1)]

        scalars = Irreps(scalars_list)
        vectors = Irreps(vectors_list)

        self.gate = Gate(
            irreps_scalars=scalars,
            act_scalars=[nn.SiLU()] * len(scalars),
            act_gates=[nn.Sigmoid()] * len(scalars),
            irreps_gates=Irreps(f"{vectors.num_irreps}x0e"),
            irreps_gated=vectors,
        )

        # Output tensor product: gated × edge_attr → output
        self.tp_out = FullyConnectedTensorProduct(
            self.gate.irreps_out,
            self.edge_attr_irreps,
            self.output_irreps
        )

    def forward(self, node_feats, pos, edge_index, batch):

        # Make graph undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_src, edge_dst = edge_index

        # Edge vectors
        edge_vec = pos[edge_dst] - pos[edge_src]
        edge_length = edge_vec.norm(dim=-1)
        edge_unit = edge_vec / edge_length.unsqueeze(-1)

        edge_attr = spherical_harmonics(0, edge_unit, normalize=True, normalization='component')  # [num_edges, 3]

        node_feats = self.embedding(node_feats)

        x = self.tp_in(node_feats[edge_src], edge_attr)

        # RBF expansion
        edge_length_embedded = self.rbf(edge_length)

        # Convolution
        x = self.conv(x, x, edge_src, edge_dst, edge_attr, edge_length_embedded)

        x = x.to(dtype=torch.float32, device=device)

        # Gate nonlinearity
        x = self.gate(x)

        # Gather edge-wise features
        x_edge = x[edge_src]

        out = self.tp_out(x_edge, edge_attr)

        # Aggregate to graph-level prediction (e.g. energy)
        energy = scatter(out, batch[edge_src], dim=0, reduce='sum')

        return energy


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


def run_training(sample=False):

    file_path = "energy_training.csv"
    data = load_training_data(file_path)

    for i, edge_feat in enumerate(data['edge_features']):
        edge_array = np.array(edge_feat)  # shape (num_edges, num_edge_features)
        edge_array = edge_array[:, 0:1]  # keeps shape (num_edges, 1)
        data['edge_features'][i] = edge_array.tolist()

    for i, graph in enumerate(data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        graph_array = graph_array[:, [0, 1, 2, 6]]

        data['node_features'][i] = graph_array.tolist()

    random.seed(int(time.time()))
    graph_indices = list(range(len(data['node_features'])))
    random.shuffle(graph_indices)

    if sample:
        test_system_names = ["H-(Rh)", "H-(Cu)", "H-(LaNiO3)"]  # adjust names to match those in your dataset

        test_indices = []
        train_indices = []

        for i, system_name in enumerate(data["system_names"]):

            if system_name in test_system_names:
                test_indices.append(i)
            else:
                train_indices.append(i)

        train_sample_size = len(train_indices)
        train_indices = [random.choice(train_indices) for _ in range(train_sample_size)]

    else:
        num_train = 29
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

    node_normaliser.fit(flat_nodes_train, 4)  # Only use first 6 node features
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

    epochs = 700

    embeddings = 100
    nodes = 256
    layers = 4
    neighbours = 7

    model = SimpleMACE(layers, nodes, neighbours, embeddings).to(device)

    optimizer = AdaBelief(model.parameters(), lr=0.001)

    loss_train_list = []
    loss_test_list = []

    loss_func = nn.SmoothL1Loss()

    batch_size_train = 29
    batch_size_test = 1

    train_loader = DataLoader(train_graphs, batch_size_train, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size_test, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:

            z = batch.x[:, 3].long().to(device)
            pos = batch.x[:, 0:3].float().to(device)
            edge_index = batch.edge_index.to(device)
            batch_idx = batch.batch.to(device)

            predicted_train_energy = model(z, pos, edge_index, batch_idx)

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

                z = batch_test.x[:, 3].long().to(device)
                pos = batch_test.x[:, 0:3].float().to(device)
                edge_index = batch_test.edge_index.to(device)
                batch_idx = batch_test.batch.to(device)

                predicted_test_energy = model(z, pos, edge_index, batch_idx)

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

    stats_choice = input('Do you want to perform feature engineering and statistical tests?: ')

    if stats_choice == 'y':
        # ================== MAE CALCULATION PER SYSTEM ================== #
        mae_per_system = defaultdict(list)

        for name, pred_en, true_en in zip(
                all_system_names,
                predicted_test_energy_denorm,
                true_energy_denorm):
            mae = abs(pred_en - true_en)
            mae_per_system[name].append(mae)

        print("\n===== Mean Absolute Error (MAE) per System =====")
        for name in mae_per_system:
            mean = np.mean(mae_per_system[name])
            print(f"{name:<30} | MAE: {mean:.4f}")

        # =================== RESIDUAL NORMALITY TEST ================== #
        all_true = []
        all_pred = []

        for true_denorm, pred_denorm in zip(true_energy_denorm, predicted_test_energy_denorm):
            all_true.append(true_denorm)
            all_pred.append(pred_denorm)

        all_true_graph = torch.tensor(all_true, dtype=torch.float32)
        all_pred_graph = torch.tensor(all_pred, dtype=torch.float32)

        test_residual_normality(all_true_graph, all_pred_graph)

    save_option = input('Do you want to save this model?: ')

    if save_option == 'y':

        hyperparameters = {
            "layers": layers,
            "nodes": nodes,
            "neighbours": neighbours,
            "embeddings": embeddings
        }  # Store the hyperparameters in a dictionary for saving.

        model_name = input('Input the model name: ')
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        model_dir = "Energy H GNN Models"  # Save the trained model in a folder along with its hyperparameters.
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"energy_H_model_{model_name}.pt")

        # Bundle everything into one dictionary
        save_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
            "normalisers": {
                'node_normaliser': node_normaliser,
                'edge_normaliser': edge_normaliser,
                'energy_normaliser': energy_normaliser
            },
            "train_system_names": [g.system_name for g in train_graphs]
        }

        # Save everything
        torch.save(save_data, model_save_path)
        print(f"Model, hyperparameters, normalisers, and system names saved at: {model_save_path}")

    else:
        pass

    test_system_names = [g.system_name for g in test_graphs]  # or however you load them
    per_graph_mae = {}

    for name, true_e, pred_e in zip(test_system_names, true_energy_denorm, predicted_test_energy_denorm):
        mae = abs(true_e - pred_e)
        per_graph_mae[name] = mae

    if sample:
        return per_graph_mae
    else:
        return None


def run_testing(name):

    test_file_path = f"Energy Testing Data/{name}_energy_testing.csv"
    test_data = load_testing_data(test_file_path)

    for i, edge_feat in enumerate(test_data['edge_features']):
        edge_array = np.array(edge_feat)  # shape (num_edges, num_edge_features)
        edge_array = edge_array[:, 0:1]  # keeps shape (num_edges, 1)
        test_data['edge_features'][i] = edge_array.tolist()

    for i, graph in enumerate(test_data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        graph_array = graph_array[:, [0, 1, 2, 6]]
        test_data['node_features'][i] = graph_array.tolist()

    test_indices = list(range(len(test_data['node_features'])))

    test_node_feats = [test_data['node_features'][i] for i in test_indices]
    test_edge_feats = [test_data['edge_features'][i] for i in test_indices]

    flat_nodes_test, test_node_sizes = flatten_graph_data(test_node_feats)
    flat_edges_test, test_edge_sizes = flatten_graph_data(test_edge_feats)

    model_dir = "Energy H GNN Models"
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = torch.load(model_file_path)

    # Load normalisers from saved model data
    node_normaliser = model_data["normalisers"]['node_normaliser']
    edge_normaliser = model_data["normalisers"]['edge_normaliser']
    energy_normaliser = model_data["normalisers"]['energy_normaliser']

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

    hyperparameters = model_data["hyperparameters"]
    layers = hyperparameters["layers"]
    nodes = hyperparameters["nodes"]
    neighbours = hyperparameters["neighbours"]
    embeddings = hyperparameters["embeddings"]

    model = SimpleMACE(layers, nodes, neighbours, embeddings)

    model.to(device)

    model.load_state_dict(model_data["model_state_dict"])  # Load the state_dict (weights).
    model.eval()  # Set the model to evaluation mode.

    all_predicted = []
    all_system_names = []
    with torch.no_grad():
        for batch_test in test_loader:
            z = batch_test.x[:, 3].long().to(device)
            pos = batch_test.x[:, 0:3].float().to(device)
            edge_index = batch_test.edge_index.to(device)
            batch_idx = batch_test.batch.to(device)

            predicted_test_energy = model(z, pos, edge_index, batch_idx)

            all_predicted.extend(predicted_test_energy.cpu().numpy())
            all_system_names.extend(batch_test.system_name)

    predicted_test_energy_denorm = energy_normaliser.inverse_transform(all_predicted)

    print("\n--- Predicted vs True Energies ---")
    for name, pred in zip(all_system_names, predicted_test_energy_denorm):
        pred = -1 * pred
        print(f"{name:<30} | Predicted Energy: {float(pred):.6f} eV |")

    return pred
