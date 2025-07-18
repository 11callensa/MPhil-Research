import ast
import os
import random
import time
import pickle
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader

from captum.attr import IntegratedGradients

from Compound_Properties import node_edge_features
from Stats_Engineering import disp_features, test_residual_normality

import Energy_Model_H
import Energy_Model_Combined
import Energy_Model_Compound
import NEW_Temperature_Model

import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog

device = torch.device("mps")

print("Device:", device)


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


class FCNN_FeatureCombiner(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(FCNN_FeatureCombiner, self).__init__()

        self.input = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(hidden_size, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.input(x)
        x = self.silu(x)
        # x = self.fc1(x)

        return x


class MyCustomGNNLayer(MessagePassing):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, aggr='add'):
        super().__init__(aggr=aggr)  # "add", "mean", or "max"

        self.message_mlp = nn.Sequential(
            nn.Linear(node_in_dim + edge_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim + node_in_dim, hidden_dim),
            nn.SiLU()
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
    def __init__(self,
                 node_dim, hidden_node_dim, output_node_dim,
                 edge_dim, hidden_edge_dim, output_edge_dim,
                 hidden_gnn_dim1, hidden_gnn_dim2, output_gnn_dim=3):
        super().__init__()

        self.node_feature_combiner = FCNN_FeatureCombiner(node_dim, hidden_node_dim, output_node_dim)
        self.edge_feature_combiner = FCNN_FeatureCombiner(edge_dim, hidden_edge_dim, output_edge_dim)

        self.gnn1 = MyCustomGNNLayer(output_node_dim, output_edge_dim, hidden_gnn_dim2)

        self.fc_out = nn.Linear(hidden_gnn_dim2, output_gnn_dim)

    def forward(self, node_features, edge_index, edge_features):

        node_features = F.relu(self.node_feature_combiner(node_features))
        edge_features = F.relu(self.edge_feature_combiner(edge_features))

        def make_undirected(edge_index, edge_attr):
            edge_index_reversed = edge_index.flip(0)
            edge_index_full = torch.cat([edge_index, edge_index_reversed], dim=1)
            edge_attr_full = torch.cat([edge_attr, edge_attr], dim=0)
            return edge_index_full, edge_attr_full

        edge_index, edge_features = make_undirected(edge_index, edge_features)

        x = self.gnn1(node_features, edge_index, edge_features)
        x = F.relu(x)

        displacement = self.fc_out(x)

        return displacement


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
        "num_fixed_atoms": num_fixed_atoms,
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
    oxidation_states = parse_column('Oxidation States')[0]

    num_fixed_atoms = df.iloc[:, 7].astype(int).tolist()

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "initial_coords": initial_coords,
        "num_fixed_atoms": num_fixed_atoms,
        "oxidation_states": oxidation_states
    }


def load_energy_file(csv_path):
    df = pd.read_csv(csv_path)

    def parse_entry(col_name, fix_strings=False):
        extracted = []
        for i, val in enumerate(df[col_name]):
            try:
                triple = ast.literal_eval(val)
                if isinstance(triple, list) and len(triple) >= 3:
                    extracted.extend([triple[0], triple[1], triple[2]])
                else:
                    print(f"Warning: Row {i} in column '{col_name}' does not have 3 elements.")
            except Exception as e:
                print(f"Error parsing row {i} in column '{col_name}': {e}")
        return extracted

    node_features = parse_entry('Node Features (Triple)')
    edge_features = parse_entry('Edge Features (Triple)')
    edge_indices = parse_entry('Edge Indices (Triple)')

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices
    }, df


def load_temp_file(csv_path):
    df = pd.read_csv(csv_path)

    def parse_entry(col_name):
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

    node_features = parse_entry('Node Features Optimised Combined')
    edge_features = parse_entry('Edge Features Optimised Combined')
    edge_indices = parse_entry('Edge Indices Combined')

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices
    }, df


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

    file_path = "diffusion_training.csv"
    data = load_training_data(file_path)  # Load the diffusion data

    element_lists = [[line.split()[0] for line in group] for group in data['output_coords']]

    data['displacements'] = []
    for i in range(len(data['output_coords'])):
        data['output_coords'][i] = [[float(x) for x in line.split()[1:]] for line in data['output_coords'][i]]
        data['initial_coords'][i] = [[float(x) for x in line.split()[1:]] for line in data['initial_coords'][i]]
        disp = np.array(data['output_coords'][i]) - np.array(data['initial_coords'][i])
        data['displacements'].append(disp.tolist())

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
        graph_array = graph_array[:, [0, 1, 2, 3, 4, 5, 6]]

        data['node_features'][i] = graph_array.tolist()

    # random.seed(int(time.time()))
    # graph_indices = list(range(len(data['node_features'])))
    # random.shuffle(graph_indices)
    #
    # num_train = 29
    # train_indices = graph_indices[:num_train]
    # test_indices = graph_indices[num_train:]

    test_systems = [
        {"Rh"},
        {"Cu"},
        {"La", "Ni", "O"}  # LaNiO₃
    ]

    test_indices = []
    train_indices = []

    for i, elements in enumerate(element_lists):
        # Remove all H atoms from the set
        core_elements = set(e for e in elements if e != "H")

        if core_elements in test_systems:
            test_indices.append(i)
        else:
            train_indices.append(i)

    def extract_by_indices(data_dict, key, indices):
        return [data_dict[key][i] for i in indices]

    train_node_feats = extract_by_indices(data, 'node_features', train_indices)
    train_edge_feats = extract_by_indices(data, 'edge_features', train_indices)
    train_coords = extract_by_indices(data, 'initial_coords', train_indices)
    train_output_coords = extract_by_indices(data, 'output_coords', train_indices)
    train_disps = extract_by_indices(data, 'displacements', train_indices)

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    coord_normaliser = PreProcess()
    output_coord_normaliser = PreProcess()
    disp_normaliser = PreProcess()

    flat_nodes_train, _ = flatten_graph_data(train_node_feats)
    flat_edges_train, _ = flatten_graph_data(train_edge_feats)
    flat_coords_train, _ = flatten_graph_data(train_coords)
    flat_output_coords_train, _ = flatten_graph_data(train_output_coords)
    flat_disps_train, _ = flatten_graph_data(train_disps)

    node_normaliser.fit(flat_nodes_train, 7)  # Only use first 6 node features
    edge_normaliser.fit(flat_edges_train, len(flat_edges_train[0]))
    coord_normaliser.fit(flat_coords_train, len(flat_coords_train[0]))
    output_coord_normaliser.fit(flat_coords_train, len(flat_output_coords_train[0]))
    disp_normaliser.fit(flat_disps_train, len(flat_disps_train[0]))

    flat_nodes, node_sizes = flatten_graph_data(data['node_features'])
    flat_edges, edge_sizes = flatten_graph_data(data['edge_features'])
    flat_coords, coord_sizes = flatten_graph_data(data['initial_coords'])
    flat_output_coords, output_coord_sizes = flatten_graph_data(data['output_coords'])
    flat_disps, disp_sizes = flatten_graph_data(data['displacements'])

    node_features_norm_flat = node_normaliser.transform(flat_nodes)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges)
    initial_coords_norm_flat = coord_normaliser.transform(flat_coords)
    output_coords_norm_flat = output_coord_normaliser.transform(flat_output_coords)
    disps_norm_flat = disp_normaliser.transform(flat_disps)

    node_features_norm = split_back(node_features_norm_flat, node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, edge_sizes)
    initial_coords_norm = split_back(initial_coords_norm_flat, coord_sizes)
    output_coords_norm = split_back(output_coords_norm_flat, output_coord_sizes)
    disps_norm = split_back(disps_norm_flat, disp_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            input_coords=torch.tensor(initial_coords_norm[i], dtype=torch.float).to(device),
            output_coords=torch.tensor(output_coords_norm[i], dtype=torch.float).to(device),
            y=torch.tensor(disps_norm[i], dtype=torch.float).to(device))

        data_obj.system_name = data['system_names'][i]
        data_obj.elements = element_lists[i]
        data_obj.num_fixed = data['num_fixed_atoms'][i]
        graph_list.append(data_obj)

    train_graphs = [graph_list[i] for i in train_indices]
    test_graphs = [graph_list[i] for i in test_indices]

    print("Train systems:")
    for g in train_graphs:
        print(" -", g.system_name)

    print("\nTest systems:")
    for g in test_graphs:
        print(" -", g.system_name)

    epochs = 1000

    node_size = 7
    node_hidden_size = 128
    node_output_size = 256

    edge_size = 1
    edge_hidden_size = 128
    edge_output_size = 256

    hidden_size1 = 128
    hidden_size2 = 1024
    gnn_output_size = 3

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8)

    loss_train_list = []
    loss_test_list = []

    loss_func = nn.SmoothL1Loss()

    batch_size_train = 29
    batch_size_test = 3

    train_loader = DataLoader(train_graphs, batch_size_train, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size_test, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        total_movable_atoms_train = 0

        for batch in train_loader:
            batch = batch.to(device)
            batch_elements = batch.elements
            batch_num_fixed = batch.num_fixed.to(device)

            predicted_disp = model(batch.x, batch.edge_index, batch.edge_attr)

            start_idx = 0
            movable_mask_global = torch.zeros(batch.y.size(0), dtype=torch.bool, device=batch.y.device)
            for i, elements in enumerate(batch_elements):
                N_fixed = batch_num_fixed[i].item()
                num_atoms = len(elements)
                end_idx = start_idx + num_atoms

                movable_mask = torch.tensor(
                    [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                    dtype=torch.bool,
                    device=batch.y.device
                )

                movable_mask_global[start_idx:end_idx] = movable_mask
                start_idx = end_idx

            pred_movable = predicted_disp[movable_mask_global]
            true_movable = batch.y[movable_mask_global]
            loss = loss_func(pred_movable, true_movable)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            movable_atoms_count = movable_mask_global.sum().item()
            train_loss += loss.item() * movable_atoms_count
            total_movable_atoms_train += movable_atoms_count

        avg_train_loss = train_loss / total_movable_atoms_train

        model.eval()
        test_loss = 0.0
        total_movable_atoms_test = 0

        all_predicted_coords = []
        all_elements = []
        all_names = []
        all_true_disps = []
        all_pred_disps = []
        all_initial_coords = []
        all_movable_masks = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch_elements = batch.elements
                batch_num_fixed = batch.num_fixed.to(device)

                predicted_disp = model(batch.x, batch.edge_index, batch.edge_attr)

                start_idx = 0
                movable_mask_global = torch.zeros(batch.y.size(0), dtype=torch.bool, device=batch.y.device)
                for i, elements in enumerate(batch_elements):
                    N_fixed = batch_num_fixed[i].item()
                    num_atoms = len(elements)
                    end_idx = start_idx + num_atoms

                    movable_mask = torch.tensor(
                        [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                        dtype=torch.bool,
                        device=batch.y.device
                    )

                    movable_mask_global[start_idx:end_idx].copy_(movable_mask)
                    start_idx = end_idx

                pred_movable = predicted_disp[movable_mask_global]
                true_movable = batch.y[movable_mask_global]

                loss = loss_func(pred_movable, true_movable)

                movable_atoms_count = movable_mask_global.sum().item()
                test_loss += loss.item() * movable_atoms_count
                total_movable_atoms_test += movable_atoms_count

                # Inverse normalization and reconstruct predicted coords per graph
                initial_coords_norm = batch.input_coords.clone()
                initial_coords = coord_normaliser.inverse_process(initial_coords_norm.cpu().numpy())
                initial_coords = torch.tensor(initial_coords, dtype=torch.float, device=device)

                true_disp_denorm = disp_normaliser.inverse_process(batch.y.cpu().numpy())
                final_disp_denorm = disp_normaliser.inverse_process(predicted_disp.cpu().numpy())

                true_disp_denorm = torch.tensor(true_disp_denorm, dtype=torch.float, device=device)
                final_disp_denorm = torch.tensor(final_disp_denorm, dtype=torch.float, device=device)

                start_idx = 0
                for i, elements in enumerate(batch_elements):
                    N_fixed = batch_num_fixed[i].item()
                    num_atoms = len(elements)
                    end_idx = start_idx + num_atoms

                    movable_mask = torch.tensor(
                        [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                        dtype=torch.bool,
                        device=device
                    )

                    init = initial_coords[start_idx:end_idx]

                    pred_disp = final_disp_denorm[start_idx:end_idx]
                    true_disp = batch.y[start_idx:end_idx]

                    pred_coords = init.clone()
                    pred_coords[movable_mask] = init[movable_mask] + pred_disp[movable_mask]

                    # Store all data for analysis
                    all_predicted_coords.append(pred_coords.cpu())
                    all_initial_coords.append(init.cpu())

                    all_pred_disps.append(pred_disp.cpu())
                    all_true_disps.append(true_disp.cpu())

                    all_movable_masks.append(movable_mask.cpu())
                    all_elements.append(elements)
                    all_names.append(batch.system_name[i])

                    start_idx = end_idx

        avg_test_loss = test_loss / total_movable_atoms_test

        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

        loss_train_list.append(avg_train_loss)
        loss_test_list.append(avg_test_loss)

    for i, (name, elements, coords) in enumerate(zip(all_names, all_elements, all_predicted_coords)):
        filename = f"Predicted Coords/{name}_predicted.xyz"
        with open(filename, "w") as f:
            f.write(f"{len(elements)}\n")
            f.write("0 1\n")
            for elem, coord in zip(elements, coords):
                f.write(f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

    # Plot loss curves
    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.plot(loss_test_list, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title('Train Loss')
    plt.show()

    stats_choice = input('Do you want to perform feature engineering and statistical tests?: ')

    if stats_choice == 'y':
        # =================== RESIDUAL NORMALITY TEST ================== #
        # Select only movable atoms' true and predicted displacements
        all_true_movable = []
        all_pred_movable = []

        for true_disp, pred_disp, movable_mask in zip(all_true_disps, all_pred_disps, all_movable_masks):
            all_true_movable.append(true_disp[movable_mask])
            all_pred_movable.append(pred_disp[movable_mask])

        # Concatenate across all graphs
        all_true_movable = torch.cat(all_true_movable, dim=0)
        all_pred_movable = torch.cat(all_pred_movable, dim=0)

        # Run residual normality test
        test_residual_normality(all_true_movable, all_pred_movable)

        # =================== FEATURE IMPORTANCE & ENGINEERING ================== #

        disp_features(test_graphs, model)

        # ---------- PER-GRAPH DISPLACEMENT ERROR ANALYSIS FOR MOVABLE ATOMS ONLY ----------
        idx = 2  # Graph index to inspect

        true_disp = all_true_disps[idx]
        movable_mask = all_movable_masks[idx]
        elements = all_elements[idx]
        name = all_names[idx]

        # Get start index into the full flat final_disp tensor
        start_idx = sum(len(e) for e in all_elements[:idx])
        end_idx = start_idx + len(elements)

        # Slice predicted displacements for this graph
        pred_disp_graph = final_disp_denorm[start_idx:end_idx]

        # Extract only movable atoms (usually H)
        pred_disp = pred_disp_graph[movable_mask.to(device)]
        true_disp = true_disp[movable_mask]

        # Compute % errors and MAE
        mae = torch.norm(pred_disp - true_disp.to(device), dim=1)  # Absolute error (L2) per atom
        perc_error = mae / (torch.norm(true_disp.to(device), dim=1) + 1e-8) * 100

        mae = mae.cpu().numpy()
        perc_error = perc_error.cpu().numpy()

        # Print detailed per-atom errors
        print(f"\nDisplacement Errors for Movable Atoms in Graph {name}:\n")
        movable_indices = [i for i, m in enumerate(movable_mask) if m]
        for i, (atom_idx, pe, abs_err) in enumerate(zip(movable_indices, perc_error, mae), start=1):
            print(f"H atom{atom_idx + 1} - % error: {pe:.2f}% | MAE: {abs_err:.3f} Å")

        # Summary
        print(f"\nGraph {name}: Mean % Error = {np.mean(perc_error):.2f}%, Max % Error = {np.max(perc_error):.2f}%")
        print(f"Mean MAE = {np.mean(mae):.3f} Å, Max MAE = {np.max(mae):.3f} Å")

        # Plot histogram
        plt.hist(perc_error, bins=20)
        plt.xlabel("Percentage Error")
        plt.ylabel("Frequency")
        plt.title(f"% Error Distribution for {name}")
        plt.show()

    else:
        pass

    save_option = input('Do you want to save this model?: ')

    if save_option == 'y':

        # Hyperparameters
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
        }

        # Ask for model name
        model_name = input('Input the model name: ')
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        # Create save directory
        model_dir = "Displacement GNN Models"
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"displacement_model_{model_name}.pt")

        # Bundle everything into one dictionary
        save_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
            "normalisers": {
                'node_normaliser': node_normaliser,
                'edge_normaliser': edge_normaliser,
                'coord_normaliser': coord_normaliser,
                'output_coord_normaliser': output_coord_normaliser,
                'disp_normaliser': disp_normaliser
            },
            "train_system_names": [g.system_name for g in train_graphs]
        }

        # Save everything
        torch.save(save_data, model_save_path)
        print(f"Model, hyperparameters, normalisers, and system names saved at: {model_save_path}")

    else:
        return None


def run_testing(name):
    """
    Opens the testing file, extracts and normalises data using loaded normalisers, and runs testing on the GNN.
    """
    test_file_path = f"Diffusion Testing Data/{name}_diffusion_testing.csv"
    test_data = load_testing_data(test_file_path)

    element_lists = [[line.split()[0] for line in group] for group in test_data['initial_coords']]

    for i in range(len(test_data['initial_coords'])):
        test_data['initial_coords'][i] = [[float(x) for x in line.split()[1:]] for line in test_data['initial_coords'][i]]

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
        graph_array = graph_array[:, [0, 1, 2, 3, 4, 5, 6]]
        test_data['node_features'][i] = graph_array.tolist()

    test_indices = list(range(len(test_data['node_features'])))

    test_node_feats = [test_data['node_features'][i] for i in test_indices]
    test_edge_feats = [test_data['edge_features'][i] for i in test_indices]
    test_coords = [test_data['initial_coords'][i] for i in test_indices]

    flat_nodes_test, test_node_sizes = flatten_graph_data(test_node_feats)
    flat_edges_test, test_edge_sizes = flatten_graph_data(test_edge_feats)
    flat_coords_test, test_coord_sizes = flatten_graph_data(test_coords)

    # -------------------- Load Model and Normalisers --------------------
    model_dir = "Displacement GNN Models"
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
    coord_normaliser = model_data["normalisers"]['coord_normaliser']
    disp_normaliser = model_data["normalisers"]['disp_normaliser']

    # -------------------- Normalise Testing Data --------------------
    node_features_norm_flat = node_normaliser.transform(flat_nodes_test)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges_test)
    initial_coords_norm_flat = coord_normaliser.transform(flat_coords_test)

    node_features_norm = split_back(node_features_norm_flat, test_node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, test_edge_sizes)
    initial_coords_norm = split_back(initial_coords_norm_flat, test_coord_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in test_data['edge_indices']]

    # -------------------- Create PyTorch Geometric Graphs --------------------
    graph_list = []
    for i in range(len(node_features_norm)):
        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            input_coords=torch.tensor(initial_coords_norm[i], dtype=torch.float).to(device)
        )
        data_obj.system_name = test_data['system_names'][i]
        data_obj.elements = element_lists[i]
        data_obj.num_fixed = test_data['num_fixed_atoms'][i]
        graph_list.append(data_obj)

    test_graphs = [graph_list[i] for i in test_indices]
    test_loader = DataLoader(test_graphs, 1, shuffle=False)

    print("\nTest system:")
    for g in test_graphs:
        print(" -", g.system_name)

    # -------------------- Load and Prepare Model --------------------
    hyperparameters = model_data["hyperparameters"]

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
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    all_predicted_coords = []
    all_elements = []
    all_names = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch_elements = batch.elements
            batch_num_fixed = batch.num_fixed

            predicted_disp = model(batch.x, batch.edge_index, batch.edge_attr)

            start_idx = 0
            movable_mask_global = torch.zeros(batch.input_coords.size(0), dtype=torch.bool, device=device)

            for i, elements in enumerate(batch_elements):
                N_fixed = batch_num_fixed[i].item()
                num_atoms = len(elements)
                end_idx = start_idx + num_atoms

                movable_mask = torch.tensor(
                    [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                    dtype=torch.bool,
                    device=device
                )

                movable_mask_global[start_idx:end_idx].copy_(movable_mask)
                start_idx = end_idx

            initial_coords_norm = batch.input_coords.clone()
            initial_coords = coord_normaliser.inverse_process(initial_coords_norm.cpu().numpy())
            final_disp = disp_normaliser.inverse_process(predicted_disp.cpu().numpy())

            initial_coords = torch.tensor(initial_coords, dtype=torch.float, device=device)
            final_disp = torch.tensor(final_disp, dtype=torch.float, device=device)

            start_idx = 0
            for i, elements in enumerate(batch_elements):
                N_fixed = batch_num_fixed[i].item()
                num_atoms = len(elements)
                end_idx = start_idx + num_atoms

                movable_mask = torch.tensor(
                    [e == 'H' and idx >= N_fixed for idx, e in enumerate(elements)],
                    dtype=torch.bool,
                    device=device
                )

                init = initial_coords[start_idx:end_idx]
                disp = final_disp[start_idx:end_idx]

                pred_coords = init.clone()
                pred_coords[movable_mask] = init[movable_mask] + disp[movable_mask]

                all_predicted_coords.append(pred_coords)
                all_elements.append(elements)
                all_names.append(batch.system_name[i])

                start_idx = end_idx

    predicted_xyz = []

    for i, (elements, coords) in enumerate(zip(all_elements, all_predicted_coords)):
        filename = f"Predicted Coords TESTING/{name}_predicted.xyz"

        xyz_string = f"{len(elements)}\n"
        xyz_string += "0 1\n"

        for elem, coord in zip(elements, coords):
            xyz_string += f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n"
            xyz_line = f"{elem} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}"
            predicted_xyz.append(xyz_line)

        with open(filename, "w") as f:
            f.write(xyz_string)

    print("Predicted Optimised XYZ: ", predicted_xyz)

    print('\n Creating Energy and Temperature Testing Files...')

    node_features_opt_pred_comb, edge_features_opt_pred_comb = node_edge_features(predicted_xyz, test_data['edge_indices'][0],
                                                                          test_data['oxidation_states'],
                                                                          test_data['num_fixed_atoms'][0], 0)

    print('Predicted Optimised Node Features: ', node_features_opt_pred_comb)
    print('Predicted Optimised Edge Features: ', edge_features_opt_pred_comb)

    print(test_data['num_fixed_atoms'])

    H_opt_pred_xyz = predicted_xyz[test_data['num_fixed_atoms'][0]:]  # Extract the optimised hydrogen positions.
    print("H Opt XYZ: ", H_opt_pred_xyz)

    energy_file_path = f"Energy Testing Data/{name}_energy_testing.csv"
    temp_file_path = f"Temperature Testing Data/{name}_temperature_testing.csv"

    energy_data, df_energy = load_energy_file(energy_file_path)
    temp_data, df_temp = load_temp_file(temp_file_path)

    node_features_opt_pred_H, edge_features_opt_pred_H = node_edge_features(H_opt_pred_xyz, energy_data['edge_indices'][2],
                                                                  test_data['oxidation_states'],
                                                                  test_data['num_fixed_atoms'][0],
                                                                  1)

    print('Predicted Optimised H Alone Node Features: ', node_features_opt_pred_H)
    print('Predicted Optimised H Alone Edge Features: ', edge_features_opt_pred_H)

    df_temp.loc[0, "Node Features Optimised Combined"] = str(node_features_opt_pred_comb)
    df_temp.loc[0, "Edge Features Optimised Combined"] = str(edge_features_opt_pred_comb)

    df_temp.to_csv(temp_file_path, index=False)

    old_node_features = energy_data['node_features']
    old_edge_features = energy_data['edge_features']

    new_node_features_energy = [
        node_features_opt_pred_comb,  # Combined (predicted)
        old_node_features[1],  # Keep existing crystal features
        node_features_opt_pred_H  # Hydrogen (predicted)
    ]

    new_edge_features_energy = [
        edge_features_opt_pred_comb,  # Combined (predicted)
        old_edge_features[1],  # Keep existing crystal features
        edge_features_opt_pred_H  # Hydrogen (predicted)
    ]

    df_energy.loc[0, "Node Features (Triple)"] = str(new_node_features_energy)
    df_energy.loc[0, "Edge Features (Triple)"] = str(new_edge_features_energy)

    df_energy.to_csv(energy_file_path, index=False)

    print(f"Updated {name} testing file successfully.")

    energy_choice = input(f'Do you want to predict the energies for {name}? y/n: ')

    if energy_choice == 'y':
        pred_combined_energy = Energy_Model_Combined.run_testing(name)
        pred_compound_energy = Energy_Model_Compound.run_testing(name)
        pred_H_energy = Energy_Model_H.run_testing(name)

        for comb, compound, H in zip(pred_combined_energy, pred_compound_energy, pred_H_energy):
            adsorption_energy = comb - (compound + H)

            print(f"{name:<30} | Predicted Adsorption Energy: {float(adsorption_energy):.6f} eV |")

    temp_choice = input(f'Do you want to predict adsorption & desorption temperatures for {name}? y/n: ')

    if temp_choice == 'y':
        print('\nSelect an Adsorption Model First')

        ads_prediction = NEW_Temperature_Model.run_testing(name)

        print("\n--- Predicted Adsorption Temperature ---")
        print(f"{name:<30} | Predicted Adsorption Temperature: {ads_prediction:<10.6f}")
        print('\n Now select a Desorption Model')

        des_prediction = NEW_Temperature_Model.run_testing(name)

        print("\n--- Predicted Desorption Temperature ---")
        print(f"{name:<30} | Predicted Desorption Temperature: {des_prediction:<10.6f}")
        print('\n Predicting complete')
