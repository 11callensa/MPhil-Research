import ast
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader

from collections import defaultdict

import matplotlib.pyplot as plt

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

        self.input = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_dim)
        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.input(x)
        x = self.silu(x)
        x = self.fc1(x)

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

        self.fc_out = nn.Sequential(
            nn.Linear(hidden_gnn_dim2, 128),
            nn.SiLU(),
            nn.Linear(128, output_gnn_dim)
        )

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
        x = F.silu(x)

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


def run_training(bootstrap=False):
    file_path = "diffusion_training.csv"
    data = load_training_data(file_path)

    # Preprocessing (your existing code)
    element_lists = [[line.split()[0] for line in group] for group in data['output_coords']]

    data['displacements'] = []
    for i in range(len(data['output_coords'])):
        data['output_coords'][i] = [[float(x) for x in line.split()[1:]] for line in data['output_coords'][i]]
        data['initial_coords'][i] = [[float(x) for x in line.split()[1:]] for line in data['initial_coords'][i]]
        disp = np.array(data['output_coords'][i]) - np.array(data['initial_coords'][i])
        data['displacements'].append(disp.tolist())

    for i, graph in enumerate(data['node_features']):
        graph_array = np.array(graph)
        coords = graph_array[:, :3]
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        graph_array = graph_array[:, [0, 1, 2, 3, 4, 5, 6]]
        data['node_features'][i] = graph_array.tolist()

    # Split train/test
    test_formula_strings = ["Cu", "TiO2-A"]
    test_indices = [i for i, name in enumerate(data["system_names"]) if name in test_formula_strings]
    train_indices = [i for i, name in enumerate(data["system_names"]) if name not in test_formula_strings]

    # Monte Carlo resampling
    if bootstrap:
        train_sample_size = len(train_indices)
        train_indices = [random.choice(train_indices) for _ in range(train_sample_size)]

    # Extract data by indices
    def extract_by_indices(data_dict, key, indices):
        return [data_dict[key][i] for i in indices]

    train_data = {k: extract_by_indices(data, k, train_indices) for k in data.keys()}
    val_data = {k: extract_by_indices(data, k, test_indices) for k in data.keys()}

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

    # print("Train systems:")
    # for g in train_graphs:
    #     print(" -", g.system_name)
    #
    # print("\nTest systems:")
    # for g in test_graphs:
    #     print(" -", g.system_name)

    epochs = 3000

    node_size = 7
    node_hidden_size = 128
    node_output_size = 256

    edge_size = 2
    edge_hidden_size = 128
    edge_output_size = 256

    hidden_size1 = 128
    hidden_size2 = 256
    gnn_output_size = 3

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.8)

    loss_func = nn.SmoothL1Loss()

    batch_size_train = 30
    batch_size_test = 2

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

                true_coords_norm = batch.output_coords.clone()
                true_coords = coord_normaliser.inverse_process(true_coords_norm.cpu().numpy())
                true_coords = torch.tensor(true_coords, dtype=torch.float, device=device)

                true_disp_denorm = disp_normaliser.inverse_process(batch.y.cpu().numpy())
                final_disp_denorm = disp_normaliser.inverse_process(predicted_disp.cpu().numpy())

                true_disp_denorm = torch.tensor(true_disp_denorm, dtype=torch.float, device=device)
                final_disp_denorm = torch.tensor(final_disp_denorm, dtype=torch.float, device=device)

                all_true_final_coords = []
                all_pred_final_coords = []

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
                    true_coord_batch = true_coords[start_idx:end_idx]
                    pred_disp = final_disp_denorm[start_idx:end_idx]

                    pred_coords = init.clone()
                    pred_coords[movable_mask] = init[movable_mask] + pred_disp[movable_mask]

                    # Store movable atoms only
                    all_pred_final_coords.append(pred_coords[movable_mask].cpu())
                    all_true_final_coords.append(true_coord_batch[movable_mask].cpu())

                    # Existing storage
                    all_predicted_coords.append(pred_coords.cpu())
                    all_initial_coords.append(init.cpu())
                    all_pred_disps.append(pred_disp.cpu())
                    all_true_disps.append(batch.y[start_idx:end_idx].cpu())
                    all_movable_masks.append(movable_mask.cpu())
                    all_elements.append(elements)
                    all_names.append(batch.system_name[i])

                    start_idx = end_idx

        avg_test_loss = test_loss / total_movable_atoms_test

        # print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")

    all_true_final_coords = torch.cat(all_true_final_coords, dim=0)
    all_pred_final_coords = torch.cat(all_pred_final_coords, dim=0)

    val_metric = torch.mean(torch.abs(all_pred_final_coords - all_true_final_coords)).item()

    return val_metric



def monte_carlo_training(run_training_func, num_runs=2):
    """
    Performs Monte Carlo training by resampling the training data with replacement.
    Keeps validation data fixed.
    Args:
        run_training_func: your existing run_training function (modified to accept train indices)
        num_runs: number of Monte Carlo repetitions
    Returns:
        metrics_list: list of validation metrics for each run
    """
    metrics_list = []

    # Wrap loop with tqdm
    for run in tqdm(range(num_runs), desc="Monte Carlo Runs"):
        # Run your training function, but resample training data
        val_metric = run_training_func(bootstrap=True)
        metrics_list.append(val_metric)

    metrics_array = np.array(metrics_list)
    mean_metric = np.mean(metrics_array)
    ci_lower, ci_upper = np.percentile(metrics_array, [2.5, 97.5])

    print(f"Mean validation metric: {mean_metric:.4f}")
    print(f"95% confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    return metrics_list, (mean_metric, ci_lower, ci_upper)


# Example usage
metrics, ci = monte_carlo_training(run_training, num_runs=50)
