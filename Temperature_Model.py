import ast
import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GlobalAttention

from Stats_Engineering import temperature_features, test_residual_normality

import matplotlib.pyplot as plt
from tkinter import filedialog
from collections import defaultdict

device = torch.device("cpu")

print("Device:", device)


class FCNN_FeatureCombiner(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(FCNN_FeatureCombiner, self).__init__()

        self.input = nn.Linear(input_dim, hidden_size)
        # self.input = nn.Linear(input_dim, output_dim)
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

        self.global_attention = GlobalAttention(gate_nn=nn.Linear(hidden_gnn_dim2, 1))

        self.relu = nn.ReLU()

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

        x = self.global_attention(x, batch)

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

    node_features = parse_column('Node Features Optimised Combined')
    edge_features = parse_column('Edge Features Optimised Combined')
    edge_indices = parse_column('Edge Indices Combined')
    system_features = parse_column('Temp. Input Features')
    temp_outputs = parse_column('Temp. Output Features')

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices,
        "system_features": system_features,
        "temp_outputs": temp_outputs
    }


def load_testing_data(csv_path):
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

    system_names = df[df.columns[0]].tolist()  # First column

    node_features = parse_entry('Node Features Optimised Combined')
    edge_features = parse_entry('Edge Features Optimised Combined')
    edge_indices = parse_entry('Edge Indices Combined')

    return {
        "system_names": system_names,
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_indices": edge_indices
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

    file_path = "temperature_training.csv"
    data = load_training_data(file_path)

    random.seed(int(time.time()))
    graph_indices = list(range(len(data['node_features'])))
    random.shuffle(graph_indices)

    if sample:
        test_system_names = ["Rh", "Cu", "LaNiO3"]  # adjust names to match those in your dataset

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

    train_node_feats = extract_by_indices(data, 'node_features', train_indices)
    train_edge_feats = extract_by_indices(data, 'edge_features', train_indices)

    train_temp_feats = [
        [float(t[0]), float(t[1])] for t in extract_by_indices(data, 'temp_outputs', train_indices)
    ]

    adsorption_temps_train = [[temp[0]] for temp in train_temp_feats]
    desorption_temps_train = [[temp[1]] for temp in train_temp_feats]

    node_normaliser = PreProcess()
    edge_normaliser = PreProcess()
    ads_normaliser = MinMaxNormalizer()
    des_normaliser = MinMaxNormalizer()

    flat_nodes_train, _ = flatten_graph_data(train_node_feats)
    flat_edges_train, _ = flatten_graph_data(train_edge_feats)
    flat_ads_train, _ = flatten_graph_data(adsorption_temps_train)
    flat_des_train, _ = flatten_graph_data(desorption_temps_train)

    node_normaliser.fit(flat_nodes_train, 4)  # Only use first 6 node features
    edge_normaliser.fit(flat_edges_train, len(flat_edges_train[0]))
    ads_normaliser.fit(flat_ads_train)
    des_normaliser.fit(flat_des_train)

    all_temp_feats = [
        [float(t[0]), float(t[1])] for t in data['temp_outputs']
    ]

    adsorption_temps_all = [[temp[0]] for temp in all_temp_feats]
    desorption_temps_all = [[temp[1]] for temp in all_temp_feats]

    flat_nodes, node_sizes = flatten_graph_data(data['node_features'])
    flat_edges, edge_sizes = flatten_graph_data(data['edge_features'])
    flat_ads, ads_sizes = flatten_graph_data(adsorption_temps_all)
    flat_des, des_sizes = flatten_graph_data(desorption_temps_all)

    node_features_norm_flat = node_normaliser.transform(flat_nodes)
    edge_features_norm_flat = edge_normaliser.transform(flat_edges)
    ads_norm_flat = ads_normaliser.transform(flat_ads)
    des_norm_flat = des_normaliser.transform(flat_des)

    node_features_norm = split_back(node_features_norm_flat, node_sizes)
    edge_features_norm = split_back(edge_features_norm_flat, edge_sizes)
    ads_norm = split_back(ads_norm_flat, ads_sizes)
    des_norm = split_back(des_norm_flat, des_sizes)

    edge_indices = [torch.tensor(ei, dtype=torch.long).T.to(device) for ei in data['edge_indices']]

    graph_list = []
    for i in range(len(node_features_norm)):

        temp_pair = torch.tensor(
            [ads_norm[i][0], des_norm[i][0]], dtype=torch.float
        ).to(device)

        data_obj = Data(
            x=torch.tensor(node_features_norm[i], dtype=torch.float).to(device),
            edge_index=edge_indices[i],
            edge_attr=torch.tensor(edge_features_norm[i], dtype=torch.float).to(device),
            y=temp_pair)

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

    epochs = 400

    node_size = 4
    node_hidden_size = 128
    node_output_size = 256

    edge_size = 1
    edge_hidden_size = 128
    edge_output_size = 256

    hidden_size1 = 256
    hidden_size2 = 512
    gnn_output_size = 2

    model = GNN(node_size, node_hidden_size, node_output_size,
                edge_size, edge_hidden_size, edge_output_size,
                hidden_size1, hidden_size2, gnn_output_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

            predicted_train_temps = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            loss = loss_func(predicted_train_temps, batch.y.view(-1, 2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        test_loss = 0.0

        all_predicted_ads = []
        all_predicted_des = []
        all_true_ads = []
        all_true_des = []
        all_system_names = []

        with torch.no_grad():
            for batch_test in test_loader:

                predicted_test_temps = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.batch)

                loss = loss_func(predicted_test_temps, batch_test.y.view(-1, 2))
                test_loss += loss.item()

                predicted_np = predicted_test_temps.cpu().numpy().reshape(-1, 2)
                true_np = batch_test.y.cpu().numpy().reshape(-1, 2)

                all_predicted_ads.extend(predicted_np[:, 0])
                all_predicted_des.extend(predicted_np[:, 1])

                all_true_ads.extend(true_np[:, 0])
                all_true_des.extend(true_np[:, 1])

                all_system_names.extend(batch_test.system_name)

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

    predicted_ads_test_temp_denorm = ads_normaliser.inverse_transform(all_predicted_ads)
    predicted_des_test_temp_denorm = des_normaliser.inverse_transform(all_predicted_des)

    true_ads_test_denorm = ads_normaliser.inverse_transform(all_true_ads)
    true_des_test_denorm = des_normaliser.inverse_transform(all_true_des)

    print("\n--- Predicted vs True Temperatures ---")
    for name, pred_ads, pred_des, true_ads, true_des in zip(
            all_system_names, predicted_ads_test_temp_denorm,
            predicted_des_test_temp_denorm, true_ads_test_denorm,
            true_des_test_denorm):
        print(f"{name:<30} | Pred Ads: {pred_ads:<10.6f} | True Ads: {true_ads:<10.6f}")
        print(f"{'':<30} | Pred Des: {pred_des:<10.6f} | True Des: {true_des:<10.6f}\n")

    stats_choice = input('Do you want to perform feature engineering and statistical tests?: ')

    if stats_choice == 'y':
        # ================== MAE CALCULATION PER SYSTEM ================== #
        mae_ads_per_system = defaultdict(list)
        mae_des_per_system = defaultdict(list)

        for name, pred_ads, pred_des, true_ads, true_des in zip(
                all_system_names,
                predicted_ads_test_temp_denorm,
                predicted_des_test_temp_denorm,
                true_ads_test_denorm,
                true_des_test_denorm):
            # Compute absolute errors
            mae_ads = abs(pred_ads - true_ads)
            mae_des = abs(pred_des - true_des)

            mae_ads_per_system[name].append(mae_ads)
            mae_des_per_system[name].append(mae_des)

        print("\n===== Mean Absolute Error (MAE) per System =====")
        for name in mae_ads_per_system:
            mean_ads = np.mean(mae_ads_per_system[name])
            mean_des = np.mean(mae_des_per_system[name])
            print(f"{name:<30} | Ads MAE: {mean_ads:.4f} | Des MAE: {mean_des:.4f}")

        # =================== FEATURE IMPORTANCE & ENGINEERING ================== #
        temperature_features(test_graphs, model)

        # =================== RESIDUAL NORMALITY TEST ================== #
        all_true_ads_tensor = torch.tensor(all_true_ads, dtype=torch.float32)
        all_pred_ads_tensor = torch.tensor(all_predicted_ads, dtype=torch.float32)

        all_true_des_tensor = torch.tensor(all_true_des, dtype=torch.float32)
        all_pred_des_tensor = torch.tensor(all_predicted_des, dtype=torch.float32)

        print("\nTesting residual normality for adsorption temperatures:")
        test_residual_normality(all_true_ads_tensor, all_pred_ads_tensor)

        print("\nTesting residual normality for desorption temperatures:")
        test_residual_normality(all_true_des_tensor, all_pred_des_tensor)

    else:
        pass

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
        }

        model_name = input('Input the model name: ')
        if not model_name:
            print("No model name entered. Exiting...")
            exit()

        model_dir = "Temperature GNN Models"  # Save the trained model in a folder along with its hyperparameters.
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, f"temperature_model_{model_name}.pt")

        # Bundle everything into one dictionary
        save_data = {
            "model_state_dict": model.state_dict(),
            "hyperparameters": hyperparameters,
            "normalisers": {
                'node_normaliser': node_normaliser,
                'edge_normaliser': edge_normaliser,
                'ads_normaliser': ads_normaliser,
                'des_normaliser': des_normaliser
            },
            "train_system_names": [g.system_name for g in train_graphs]
        }

        # Save everything
        torch.save(save_data, model_save_path)
        print(f"Model, hyperparameters, normalisers, and system names saved at: {model_save_path}")

    else:
        pass

    test_system_names = [g.system_name for g in test_graphs]  # or however you get the system names
    per_graph_mae = {}

    for name, true_ads, pred_ads, true_des, pred_des in zip(test_system_names, true_ads_test_denorm, predicted_ads_test_temp_denorm,
                                        true_des_test_denorm, predicted_des_test_temp_denorm):

        mae_ads = abs(true_ads - pred_ads)
        mae_des = abs(true_des - pred_des)

        per_graph_mae[name] = (mae_ads, mae_des)

    if sample:
        return per_graph_mae
    else:
        return None


def run_testing(name):

    test_file_path = f"Temperature Testing Data/{name}_temperature_testing.csv"
    test_data = load_testing_data(test_file_path)

    for i, graph in enumerate(test_data['node_features']):
        graph_array = np.array(graph)  # shape (num_nodes, num_features)
        coords = graph_array[:, :3]  # extract xyz
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
        graph_array[:, :3] = centered_coords
        test_data['node_features'][i] = graph_array.tolist()

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

    model_dir = "Temperature GNN Models"
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
    ads_normaliser = model_data["normalisers"]['ads_normaliser']
    des_normaliser = model_data["normalisers"]['des_normaliser']

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

    all_predicted_ads = []
    all_predicted_des = []

    all_system_names = []
    with torch.no_grad():
        for batch_test in test_loader:
            predicted_test_temps = model(batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.batch)

            predicted_np = predicted_test_temps.cpu().numpy().reshape(-1, 2)

            all_predicted_ads.extend(predicted_np[:, 0])
            all_predicted_des.extend(predicted_np[:, 1])

            all_system_names.extend(batch_test.system_name)

    predicted_ads_test_temp_denorm = ads_normaliser.inverse_transform(all_predicted_ads)
    predicted_des_test_temp_denorm = des_normaliser.inverse_transform(all_predicted_des)

    return predicted_ads_test_temp_denorm, predicted_des_test_temp_denorm
