import ast
import os
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("mps")

print("Device:", device)


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_size2, output_dim):
        super(FCNN, self).__init__()

        self.input = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_dim)

        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.input(x)
        x = self.silu(x)
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.silu(x)

        return x


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


class Temp_PreProcess:
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


def load_training_data(csv_path):
    df = pd.read_csv(csv_path)

    # Filter out rows where the first column (system name) is 'CuCr2O4'
    df = df[df[df.columns[0]] != 'CuCr2O4'].reset_index(drop=True)

    def parse_column(col_name):
        all_graphs = []

        for i, value in enumerate(df[col_name]):
            try:
                parsed_value = ast.literal_eval(value)  # Parses string to list
                all_graphs.append(parsed_value)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing {col_name} in row {i}: {e}")
                return None

        return all_graphs

    system_names = df[df.columns[0]].tolist()  # First column
    system_features = parse_column('Temp. Input Features')
    uncertain_features = parse_column('Uncertain Features')
    temp_outputs = parse_column('Temp. Output Features')

    return {
        "system_names": system_names,
        "system_features": system_features,
        "uncertain_features": uncertain_features,
        "temp_outputs": temp_outputs
    }


def run_training(select, mode='ads', sample=True):

    file_path = "temperature_training.csv"
    data = load_training_data(file_path)

    raw_inputs = [
        data['uncertain_features'][i][:3] + [data['system_features'][i][4]]
        for i in range(len(data['system_features']))
    ]

    for i in range(len(raw_inputs)):
        raw_inputs[i] = [raw_inputs[i][j] for j in select]

    if mode == 'ads':
        raw_outputs = [[float(pair[0])] for pair in data['temp_outputs']]
    elif mode == 'des':
        raw_outputs = [[float(pair[1])] for pair in data['temp_outputs']]
    else:
        raise ValueError("mode must be either 'ads' or 'des'")

    system_names = data['system_names']

    if sample:
        test_materials = ["Pd", "Au"]  # Fixed test set

        # Separate test set
        X_test, y_test, names_test = [], [], []
        train_pool = []

        for i in range(len(system_names)):
            name = system_names[i]
            input_row = raw_inputs[i]
            output_row = raw_outputs[i]

            if name in test_materials:
                X_test.append(input_row)
                y_test.append(output_row)
                names_test.append(name)
            else:
                train_pool.append((input_row, output_row, name))

        # Sample training set with replacement
        num_train_samples = len(raw_inputs) - len(X_test)  # or set a fixed number
        sampled_train = random.choices(train_pool, k=num_train_samples)

        X_train = [x for x, _, _ in sampled_train]
        y_train = [y for _, y, _ in sampled_train]
        names_train = [n for _, _, n in sampled_train]

    else:
        pass

    input_normalizer = Temp_PreProcess()
    input_normalizer.fit(X_train, num_feats=len(select))

    X_train_norm = input_normalizer.transform(X_train)
    X_test_norm = input_normalizer.transform(X_test)

    output_normalizer = Temp_PreProcess()
    output_normalizer.fit(y_train, num_feats=1)

    y_train_norm = output_normalizer.transform(y_train)
    y_test_norm = output_normalizer.transform(y_test)

    X_train_norm = torch.FloatTensor(X_train_norm)
    X_test_norm = torch.FloatTensor(X_test_norm)

    y_train_norm = torch.FloatTensor(y_train_norm)
    y_test_norm = torch.FloatTensor(y_test_norm)

    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    test_dataset = TensorDataset(X_test_norm, y_test_norm)

    input_size = len(select)
    hidden_size = 128
    hidden_size2 = 2048
    output_size = 1

    batch_size = 34

    model = FCNN(input_size, hidden_size, hidden_size2, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loss_func = nn.SmoothL1Loss()

    epochs = 2000
    # epochs = 20
    loss_train_list = []
    loss_test_list = []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            pred = model(X_batch).squeeze()
            y_batch = y_batch.squeeze()
            loss = loss_func(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * X_batch.size(0)

        train_loss = epoch_train_loss / len(train_loader.dataset)

        model.eval()
        epoch_test_loss = 0.0
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                pred = model(X_batch).squeeze()
                y_batch = y_batch.squeeze()
                loss = loss_func(pred, y_batch)
                epoch_test_loss += loss.item() * X_batch.size(0)
                all_preds.append(pred.unsqueeze(0))

        test_loss = epoch_test_loss / len(test_loader.dataset)

        loss_train_list.append(train_loss)
        loss_test_list.append(test_loss)

    return train_loss, test_loss


FEATURE_NAMES = {0: "Bulk", 1: "Shear", 2: 'Poisson', 3: 'Energy'}

selections = [[0, 1, 2, 3],
              [0, 1, 2],
              [0, 1, 3],
              [0, 2, 3],
              [1, 2, 3],
              [0, 1],
              [0, 2],
              [0, 3],
              [1, 2],
              [1, 3],
              [2, 3],
              [0],
              [1],
              [2],
              [3]
             ]

results = []

pbar = tqdm(total=len(selections), desc="Training combos")

for select in selections:

    train_losses = []
    test_losses = []

    for i in range(3):
        train_loss, test_loss = run_training(select)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    avg_train = sum(train_losses) / len(train_losses)
    avg_test = sum(test_losses) / len(test_losses)

    result = {
        "features": select,
        "avg_train_loss": avg_train,
        "avg_test_loss": avg_test
    }

    results.append(result)
    pbar.update(1)

pbar.close()


def feature_names(indices, name_map):
    return [name_map[i] for i in indices]

# Sort results
best_train = min(results, key=lambda x: x["avg_train_loss"])
best_test = min(results, key=lambda x: x["avg_test_loss"])
best_combined = min(results, key=lambda x: x["avg_train_loss"] + x["avg_test_loss"])

# Sort worst results
worst_train = max(results, key=lambda x: x["avg_train_loss"])
worst_test = max(results, key=lambda x: x["avg_test_loss"])
worst_combined = max(results, key=lambda x: x["avg_train_loss"] + x["avg_test_loss"])

# Print results
print("\nBest Average Train Loss:")
print(f"  Train Loss: {best_train['avg_train_loss']:.5f}")
print(f"  Features: {feature_names(best_train['features'], FEATURE_NAMES)}")


print("\nBest Average Test Loss:")
print(f"  Test Loss: {best_test['avg_test_loss']:.5f}")
print(f"  Features: {feature_names(best_test['features'], FEATURE_NAMES)}")

print("\nBest Combined (Train + Test) Loss:")
print(f"  Total Loss: {(best_combined['avg_train_loss'] + best_combined['avg_test_loss']):.5f}")
print(f"  Train Loss: {best_combined['avg_train_loss']:.5f}")
print(f"  Test Loss: {best_combined['avg_test_loss']:.5f}")
print(f"  Features: {feature_names(best_combined['features'], FEATURE_NAMES)}")

# --- Worsts ---
print("\nWorst Average Train Loss:")
print(f"  Train Loss: {worst_train['avg_train_loss']:.5f}")
print(f"  Features: {feature_names(worst_train['node_features'], FEATURE_NAMES)}")

print("\nWorst Average Test Loss:")
print(f"  Test Loss: {worst_test['avg_test_loss']:.5f}")
print(f"  Features: {feature_names(worst_test['node_features'], FEATURE_NAMES)}")

print("\nWorst Combined (Train + Test) Loss:")
print(f"  Total Loss: {(worst_combined['avg_train_loss'] + worst_combined['avg_test_loss']):.5f}")
print(f"  Train Loss: {worst_combined['avg_train_loss']:.5f}")
print(f"  Test Loss: {worst_combined['avg_test_loss']:.5f}")
print(f"  Features: {feature_names(worst_combined['node_features'], FEATURE_NAMES)}")

print("\nAll Combinations and Losses:")
for res in results:
    feats = feature_names(res["node_features"], FEATURE_NAMES)
    print(f"  Features: {feats} | "
          f"Train Loss: {res['avg_train_loss']:.5f} | Test Loss: {res['avg_test_loss']:.5f}")