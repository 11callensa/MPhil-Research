import ast
import os
import numpy as np
import pandas as pd
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_optimizer import AdaBelief

from sklearn.model_selection import train_test_split

from Stats_Engineering import temperature_features, test_residual_normality

import matplotlib.pyplot as plt
from tkinter import filedialog
from collections import defaultdict

device = torch.device("cuda")

print("Device:", device)


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_size, hidden_size2, output_dim):
        super(FCNN, self).__init__()

        self.input = nn.Linear(input_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size2)
        self.fc2 = nn.Linear(hidden_size2, output_dim)

        self.relu = nn.SiLU()

    def forward(self, x):

        x = self.input(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

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
    system_features = parse_entry('Temperature Input Features')

    uncertain_features = parse_entry('Uncertain Features')

    return {
        "system_names": system_names,
        "system_features": system_features,
        "uncertain_features": uncertain_features,
    }


def run_training(mode='ads', sample=False):

    file_path = "temperature_training.csv"
    data = load_training_data(file_path)

    raw_inputs = [
        data['uncertain_features'][i][:3] + [data['system_features'][i][4]]
        for i in range(len(data['system_features']))
    ]

    if mode == 'ads':
        raw_outputs = [[float(pair[0])] for pair in data['temp_outputs']]
    elif mode == 'des':
        raw_outputs = [[float(pair[1])] for pair in data['temp_outputs']]
    else:
        raise ValueError("mode must be either 'ads' or 'des'")

    system_names = data['system_names']

    if sample:
        test_materials = ["Rh", "Cu", "LaNiO3"]  # Fixed test set

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
        material_choice = input('Do you want to select which materials to test on? y/n: ')
        if material_choice == 'y':
            test_materials = []
            print("Enter material names one at a time. Type 'done' when finished:")

            while True:
                mat = input("Material name: ")
                if mat.lower() == 'done':
                    break
                elif mat.strip() == '':
                    print("Empty input. Try again.")
                    continue
                else:
                    test_materials.append(mat.strip())

            print(f"Selected test materials: {test_materials}")

            X_train, y_train, names_train = [], [], []
            X_test, y_test, names_test = [], [], []

            for i in range(len(system_names)):
                name = system_names[i]
                input_row = raw_inputs[i]
                output_row = raw_outputs[i]

                if name in test_materials:
                    X_test.append(input_row)
                    y_test.append(output_row)
                    names_test.append(name)
                else:
                    X_train.append(input_row)
                    y_train.append(output_row)
                    names_train.append(name)
        else:
            X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
                raw_inputs, raw_outputs, data['system_names'], test_size=0.05)

    input_normalizer = PreProcess()
    input_normalizer.fit(X_train, num_feats=4)

    X_train_norm = input_normalizer.transform(X_train)
    X_test_norm = input_normalizer.transform(X_test)

    output_normalizer = PreProcess()
    output_normalizer.fit(y_train, num_feats=1)
    y_train_norm = output_normalizer.transform(y_train)
    y_test_norm = output_normalizer.transform(y_test)

    print('Number of training graphs: ', len(y_train_norm))
    print('Number of test graphs: ', len(y_test_norm))

    X_train_norm = torch.FloatTensor(X_train_norm)
    X_test_norm = torch.FloatTensor(X_test_norm)

    y_train_norm = torch.FloatTensor(y_train_norm)
    y_test_norm = torch.FloatTensor(y_test_norm)

    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    test_dataset = TensorDataset(X_test_norm, y_test_norm)

    input_size = 4
    hidden_size = 128
    hidden_size2 = 2048
    output_size = 1

    batch_size = 29

    model = FCNN(input_size, hidden_size, hidden_size2, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    epochs = 100
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

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

        loss_train_list.append(train_loss)
        loss_test_list.append(test_loss)

    all_preds_tensor = torch.cat(all_preds).unsqueeze(1).numpy()
    preds_denorm = output_normalizer.inverse_process(all_preds_tensor)
    true_vals_denorm = output_normalizer.inverse_process(y_test_norm.numpy())

    print(f"\n=== Final {mode.capitalize()}orption Temp Predictions vs Ground Truth ===")
    for name, pred, true in zip(names_test, preds_denorm, true_vals_denorm):
        print(f"{name:<20} | Predicted: {pred[0]:.1f} | True: {true[0]:.1f}")

    # Plot
    plt.figure()
    plt.plot(loss_train_list, label="Train Loss")
    plt.plot(loss_test_list, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f'Train/Test Loss ({mode.capitalize()}orption Temp Only)')
    plt.show()

    if not sample:

        save_option = input('Do you want to save this model?: ')

        if save_option == 'y':

            hyperparameters = {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "hidden_size2": hidden_size2,
                "output_size": output_size
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
                    'input_normaliser': input_normalizer,
                    'output_normaliser': output_normalizer,
                },
                "train_system_names": names_train
            }

            torch.save(save_data, model_save_path)
            print(f"Model, hyperparameters, normalisers, and system names saved at: {model_save_path}")

        else:
            pass
    else:
        pass

    test_system_names = [name for name in names_test]  # or however you get the system names

    per_graph_mae = {}

    for name, true_val, pred_val in zip(test_system_names, true_vals_denorm, preds_denorm):

        mae = abs(true_val[0] - pred_val[0])
        print(f'\n MAE for {name}: {mae}')

        per_graph_mae[name] = (mae)

    if sample:
        return per_graph_mae
    else:
        return None


def run_testing(name):

    test_file_path = f"Temperature Testing Data/{name}_temperature_testing.csv"
    test_data = load_testing_data(test_file_path)

    raw_inputs = [
        test_data['uncertain_features'][i][:3] + [test_data['system_features'][i][4]]
        for i in range(len(test_data['system_features']))
    ]

    print(raw_inputs)

    model_dir = "Temperature GNN Models"
    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("PyTorch Models", "*.pt")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = model_data = torch.load(model_file_path, weights_only=False)

    input_normaliser = model_data["normalisers"]['input_normaliser']
    output_normaliser = model_data["normalisers"]['output_normaliser']

    X_norm = input_normaliser.transform(raw_inputs)
    X_norm = torch.tensor(X_norm, dtype=torch.float32, device=device)

    hyperparameters = model_data[
        "hyperparameters"]  # Retrieve the hyperparameters and use them to initialize the model.

    model = FCNN(
        input_dim=hyperparameters["input_size"],
        hidden_size=hyperparameters["hidden_size"],
        hidden_size2=hyperparameters["hidden_size2"],
        output_dim=hyperparameters["output_size"]
    )

    model.to(device)

    model.load_state_dict(model_data["model_state_dict"])  # Load the state_dict (weights).
    model.eval()  # Set the model to evaluation mode.

    with torch.no_grad():
        pred = model(X_norm)

    pred_denorm = output_normaliser.inverse_process(pred)

    print(pred_denorm)

    return pred_denorm
