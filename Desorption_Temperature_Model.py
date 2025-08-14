import ast
import numpy as np
import pandas as pd
import random
import os
import joblib
import torch

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Stats_Engineering import des_temperature_shap_test

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


def run_training(sample=False):

    file_path = "temperature_training.csv"
    data = load_training_data(file_path)

    raw_inputs = [
        [data['uncertain_features'][i][j] for j in [0, 1, 2]] + data['system_features'][i][4:6]
        for i in range(len(data['system_features']))
    ]                                                                                                               # DESORPTION - Shear, Poisson, Energy

    raw_outputs = [[float(pair[1])] for pair in data['temp_outputs']]

    system_names = data['system_names']

    if sample:
        test_materials = ["Cu", "TiO2-A"]  # Fixed test set

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

        test_materials = ['WC', 'CoO']
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

    input_normalizer = Temp_PreProcess()
    input_normalizer.fit(X_train, num_feats=5)

    X_train_norm = input_normalizer.transform(X_train)
    X_test_norm = input_normalizer.transform(X_test)

    output_normalizer = Temp_PreProcess()
    output_normalizer.fit(y_train, num_feats=1)

    y_train_norm = output_normalizer.transform(y_train)
    y_test_norm = output_normalizer.transform(y_test)

    print('Number of training graphs: ', len(y_train_norm))
    print('Number of test graphs: ', len(y_test_norm))

    X_train_norm = torch.FloatTensor(X_train_norm)
    X_test_norm = torch.FloatTensor(X_test_norm)

    y_train_norm = torch.FloatTensor(y_train_norm)
    y_test_norm = torch.FloatTensor(y_test_norm)

    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=42)

    # Train on normalized data
    model.fit(X_train_norm, y_train_norm)

    # Predict on normalized test set
    test_preds_norm = model.predict(X_test_norm)
    test_preds_norm_reshaped = [[float(p)] for p in test_preds_norm]
    test_preds_denorm = output_normalizer.inverse_process(test_preds_norm_reshaped)
    test_preds_denorm_np = np.array(test_preds_denorm).flatten()

    # Predict on normalized train set
    train_preds_norm = model.predict(X_train_norm)
    train_preds_norm_reshaped = [[float(p)] for p in train_preds_norm]
    train_preds_denorm = output_normalizer.inverse_process(train_preds_norm_reshaped)
    train_preds_denorm_np = np.array(train_preds_denorm).flatten()

    # Denormalise targets too
    y_train_denorm = output_normalizer.inverse_process([[float(p)] for p in y_train_norm.numpy().flatten()])
    y_test_denorm = output_normalizer.inverse_process([[float(p)] for p in y_test_norm.numpy().flatten()])
    y_train_denorm_np = np.array(y_train_denorm).flatten()
    y_test_denorm_np = np.array(y_test_denorm).flatten()

    # Metrics
    train_mae = mean_absolute_error(y_train_denorm_np, train_preds_denorm_np)
    train_rmse = mean_squared_error(y_train_denorm_np, train_preds_denorm_np, squared=False)
    test_mae = mean_absolute_error(y_test_denorm_np, test_preds_denorm_np)
    test_rmse = mean_squared_error(y_test_denorm_np, test_preds_denorm_np, squared=False)

    # Output predictions
    print("\n--- Test Predictions ---")
    for name, pred in zip(names_test, test_preds_denorm):
        print(f"{name}: {pred[0]:.4f}")

    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    stats_choice = input('Do you want to perform feature engineering and statistical tests?: ')

    if stats_choice == 'y':
        # =================== FEATURE IMPORTANCE & ENGINEERING ================== #

        des_temperature_shap_test(X_train_norm, X_test_norm, model)

    else:
        pass

    if not sample:

        save_option = input("Do you want to save this model? (y/n): ")

        if save_option.lower() == 'y':

            model_name = input("Input the model name: ").strip()
            if not model_name:
                print("No model name entered. Exiting...")
                exit()

            model_dir = "Temperature Models"
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, f"xgboost_temperature_model_{model_name}.joblib")

            # Save model, normalisers, and any additional info together
            save_data = {
                "model": model,  # Trained XGBRegressor
                "normalisers": {
                    "input_normaliser": input_normalizer,
                    "output_normaliser": output_normalizer,
                },
                "train_system_names": names_train,
            }

            joblib.dump(save_data, model_save_path)
            print(f"XGBoost model and associated data saved at: {model_save_path}")

        else:
            pass
    else:
        pass

    return None


def run_testing(name):

    test_file_path = f"Temperature Testing Data/{name}_temperature_testing.csv"
    test_data = load_testing_data(test_file_path)

    raw_inputs = [[test_data['uncertain_features'][i][j] for j in [0, 1, 2]] + test_data['system_features'][i][4:6]
        for i in range(len(test_data['system_features']))
    ]

    model_dir = "Temperature Models"

    model_file_path = filedialog.askopenfilename(title="Select Model to Test", initialdir=model_dir,
                                                 filetypes=[("JobLib Models", "*.joblib")])

    if not model_file_path:
        print("No model selected. Exiting...")
        return

    print(f"Selected model: {model_file_path}")

    model_data = joblib.load(model_file_path)

    input_normaliser = model_data["normalisers"]['input_normaliser']
    output_normaliser = model_data["normalisers"]['output_normaliser']

    X_norm = input_normaliser.transform(raw_inputs)
    X_norm = torch.tensor(X_norm, dtype=torch.float32, device=device)

    model = model_data['model']

    pred = model.predict(X_norm.cpu().numpy())
    pred_reshaped = [[float(p)] for p in pred]

    pred_denorm = output_normaliser.inverse_process(pred_reshaped)

    print(pred_denorm)

    return pred_denorm[0][0].item()
