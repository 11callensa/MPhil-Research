import torch
from captum.attr import IntegratedGradients
from scipy.stats import shapiro
import shap

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DispGNNWrapper(torch.nn.Module):
    def __init__(self, model, edge_index, other_input, output_dim, mode='node'):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.other_input = other_input  # edge_attr if mode=node, or node_features if mode=edge
        self.output_dim = output_dim
        self.mode = mode  # 'node' or 'edge'

    def forward(self, features):
        """
        If mode='node': features = node_features, self.other_input = edge_attr
        If mode='edge': features = edge_attr, self.other_input = node_features
        """
        if self.mode == 'node':
            node_features = features
            edge_attr = self.other_input
        elif self.mode == 'edge':
            edge_attr = features
            node_features = self.other_input
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Use 'node' or 'edge'.")

        output = self.model(node_features, self.edge_index, edge_attr)

        # output shape: (num_atoms, 3)
        return output[:, self.output_dim]  # return only one displacement component per atom


def disp_features(test_graphs, model):
    # ========== Setup ==========
    sample = test_graphs[1]
    node_features = sample.x.clone().detach().requires_grad_(True)
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    baseline = torch.zeros_like(node_features)

    num_atoms = node_features.shape[0]
    num_features = node_features.shape[1]
    num_outputs = 3  # x, y, z displacement

    feature_names = ["x", "y", "z", "mass", "protons", "neutrons", "electrons"]

    # ========== Attribution Calculation ==========
    all_attributions = np.zeros((num_atoms, num_features, num_outputs))

    for output_dim in range(num_outputs):
        wrapped_model = DispGNNWrapper(model, edge_index, edge_attr, output_dim, mode='node')
        ig = IntegratedGradients(wrapped_model)
        attributions, _ = ig.attribute(inputs=node_features,
                                       baselines=baseline,
                                       return_convergence_delta=True)
        all_attributions[:, :, output_dim] = attributions.detach().cpu().numpy()

    # ========== Per-Atom Heatmap ==========
    mean_attr_per_atom = np.mean(np.abs(all_attributions), axis=2)
    threshold = 1e-5
    nonzero_rows = np.where(np.any(mean_attr_per_atom > threshold, axis=1))[0]

    fig, ax = plt.subplots(figsize=(10, len(nonzero_rows) * 0.3 + 2))
    sns.heatmap(mean_attr_per_atom[nonzero_rows],
                cmap='coolwarm',
                center=0,
                yticklabels=nonzero_rows,
                xticklabels=feature_names,
                ax=ax,
                cbar_kws={"label": "Mean Absolute Attribution (unitless)"})

    # Show all ticks, but only label every 10th
    ax.set_yticks(np.arange(len(nonzero_rows)))
    y_labels = [str(nonzero_rows[i]) if i % 10 == 0 else "" for i in range(len(nonzero_rows))]
    ax.set_yticklabels(y_labels)

    ax.set_title(f'Node Feature Attribution Heatmap\n(Showing {len(nonzero_rows)} atoms)')
    ax.set_xlabel('Node Feature')
    ax.set_ylabel('Atom Index')
    plt.tight_layout()
    plt.show()

    # ========== Global Feature Importance ==========
    abs_attr = np.abs(all_attributions)
    global_feature_importance = np.mean(abs_attr, axis=(0, 2))

    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=global_feature_importance)
    plt.xlabel("Node Feature")
    plt.ylabel("Mean Absolute Attribution (unitless)")
    plt.title("Global Node Feature Attribution (Averaged Over All Atoms and Axes)")
    plt.tight_layout()
    plt.show()


def ads_temperature_shap_test(X_train_norm, X_test_norm, model):
    # Use a small subset for SHAP to avoid memory issues
    X_background = X_train_norm[:10]
    X_explain = X_test_norm[:2]

    # Define a wrapper function for the model
    def model_forward(x_numpy):
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
        with torch.no_grad():
            preds = model(x_tensor).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            return preds.detach().cpu().numpy()

    # Use KernelExplainer for tabular models with small input size
    explainer = shap.KernelExplainer(model_forward, X_background.numpy())
    shap_values = explainer.shap_values(X_explain.numpy())

    # Plot feature importance using custom bar plot
    feature_names = ["Bulk Modulus", "Shear Modulus", "Poisson's Ratio", "Energy above Hull",
                     "Average Electronegativity"]

    # If shap_values is a list (e.g., for multi-output), take the first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_shap = mean_abs_shap[sorted_idx]
    sorted_names = np.array(feature_names)[sorted_idx]

    # Custom bar plot
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_shap)), sorted_shap, color='skyblue')
    plt.yticks(range(len(sorted_shap)), sorted_names)
    plt.xlabel("Mean SHAP Value (Influence on Prediction)", fontsize=12)
    plt.title("Feature Importance from SHAP", fontsize=14)
    plt.tight_layout()
    plt.show()


def des_temperature_shap_test(X_train_norm, X_test_norm, model):

    # Ensure NumPy arrays
    X_background = np.array(X_train_norm[:10])
    X_explain = np.array(X_test_norm[:2])

    # Model forward function
    def model_forward(x_numpy):
        return model.predict(x_numpy)

    explainer = shap.KernelExplainer(model_forward, X_background)
    shap_values = explainer.shap_values(X_explain)

    feature_names = [
        "Bulk Modulus",
        "Shear Modulus",
        "Poisson's Ratio",
        "Energy above Hull",
        "Average Electronegativity"
    ]

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_shap = mean_abs_shap[sorted_idx]
    sorted_names = np.array(feature_names)[sorted_idx]

    plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_shap)), sorted_shap, color='skyblue')
    plt.yticks(range(len(sorted_shap)), sorted_names)
    plt.xlabel("Mean SHAP Value (Influence on Prediction)", fontsize=12)
    plt.title("Feature Importance from SHAP", fontsize=14)
    plt.tight_layout()
    plt.show()


def displacement_errors(pred_disp, true_disp, elements, num_fixed):
    percent_errors = []
    for i, (e, pred, true) in enumerate(zip(elements, pred_disp, true_disp)):
        if e == 'H' and i >= num_fixed:
            true_mag = torch.norm(true).item()
            abs_error = torch.norm(true - pred).item()
            percent_error = abs_error / (true_mag + 1e-8) * 100
            percent_errors.append(percent_error)

    return percent_errors


def test_residual_normality(true_displacements, predicted_displacements, plot=True):
    """
    Perform a Shapiro-Wilk test for normality on residuals.

    Args:
        true_displacements (torch.Tensor or np.ndarray): shape (N, 3)
        predicted_displacements (torch.Tensor or np.ndarray): shape (N, 3)
        plot (bool): Whether to plot histogram and QQ plot.

    Returns:
        stat (float), p_value (float)
    """
    if isinstance(true_displacements, torch.Tensor):
        true_displacements = true_displacements.detach().cpu().numpy()
    if isinstance(predicted_displacements, torch.Tensor):
        predicted_displacements = predicted_displacements.detach().cpu().numpy()

    residuals = predicted_displacements - true_displacements
    residuals_flat = residuals.reshape(-1)  # Flatten x/y/z for 1D test

    # Shapiro-Wilk test
    stat, p_value = shapiro(residuals_flat)

    print(f"\nShapiro-Wilk Test:\nStatistic = {stat:.4f}, p-value = {p_value:.4e}")
    if p_value > 0.01:
        print("✅ Residuals appear to follow a normal distribution (fail to reject H0).")
    else:
        print("❌ Residuals do NOT appear normally distributed (reject H0).")

    if plot:
        import statsmodels.api as sm

        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(residuals_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals (y_true - y_pred)")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        sm.qqplot(residuals_flat, line='s')
        plt.title("QQ-Plot of Residuals")

        plt.tight_layout()
        plt.show()

    return stat, p_value
