import torch
from captum.attr import IntegratedGradients
from scipy.stats import shapiro


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


class EnergyGNNWrapper(torch.nn.Module):
    def __init__(self, model, edge_index, edge_attr, batch, mode='node'):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.mode = mode  # if you want to support different modes

    def forward(self, node_features):
        # Here you pass the node_features, edge_index, edge_attr, batch to your model
        # Your model returns a scalar per graph, but make sure output shape is (1,) or (batch_size,1)
        output = self.model(node_features, self.edge_index, self.edge_attr, self.batch)

        # If output is a scalar tensor (0-dim), unsqueeze to make shape (1,)
        if output.dim() == 0:
            output = output.unsqueeze(0)

        # output shape now (1,) or (batch_size, 1)
        return output


class TemperatureGNNWrapper(torch.nn.Module):
    def __init__(self, model, edge_index, edge_attr, batch, mode='node'):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.mode = mode  # to keep interface consistent

    def forward(self, node_features):
        # model returns shape (batch_size, 2)
        output = self.model(node_features, self.edge_index, self.edge_attr, self.batch)

        # If single graph, output shape might be (2,), unsqueeze to (1,2)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Output shape: (batch_size, 2)
        return output


def disp_features(test_graphs, model):
# ========== Setup ==========
    sample = test_graphs[0]
    node_features = sample.x.clone().detach().requires_grad_(True)
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    baseline = torch.zeros_like(node_features)

    num_atoms = node_features.shape[0]
    num_features = node_features.shape[1]
    num_outputs = 3  # x, y, z displacement

    feature_names = ["x", "y", "z", "mass", "protons", "neutrons", "electrons"]

    # ================ NODE FEATURE ➝ ALL NODES' DISPLACEMENT ============ #
    all_attributions = np.zeros((num_atoms, num_features, num_outputs))

    for output_dim in range(num_outputs):
        wrapped_model = DispGNNWrapper(model, edge_index, edge_attr, output_dim, mode='node')
        ig = IntegratedGradients(wrapped_model)

        attributions, delta = ig.attribute(
            inputs=node_features,
            baselines=baseline,
            return_convergence_delta=True
        )
        all_attributions[:, :, output_dim] = attributions.detach().cpu().numpy()

    # Mean attribution over displacement axes (X, Y, Z)
    mean_attr_per_atom = np.mean(np.abs(all_attributions), axis=2)
    threshold = 1e-5
    nonzero_rows = np.where(np.any(mean_attr_per_atom > threshold, axis=1))[0]

    plt.figure(figsize=(10, len(nonzero_rows) * 0.3 + 2))
    sns.heatmap(mean_attr_per_atom[nonzero_rows],
                cmap='coolwarm',
                center=0,
                yticklabels=nonzero_rows,
                xticklabels=feature_names)
    plt.title(f'Node Feature Attribution Heatmap\n(Showing {len(nonzero_rows)} atoms)')
    plt.xlabel('Node Feature')
    plt.ylabel('Atom Index')
    plt.show()

    # ================ GLOBAL NODE FEATURE IMPORTANCE ============ #
    abs_attr = np.abs(all_attributions)
    global_feature_importance = np.mean(abs_attr, axis=(0, 2))

    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=global_feature_importance)
    plt.xlabel("Node Feature")
    plt.ylabel("Mean Attribution (X + Y + Z, All Atoms)")
    plt.title("Global Node Feature Attribution across All Displacements")
    plt.show()


def energy_features(test_graphs, model, mode='node'):
    # Use a single graph for visualization
    sample = test_graphs[0]
    node_features = sample.x.clone().detach().requires_grad_(True)
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    batch = sample.batch if hasattr(sample, 'batch') else torch.zeros(len(sample.x), dtype=torch.long).to(model.device)

    feature_names = ["x", "y", "z", "mass", "protons", "neutrons", "electrons"][:node_features.shape[1]]

    baseline = torch.zeros_like(node_features)

    wrapped_model = EnergyGNNWrapper(model, edge_index, edge_attr, batch, mode=mode)

    ig = IntegratedGradients(wrapped_model)
    attributions, delta = ig.attribute(
        inputs=node_features,
        baselines=baseline,
        return_convergence_delta=True
    )

    attributions_np = attributions.detach().cpu().numpy()
    mean_attr_per_atom = np.abs(attributions_np)  # shape: (num_atoms, num_features)

    # Visualize attribution per atom
    threshold = 0.001e-5
    nonzero_rows = np.where(np.any(mean_attr_per_atom > threshold, axis=1))[0]

    plt.figure(figsize=(10, len(nonzero_rows) * 0.3 + 2))
    sns.heatmap(mean_attr_per_atom[nonzero_rows],
                cmap='coolwarm',
                center=0,
                yticklabels=nonzero_rows,
                xticklabels=feature_names)
    plt.title(f'Node Feature Attribution Heatmap for Energy Prediction\n(Showing {len(nonzero_rows)} atoms)')
    plt.xlabel('Node Feature')
    plt.ylabel('Atom Index')
    plt.show()

    # Global feature importance
    global_feature_importance = np.mean(mean_attr_per_atom, axis=0)

    plt.figure(figsize=(10, 4))
    sns.barplot(x=feature_names, y=global_feature_importance)
    plt.xlabel("Node Feature")
    plt.ylabel("Mean Attribution (All Atoms)")
    plt.title("Global Node Feature Attribution for Energy Prediction")
    plt.show()


def temperature_features(test_graphs, model, mode='node'):
    sample = test_graphs[0]  # single graph for visualization

    node_features = sample.x.clone().detach().requires_grad_(True)
    edge_index = sample.edge_index
    edge_attr = sample.edge_attr
    batch = sample.batch if hasattr(sample, 'batch') else torch.zeros(len(sample.x), dtype=torch.long).to(model.device)

    feature_names = ["x", "y", "z", "mass", "protons", "neutrons", "electrons"][:node_features.shape[1]]
    baseline = torch.zeros_like(node_features)

    wrapped_model = TemperatureGNNWrapper(model, edge_index, edge_attr, batch, mode=mode)
    ig = IntegratedGradients(wrapped_model)

    attributions_list = []
    delta_list = []

    for target_idx in range(2):  # 2 outputs
        attr, delta = ig.attribute(
            inputs=node_features,
            baselines=baseline,
            target=target_idx,
            return_convergence_delta=True
        )
        attributions_list.append(attr.detach().cpu().numpy())
        delta_list.append(delta.detach().cpu().numpy())

    attributions = np.stack(attributions_list)  # shape (2, num_atoms, num_features)
    delta = np.stack(delta_list)

    for i, output_name in enumerate(['Output 1', 'Output 2']):
        mean_attr_per_atom = np.abs(attributions[i])  # shape: (num_atoms, num_features)

        threshold = 1e-5
        nonzero_rows = np.where(np.any(mean_attr_per_atom > threshold, axis=1))[0]
        if len(nonzero_rows) == 0:
            print(f"No atoms with attribution above threshold for {output_name}, showing all.")
            nonzero_rows = np.arange(mean_attr_per_atom.shape[0])

        plt.figure(figsize=(10, len(nonzero_rows)*0.3 + 2))
        sns.heatmap(mean_attr_per_atom[nonzero_rows],
                    cmap='coolwarm',
                    center=0,
                    yticklabels=nonzero_rows,
                    xticklabels=feature_names)
        plt.title(f'Node Feature Attribution Heatmap for {output_name}')
        plt.xlabel('Node Feature')
        plt.ylabel('Atom Index')
        plt.show()

        global_feature_importance = np.mean(mean_attr_per_atom, axis=0)

        plt.figure(figsize=(10, 4))
        sns.barplot(x=feature_names, y=global_feature_importance)
        plt.xlabel("Node Feature")
        plt.ylabel("Mean Attribution (All Atoms)")
        plt.title(f"Global Node Feature Attribution for {output_name}")
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

        plt.subplot(1, 2, 1)
        plt.hist(residuals_flat, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title("Histogram of Residuals")

        plt.subplot(1, 2, 2)
        sm.qqplot(residuals_flat, line='s')
        plt.title("QQ-Plot of Residuals")

        plt.tight_layout()
        plt.show()

    return stat, p_value
