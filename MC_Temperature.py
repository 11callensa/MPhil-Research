import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import Temperature_Model

trials = 2
all_errors_ads = []  # To store adsorption MAE per trial
all_errors_des = []  # To store desorption MAE per trial

for trial in range(trials):
    print(f"Running trial {trial + 1}/{trials}")
    sample = True
    per_graph_error = Temperature_Model.run_training(sample)  # returns dict: {system_name: (mae_ads, mae_des)}
    # Assuming your run_training now returns dict with tuples for ads and des MAEs per system

    # Separate ads and des errors per trial
    ads_errors = {}
    des_errors = {}

    for system_name, mae_tuple in per_graph_error.items():
        mae_ads, mae_des = mae_tuple
        ads_errors[system_name] = mae_ads
        des_errors[system_name] = mae_des

    all_errors_ads.append(ads_errors)
    all_errors_des.append(des_errors)

# Step 1: Organize errors by system for adsorption and desorption separately
error_dict_ads = defaultdict(list)
error_dict_des = defaultdict(list)

for trial_errors_ads, trial_errors_des in zip(all_errors_ads, all_errors_des):
    for system_name, error in trial_errors_ads.items():
        error_dict_ads[system_name].append(error)
    for system_name, error in trial_errors_des.items():
        error_dict_des[system_name].append(error)

# Step 2: Compute stats per system separately for adsorption and desorption
summary_stats_ads = {}
summary_stats_des = {}

for system_name, errors in error_dict_ads.items():
    errors_np = np.array(errors)
    mean = np.mean(errors_np)
    std = np.std(errors_np, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(errors_np))
    summary_stats_ads[system_name] = {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "all_errors": errors_np,
    }

for system_name, errors in error_dict_des.items():
    errors_np = np.array(errors)
    mean = np.mean(errors_np)
    std = np.std(errors_np, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(errors_np))
    summary_stats_des[system_name] = {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "all_errors": errors_np,
    }

# Step 3: Display summary
print("\n===== Summary (Per Graph) - Adsorption Temperature =====")
for system_name, stats in summary_stats_ads.items():
    print(f"{system_name:30s} | MAE: {stats['mean']:.4f} ± {stats['ci95']:.4f} (95% CI)")

print("\n===== Summary (Per Graph) - Desorption Temperature =====")
for system_name, stats in summary_stats_des.items():
    print(f"{system_name:30s} | MAE: {stats['mean']:.4f} ± {stats['ci95']:.4f} (95% CI)")

# Step 4: Optional histogram for adsorption temperatures
all_mae_values_ads = [float(error) for stats in summary_stats_ads.values() for error in stats["all_errors"]]

plt.figure(figsize=(8, 5))
plt.hist(all_mae_values_ads, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Mean Absolute Error (Adsorption Temperature)")
plt.ylabel("Frequency")
plt.title("Distribution of Adsorption Temperature MAE across Monte Carlo Trials")
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Optional histogram for desorption temperatures
all_mae_values_des = [float(error) for stats in summary_stats_des.values() for error in stats["all_errors"]]

plt.figure(figsize=(8, 5))
plt.hist(all_mae_values_des, bins=20, color="salmon", edgecolor="black")
plt.xlabel("Mean Absolute Error (Desorption Temperature)")
plt.ylabel("Frequency")
plt.title("Distribution of Desorption Temperature MAE across Monte Carlo Trials")
plt.grid(True)
plt.tight_layout()
plt.show()
