import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import NEW_Temperature_Model

trials = 50
all_errors = []  # To store adsorption MAE per trial

for trial in range(trials):
    print(f"Running trial {trial + 1}/{trials}")
    sample = True
    per_graph_error = NEW_Temperature_Model.run_training('ads', sample)  # returns dict: {system_name: (mae_ads, mae_des)}
    # Assuming your run_training now returns dict with tuples for ads and des MAEs per system

    # Separate ads and des errors per trial
    errors = {}

    for system_name, mae in per_graph_error.items():
        errors[system_name] = mae

    all_errors.append(errors)

# Step 1: Organize errors by system for adsorption and desorption separately
error_dict = defaultdict(list)

for trial_errors_ads in all_errors:
    for system_name, error in trial_errors_ads.items():
        error_dict[system_name].append(error)

# Step 2: Compute stats per system separately for adsorption and desorption
summary_stats = {}

for system_name, errors in error_dict.items():
    errors_np = np.array(errors)
    mean = np.mean(errors_np)
    std = np.std(errors_np, ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(errors_np))
    summary_stats[system_name] = {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "all_errors": errors_np,
    }



# Step 3: Display summary
print("\n===== Summary (Per Graph) - Adsorption Temperature =====")
for system_name, stats in summary_stats.items():
    print(f"{system_name:30s} | MAE: {stats['mean']:.4f} ± {stats['ci95']:.4f} (95% CI)")

# Step 4: Optional histogram for adsorption temperatures
all_mae_values = [float(error) for stats in summary_stats.values() for error in stats["all_errors"]]

plt.figure(figsize=(8, 5))
plt.hist(all_mae_values, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Mean Absolute Error (Adsorption Temperature)")
plt.ylabel("Frequency")
plt.title("Distribution of Adsorption Temperature MAE across Monte Carlo Trials")
plt.grid(True)
plt.tight_layout()
plt.show()
