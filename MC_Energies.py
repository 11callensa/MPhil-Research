import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

import Old_Energy_Model_H
import Old_Energy_Model_Combined
import Old_Energy_Model_Compound


trials = 2
all_errors = []

for trial in range(trials):
    print(f"Running trial {trial + 1}/{trials}")
    sample = True
    per_graph_error = Energy_Model_Compound.run_training(sample)  # returns dict: {system_name: mae}
    all_errors.append(per_graph_error)

# Step 1: Organize errors by system
error_dict = defaultdict(list)

for trial_errors in all_errors:
    for system_name, error in trial_errors.items():
        error_dict[system_name].append(error)

# Step 2: Compute stats per system
summary_stats = {}

for system_name, errors in error_dict.items():
    errors_np = np.array(errors)
    mean = np.mean(errors_np)
    std = np.std(errors_np, ddof=1)  # sample std dev
    ci95 = 1.96 * std / np.sqrt(len(errors_np))  # 95% confidence interval

    summary_stats[system_name] = {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "all_errors": errors_np,
    }

# Step 3: Display summary
print("\n===== Summary (Per Graph) =====")
for system_name, stats in summary_stats.items():
    print(f"{system_name:30s} | MAE: {stats['mean']:.4f} ± {stats['ci95']:.4f} (95% CI)")

# Step 4: Optional histogram (aggregate across all systems)
# Flatten the list: extract scalar float from each array
all_mae_values = [float(error) for stats in summary_stats.values() for error in stats["all_errors"]]

plt.figure(figsize=(8, 5))
plt.hist(all_mae_values, bins=20, color="skyblue", edgecolor="black")
plt.xlabel("Mean Absolute Error (Å)")
plt.ylabel("Frequency")
plt.title("Distribution of MAE across Monte Carlo Trials")
plt.grid(True)
plt.tight_layout()
plt.show()