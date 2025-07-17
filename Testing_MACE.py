from ase.io import read
from mace.calculators import mace_mp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from DFT_New import calculate_energy


def write_xyz_file(coords, filepath, filename):
    """
    Writes a list of element-coordinate strings to an .xyz file.

    Args:
        coords (list): List of strings, each formatted like 'Cu x y z'.
        filepath (str): Path to directory where file should be saved.
        filename (str): Name of the .xyz file to create (e.g., 'Cu_cluster.xyz').
    """
    num_atoms = len(coords)
    full_path = f"{filepath}/{filename}"

    with open(full_path, 'w') as f:
        f.write(f"{num_atoms}\n")
        f.write("0 1\n")  # Second line is typically a comment; can customize as needed
        for line in coords:
            f.write(f"{line}\n")

    print(f"XYZ file written to: {full_path}")


# Load MACE model
calc = mace_mp(model="medium", device='cpu')

# File location
folder_path = 'MACE Tests'
name = 'Cu'

# coords = ['Al -10.09732  -4.03893   1.02527', 'Al -12.11679  -2.01946   1.02527', ' Al -10.09732   0.00000   1.02527',
#           'Al -8.07786  -2.01946   1.02527', 'Al -10.09732  -2.01946  -0.99420', 'Al -12.11679  -4.03893  -0.99420',
#           'Al -12.11679  -0.00000  -0.99420', 'Al -8.07786  -0.00000  -0.99420', 'Al -8.07786  -4.03893  -0.99420']

coords = ['Cu -8.94358  -3.57743   0.90812', 'Cu -10.73229  -1.78872   0.90812', 'Cu -8.94358   0.00000   0.90812',
          'Cu -7.15486  -1.78872   0.90812', 'Cu -8.94358  -1.78872  -0.88060', 'Cu -10.73229  -3.57743  -0.88060',
          'Cu -10.73229  -0.00000  -0.88060', 'Cu -7.15486  -0.00000  -0.88060', 'Cu -7.15486  -3.57743  -0.88060']

current_coords = []

mace_energies = []
pyscf_energies = []

for i in range(1, len(coords)+1):

    current_coords.append(coords[i-1])

    write_xyz_file(current_coords, folder_path, f'{name}_{i}.xyz')

    atoms = read(f'{folder_path}/{name}_{i}.xyz')

    atoms.calc = calc

    energy = atoms.get_total_energy()
    print(f'MACE energy for {1} {name} atom(s): ', energy)
    mace_energies.append(energy)

    pyscf_energy = calculate_energy(current_coords)
    print(f'PySCF energy for {1} {name} atom(s): ', pyscf_energy)
    pyscf_energies.append(pyscf_energy)


# Convert to NumPy arrays
X = np.array(pyscf_energies).reshape(-1, 1)
y = np.array(mace_energies)

# Fit linear regression
reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]
intercept = reg.intercept_

# Print the regression equation
print(f"Regression equation: y = {slope:.6f} * x + {intercept:.6f}")

# Plot
plt.figure(figsize=(6, 6))
plt.scatter(pyscf_energies, mace_energies, color='blue', label='Data')
plt.plot(pyscf_energies, reg.predict(X), color='red', linestyle='--',
         label=f'Regression line:\n y = {slope:.4f}x + {intercept:.2f}')
plt.xlabel("DFT Energy (eV)")
plt.ylabel("MACE Energy (eV)")
plt.title(f"DFT vs MACE Energies ({name} Clusters)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
