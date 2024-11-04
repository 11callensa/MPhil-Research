import pandas as pd

from Genetic_Algorithm import set_materials
from Materials_Extractor import extract_material_info, extract_hydrogen
from DFT import create_molecule, calculate_energy


hydrogen, compound_name, compound_ID = set_materials()

hydrogen_smiles, hydrogen_bond = extract_hydrogen(f"{hydrogen}.poscar")
compound_smiles, compound_bond = extract_material_info(compound_ID)

create_molecule(smiles=hydrogen_smiles, bond_lengths=hydrogen_bond, add_explicit_h=False, filename=f"{hydrogen}.xyz")
create_molecule(smiles=compound_smiles, bond_lengths=compound_bond, add_explicit_h=True, filename=f"{compound_name}.xyz")

energy_dataframe = pd.DataFrame(columns=['Molecule', 'Energy'])

for material in [hydrogen, compound_name]:

    energy = calculate_energy(f"{material}.xyz")
    new_row = pd.DataFrame({'Molecule': [material], 'Energy': [energy]})
    energy_dataframe = pd.concat([energy_dataframe, new_row], ignore_index=True)

    print(f"{material} energy: ", energy)

input("Pause: ")

energy_combined = calculate_energy([
                        'Li 0 0 0',
                        f'H 0 0 {bond}',
                        f'H 0 0 {z}',
                        f'H 0 0 {z + 0.74}'
                        ])





