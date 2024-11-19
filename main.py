import pandas as pd

from Genetic_Algorithm import set_materials
from Materials_Extractor import extract_hydrogen, extract_compound
from DFT import calculate_energy
from Mol_Geometry import plot, surface_finder, reorient_coordinates
from Combine_Matrices import combine_matrices, place_hydrogen_molecules

import test_bench


hydrogen, compound_name, compound_ID = set_materials()

hydrogen_bond, hydrogen_xyz = extract_hydrogen(f"{hydrogen}.poscar")
compound_bond, compound_xyz = extract_compound(compound_ID)

dataframe = pd.DataFrame(columns=['Molecule', 'Symbols', 'Bond Lengths', 'XYZ', 'Energy'])

dataframe.loc[0] = [compound_name, None, compound_bond, compound_xyz, None]
dataframe.loc[1] = [hydrogen, None, hydrogen_bond, hydrogen_xyz, None]

for index, row in dataframe.iterrows():

    material_xyz = row['XYZ']

    energy, symbol_count = calculate_energy(material_xyz)

    dataframe.loc[index, 'Symbols'] = str(symbol_count)
    dataframe.loc[index, 'Energy'] = energy

compound_matrix = dataframe.loc[0]['XYZ']

plot(compound_matrix)

surface_points = surface_finder(compound_matrix)

plot(reorient_coordinates(compound_matrix, surface_points))

dataframe.at[0, 'XYZ'] = reorient_coordinates(compound_matrix, surface_points)
reo_coords = dataframe.loc[0]['XYZ']

loops = 0
ads_energies = []

while loops < 10:

    print("Overall Loop: ", loops)
    hydrogen_placed, mol_count = place_hydrogen_molecules(dataframe)
    print("Hydrogen successfully placed")

    matrix_combined = combine_matrices(reo_coords, hydrogen_placed)

    energy_combined, symbol_count = calculate_energy(matrix_combined)

    adsorption_energy = (energy_combined - (dataframe.loc[0, 'Energy'] + dataframe.loc[1, 'Energy']))/mol_count
    print(f"Adsorption Energy per H2 molecule: {adsorption_energy} eV")
    ads_energies.append(adsorption_energy)

    loops += 1

average_adsorption = sum(ads_energies) / len(ads_energies)

print(f"Average Adsorption Energy per H2 molecule: {average_adsorption} eV")






