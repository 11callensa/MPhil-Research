import csv

from Genetic_Algorithm import set_materials
from Training_Creator import training_data_creator


hydrogen, compound_ID_set = set_materials()

ads_energies = []

for compound_ID in compound_ID_set:

    adsorption_energy = training_data_creator(hydrogen, compound_ID)

    ads_energies.append(adsorption_energy)


with open('data.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    for value in ads_energies:
        writer.writerow([value])

