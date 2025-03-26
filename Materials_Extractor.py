import os
import numpy as np

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.local_env import CrystalNN
from ase.io import read
from dotenv import load_dotenv
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.phase_diagram import PhaseDiagram

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def extract_hydrogen(file_path):
    """
        Extracts information about H2 and calculates bond length, necessary for molecule placement.

        :param file_path: Contains the name of the poscar file for H2.
        :return: Hydrogen bond lengths.
    """

    structure = read(file_path)
    atoms = structure.get_chemical_symbols()
    positions = structure.get_positions()

    if atoms.count('H') == 2:
        pos1 = positions[0]
        pos2 = positions[1]
        bond_length = np.linalg.norm(pos1 - pos2)                                                                       # Calculate distance between two H atoms.

        return bond_length
    else:
        print("File does not contain exactly two hydrogen atoms.")                                                      # Process only works for H2, not other forms of hydrogen.
        return None, None


def extract_compound(material_id, name):
    """
        Takes in the Material Project ID and extracts coordinates and features of the compound.

        :param: material_id: Materials Project ID.
        :param: name: Chemical name of the compound.
        :return: Compound coordinates.
    """

    with MPRester(key) as m:                                                                                            # Access API using key.

        print("Material ID: ", material_id)

        material_data = m.get_entry_by_material_id(material_id)                                                         # Extract general material data.

        try:
            energy_above_hull = m.get_entry_by_material_id(material_id, property_data=['energy_above_hull'])[0].data.get(
                'energy_above_hull')                                                                                    # Extract energy above hull if m is a list.
        except:
            detail = m.get_entry_by_material_id(material_id)                                                            # If m is not a list, build the compound phase diagram and find energy above hull.
            elements = [str(el) for el in detail.composition.elements]
            details = m.get_entries_in_chemsys(elements)
            phase_diagram = PhaseDiagram(details)
            energy_above_hull = phase_diagram.get_e_above_hull(detail)

        try:
            elasticity_doc = m.materials.elasticity.search(material_ids=[material_id])[0]                               # Extract mechanical properties of the compound.
        except (IndexError, AttributeError, KeyError):
            elasticity_doc = None

        print(elasticity_doc)

        bulk_voigt = None                                                                                               # Initialize variables with default values (None) in case input is required
        bulk_reuss = None
        shear_voigt = None
        shear_reuss = None
        poisson_ratio = None

        if elasticity_doc.bulk_modulus is not None:                                                                     # If there are mechanical properties, extract shear and bulk moduli and poisson's ratio
            bulk_voigt = elasticity_doc.bulk_modulus.voigt
            bulk_reuss = elasticity_doc.bulk_modulus.reuss
            shear_voigt = elasticity_doc.shear_modulus.voigt
            shear_reuss = elasticity_doc.shear_modulus.reuss
            poisson_ratio = elasticity_doc.homogeneous_poisson

            avg_bulk = (bulk_voigt + bulk_reuss) / 2
            avg_shear = (shear_voigt + shear_reuss) / 2

        else:                                                                                                           # If elasticity_doc is None, prompt the user for inputs.
            print("Elasticity document retrieval failed or returned None.")
            print("You can manually input the required values.")

            avg_bulk, avg_shear, poisson_ratio = manual_input()

        print("Energy Above Hull: ", energy_above_hull)
        print("Average Bulk Modulus: ", avg_bulk)
        print("Average Shear Modulus: ", avg_shear)
        print("Poisson's Ratio: ", poisson_ratio)

        adsorption_temp = input("Input the compound's adsorption temperature in K (If in TESTING mode input 0): ")      # User inputs adsorption and desorption temperature.
        desorption_temp = input("Input the compound's desorption temperature in K (If in TESTING mode input 0: ")

        print("Adsorption Temperature: ", adsorption_temp)
        print("Desorption Temperature: ", desorption_temp)

        extracted_input_features = [energy_above_hull]                                                                  # Store the energy above hull and temperatures.
        extracted_output_features = [adsorption_temp, desorption_temp]

        uncertain_features = [avg_bulk, avg_shear, poisson_ratio]                                                       # Store the mechanical features as uncertain features (These may or may not be used in the model training).

        structure = m.get_structure_by_material_id(material_id)                                                         # Extract the regular structure of the compound.
        supercell_structure = structure * (3, 3, 3)                                                                     # Form the compound supercell.

        sga = SpacegroupAnalyzer(supercell_structure)                                                                   # Extract data of the supercell.
        conventional_structure = sga.get_conventional_standard_structure()                                              # Convert that data into a unit cell.

        cif_writer = CifWriter(conventional_structure)                                                                  # Save this unit cell in a .cif file.
        cif_writer.write_file(f"CIF Files/{name}_supercell.cif")

        print(f"CIF file saved as '{name}_supercell.cif'.")

        task = True
        while task:                                                                                                     # Need to use VESTA as cannot extract the data from the variable conventional_structure.
            choice = input(
                "Pause - 1) Open saved CIF file in Vesta. \n"
                "2) Click Export Data and select VASP format - name the file in the format {chemical_name}_supercell.vasp.\n"
                "3) Click Save and select 'Cartesian Coordinates'.\n"
                "4) From the drop down menu select 'Output coordinates of all displayed atoms', click ok.\n"
                "5) Return to the python program and type 'y' to continue: \n")

            if choice.lower() == 'y':                                                                                   # Once completed, mark the task as done.
                task = False
            else:
                print("Please complete the task and type 'y' to continue.")

    with open(f"POSCAR Files/{name}_supercell.vasp", "r") as file:                                                      # Extract the ocntents of the .vasp file.
        lines = file.readlines()

    atom_types = lines[5].split()                                                                                       # Extract atom types and their counts.
    atom_counts = list(map(int, lines[6].split()))

    atoms = [atom for atom, count in zip(atom_types, atom_counts) for _ in range(count)]                                # Create the atoms list.

    coordinate_lines = lines[8:]                                                                                        # Start reading coordinates from the 9th line.
    positions = []

    for line in coordinate_lines:
        positions.append(list(map(float, line.split())))

    xyz_format = []
    unique_atoms = set()

    for symbol, pos in zip(atoms, positions):                                                                           # Store information in the desired format.
        xyz_format.append(f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}")
        unique_atoms.add(symbol)

    oxidation_states = material_data[0].data.get('oxidation_states')                                                    # Extract oxidation states for all atoms in the compound.

    if oxidation_states is None:                                                                                        # If oxidation information is not available, have the user input the states.
        oxidation_states = {}                                                                                           # Create an empty dictionary to store oxidation states.
        for symbol in unique_atoms:
            state = float(input(f"Input the oxidation state of atom {symbol} within this compound: "))
            oxidation_states[symbol] = state                                                                            # Store states in the dictionary

    print("Oxidation states: ", oxidation_states)
    print("Extracted input features: ", extracted_input_features)
    print("Extracted output features: ", extracted_output_features)

    return xyz_format, oxidation_states, extracted_input_features, extracted_output_features, uncertain_features


def manual_input():
    """
        A function in case the user must input bulk and shear modulus and poisson's ratio.

        :return: Bulk modulus, shear modulus and poisson's ratio.
    """

    while True:
        try:
            avg_bulk = float(input("Enter the Bulk Modulus in : "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            avg_shear = float(input("Enter the Shear Modulus in : "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    while True:
        try:
            poisson_ratio = float(input("Enter Poisson's Ratio: "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    return avg_bulk, avg_shear, poisson_ratio


def compute_bonds(positions):
    """
        Computes the bonds between atoms based on maximum and minimum cutoff distances.
        These distances are manually input by the user who finds them by opening the compound in VESTA.

        :param positions: Coordinates of all atoms in the compound.
        :return: The list of pairs of atom indices that are bonded.
    """

    pairs_with_cutoffs = {}                                                                                             # Set a dictionary to store atom pairs and their max. and min. cutoffs.

    task = True
    while task:                                                                                                         # Have the user run VESTA and find the bond cutoff distances.
        choice = input(
            "Pause - 1) In Vesta, click Edit - Bonds. \n"
            "2) Look at the bonds that exist in the list - add any new bond pairs and cutoff distances and"
            "satisfy yourself that those are the bonds that actually form in the crystal.\n"
            "3) Note the bond pairs (e.g., Ti-O) and associated cutoff distances.\n"
            "4) Return to the python program and type 'y' to continue: \n"
        )
        if choice.lower() == 'y':                                                                                       # If the task is complete exit the loop.
            task = False
        else:
            print("Please complete the task and type 'y' to continue.")

    while True:                                                                                                         # For every identified pair of atoms in the compound, ask for bond cutoff distances.
        pair = input("Enter a pair of elements to form bonds between (e.g., 'Ti-O') or 'done' to stop: ").strip()

        if pair.lower() == 'done':                                                                                      # Exit loop if the user types 'done'
            break

        if '-' not in pair:                                                                                             # Validate the pair format.
            print("Invalid format. Please enter a pair in the format 'Element1-Element2'.")
            continue

        try:                                                                                                            # Ask for the maximum cutoff distance.
            cutoff = float(input(f"Enter the maximum cutoff distance for the pair {pair}: "))
        except ValueError:
            print("Invalid cutoff distance. Please enter a valid number.")
            continue

        try:                                                                                                            # Ask for the minimum cutoff distance with a default value of 0.
            min_cutoff = input(f"Enter the minimum cutoff distance for the pair {pair} (default is 0): ").strip()
            if min_cutoff == '':
                min_cutoff = 0.0
            else:
                min_cutoff = float(min_cutoff)
        except ValueError:
            print("Invalid minimum cutoff distance. Setting to default value of 0.")
            min_cutoff = 0.0

        pairs_with_cutoffs[pair] = (min_cutoff, cutoff)                                                                 # Save the pair and cutoff distances to the dictionary.
        print(f"Added {pair} with minimum cutoff {min_cutoff} and maximum cutoff {cutoff}.")

    print("\nThe following pairs and their cutoff distances were added:")                                               # Print the resulting pairs and their cutoffs.
    for pair, (min_cutoff, cutoff) in pairs_with_cutoffs.items():
        print(f"{pair}: min_cutoff = {min_cutoff}, max_cutoff = {cutoff}")

    atom_symbols = []                                                                                                   # Parse positions into separate atom types and coordinates.
    atom_coordinates = []
    for position in positions:

        parts = position.split()                                                                                        # Split each line into element and coordinates.
        atom_symbols.append(parts[0])                                                                                   # Atom type (Ti, O, etc.).
        atom_coordinates.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    atom_coordinates = np.array(atom_coordinates)                                                                       # Convert the list of atom coordinates to a numpy array for vectorized operations.

    edge_indices = []                                                                                                   # List to hold the bonds (edges).

    num_atoms = len(atom_coordinates)

    for i in range(num_atoms):                                                                                          # Compute the distance between all pairs of atoms.
        for j in range(i + 1, num_atoms):                                                                               # Only check each pair once (since bonding is symmetric).
            element_pair = f"{atom_symbols[i]}-{atom_symbols[j]}"                                                       # Get the element types of the two atoms.

            if element_pair in pairs_with_cutoffs:                                                                      # Check if the current pair is in the list of valid pairs.
                min_cutoff, max_cutoff = pairs_with_cutoffs[element_pair]                                               # Get the cutoff distances for this pair.

                distance = np.linalg.norm(atom_coordinates[i] - atom_coordinates[j])                                    # Calculate the Euclidean distance between atoms i and j.

                if min_cutoff <= distance <= max_cutoff:                                                                # Check if the distance is within the specified range.
                    edge_indices.append([i, j])                                                                         # Append the bond pair (in both directions).

    return edge_indices


def reassociate_coordinates(raw_optimised_xyz, combined_xyz):
    """
        Re-associates the optimised coordinates with the corresponding atoms type.

        :param raw_optimised_xyz: Raw optimised 3D coordinates of the system.
        :param combined_xyz: Un-optimised 3D coordinates of the system with atom types.
        :return: Optimised 3D coordinates with atom types in XYZ format.
    """

    elements = [line.split()[0] for line in combined_xyz]                                                               # Extract element symbols from combined_xyz.

    if len(elements) != len(raw_optimised_xyz):                                                                         # Ensure number of elements matches number of coordinates.
        raise ValueError("Mismatch between number of atoms and coordinates")

    optimised_xyz = [
        f"{elem} {x:.10f} {y:.10f} {z:.10f}"
        for elem, (x, y, z) in zip(elements, raw_optimised_xyz)
        ]                                                                                                               # Format new coordinates with atom labels.

    return optimised_xyz