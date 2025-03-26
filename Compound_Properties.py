import numpy as np

from mendeleev import element


def get_spin(element_list):
    """
        Calculates the net spin of the system based on unpaired electrons.

        :param element_list: Each type of element and how many of that element is present in the system.
        :return: Net spin.
    """

    unpaired_electrons = {
        'H': 1,
        'Li': 1,
        'Be': 0,
        'B': 1,
        'C': 0,
        'N': 3,
        'O': 2,
        'F': 1,
        'Ne': 0,
        'Na': 1,
        'Al': 1,
        'La': 1,
        'Ni': 2,
        'Mg': 0,
        'V': 3,
        'Zr': 2,
        'Ca': 0,
        'Ti': 2,
        'Fe': 4,
        'Pt': 2,  # Platinum
        'Pd': 0,  # Palladium
        'Cr': 6,  # Chromium
        'Ga': 1,  # Gallium
        'W': 4,  # Tungsten
        'Co': 3,  # Cobalt
        'Zn': 0,  # Zinc
        'Y': 1,  # Yttrium
        'Cu': 1,  # Copper
        'Ag': 1,  # Silver
        'Au': 1,  # Gold
        'Si': 2,  # Silicon
        'Ru': 4,  # Ruthenium
        'Ir': 3,  # Iridium
        'Rh': 1,  # Rhodium
        'Ta': 3,  # Tantalum
        'Nb': 3,  # Niobium
        'At': 1,  # Astatine
        'Re': 5  # Rhenium
    }

    total_electrons = 0

    for element, count in element_list:
        count = int(count) if count else 1                                                                              # Default to 1 if no number is given.
        if element in unpaired_electrons:
            total_electrons += unpaired_electrons[element] * count                                                      # Accumulate unpaired electrons.
        else:
            print(f"Warning: Element '{element}' not defined in unpaired electrons.")

    if total_electrons % 2 == 0:                                                                                        # If the electrons pair up, then net spin is 0.
        spin = 0
    else:                                                                                                               # Else if they don't then net spin is 1.
        spin = 1

    return spin


def node_edge_features(centered_xyz, edge_indices, oxidation_states, num_fixed, flag):
    """
        Extracts features of all atoms (nodes) and bonds (edges) in the system.

        :param centered_xyz: Centered 3D coordinates.
        :param edge_indices: Bonded pairs in the system.
        :param oxidation_states: Oxidation states of atoms in the compound.
        :param num_fixed: Number of atoms in the compound.
        :param flag: Sets the mode - Crystal alone, combined crystal + hydrogen or hydrogen alone.
        :return: All node and edge features of the system.
    """

    node_features = []
    coordinates = []

    fixed_atoms = set(centered_xyz[:num_fixed])                                                                         # Define fixed atoms from the compound.

    bond_strengths = {}                                                                                                 # Bond strength dictionary.

    for index, current_node in enumerate(centered_xyz):
        node_list = []
        parts = current_node.split()

        symbol = parts[0]                                                                                               # Extract element.
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        z_coord = float(parts[3])

        coordinates.append([x_coord, y_coord, z_coord])                                                                 # Store coordinates as node features.

        mass_number, protons, neutrons, electrons = atom_config(current_node)                                           # Using the atoms type, find the mass, proton count, neutron count and electron count of each atom.

        if flag == 0:                                                                                                   # For a combined compound-hydrogen system, any hydrogen atom not in the compound has 1 electron.
            if index >= num_fixed:
                electrons = 1
            else:                                                                                                       # If we are looping through the atoms in the compound, edit the elctron count using the oxidation states.
                if symbol in oxidation_states:
                    electrons -= int(oxidation_states[symbol])  # Adjust electron count for oxidation state

        if flag == 1:                                                                                                   # For a hydrogen alone system, each hydrogen atom has 1 electron.
            electrons = 1

        node_list.extend([x_coord, y_coord, z_coord])                                                                   # Collect all features of the node and add to a list.
        node_list.extend([mass_number, protons, neutrons, electrons])

        node_features.append(node_list)                                                                                 # Add the list of features to the overall system node features list.

    edge_features = []
    for edge in edge_indices:                                                                                           # Loop through each bonded pair.

        atom1_index, atom2_index = edge
        atom1 = centered_xyz[atom1_index].split()[0]                                                                    # Extract 3D coordinates of each bonded atom.
        atom2 = centered_xyz[atom2_index].split()[0]

        if flag == 0:                                                                                                   # If this is a compounds only or compound-hydrogen system.

            within_compound = centered_xyz[atom1_index] in fixed_atoms and centered_xyz[atom2_index] in fixed_atoms     # Determine if bond is within or outside the compound
            bond_key = (atom1, atom2, "within") if within_compound else (atom1, atom2, "outside")

            if bond_key not in bond_strengths and (atom2, atom1, bond_key[2]) not in bond_strengths:                    # Ask for user input only if this bond type hasn't been entered before.
                while True:
                    try:
                        bond_context = "WITHIN THE COMPOUND" if within_compound else "OUTSIDE THE COMPOUND"
                        bond_strength = float(input(
                            f"Enter bond strength for {atom1}-{atom2} {bond_context} "
                            f"(Covalent=1, Ionic=0.75, Metallic=0.5, Alloy=0.25, None=0): "
                        ))
                        if bond_strength not in [0, 0.25, 0.5, 0.75, 1]:
                            raise ValueError("Invalid bond strength. Please enter one of: 1, 0.75, 0.5, 0.25, or 0.")

                        bond_strengths[bond_key] = bond_strength
                        break
                    except ValueError as e:
                        print(e)

            bond_strength = bond_strengths.get(bond_key, bond_strengths.get((atom2, atom1, bond_key[2]), 0))            # Retrieve bond strength

        else:                                                                                                           # For a hydrogen alone system.
            if (atom1, atom2) not in bond_strengths and (atom2, atom1) not in bond_strengths:                           # Check if bond strength for this pair has already been recorded, if not, prompt for input.
                print(f"For the bond between {atom1} and {atom2} -")
                while True:                                                                                             # Allow user to manually assign the bond strength based on the atom pair.
                    try:
                        bond_strength = float(input(
                            f"Enter bond strength for {atom1}-{atom2} (Covalent=1, Ionic=0.75, Metallic=0.5, Alloy=0.25, None=0): "))
                        if bond_strength not in [0, 0.25, 0.5, 0.75, 1]:
                            raise ValueError(
                                "Invalid bond strength value. Please enter one of the following: 1, 0.66, 0.33, or 0.")
                        bond_strengths[(atom1, atom2)] = bond_strength
                        break
                    except ValueError as e:
                        print(e)

        distance = edge_lengths(edge, coordinates)

        edge_features.append([distance, bond_strength])                                                                 # Append edge features.

    return node_features, edge_features


def atom_config(node_coords):
    """
        Finds the mass, proton no. neutron no. and electron no. of an atom.

        :param node_coords: Information on the atoms type and its position in 3D space.
        :return: Mass, proton no. neutron no. and electron no. of an atom.
    """

    parts = node_coords.split()
    symbol = parts[0]                                                                                                   # Extract element symbol (e.g., 'Li').

    el = element(symbol)                                                                                                # Get the element's data from the periodic table.

    protons = el.atomic_number                                                                                          # Extract proton and neutron counts.
    neutrons = round(el.atomic_weight) - el.atomic_number                                                               # Using most common isotope approximation.

    electrons = protons                                                                                                 # Assume a neutral atom, so electrons = protons.

    mass_number = protons + neutrons

    return [mass_number, protons, neutrons, electrons]


def edge_lengths(edge, coordinates):
    """
        Calculates the distance in 3D space between two bonded atoms.

        :param edge: Bonded pair of atoms.
        :param coordinates: Coordinates of the bonded atoms.
        :return: Euclidean distance.
    """

    node1, node2 = edge
    coord1 = coordinates[node1]
    coord2 = coordinates[node2]

    distance = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5      # Calculate Euclidean distance

    return distance


def calculate_bond_angle(coord_a, coord_center, coord_b):

    vec1 = np.array(coord_a) - np.array(coord_center)
    vec2 = np.array(coord_b) - np.array(coord_center)
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Ensure cos_theta is in valid range


def mass_and_charge(compound_reo_xyz, oxidation_states, num_atoms, flag):
    """
        Calcualtes the total mass and charge of the system.

        :param compound_reo_xyz: 3D coordinates of the system.
        :param oxidation_states: Oxidation states of atoms in the compound.
        :param num_atoms: NUmber of atoms in the compound.
        :param flag: Sets the mode - compound alone, compound-hydrogen and hydrogen aloen.
        :return: Total mass and total mass of the charge.
    """

    total_mass = 0
    total_charge = 0

    for index, atom_info in enumerate(compound_reo_xyz):
        element_symbol = atom_info.split()[0]                                                                           # Extract element symbol (first part of the string).

        el = element(element_symbol)                                                                                    # Use mendeleev to fetch atomic properties.

        total_mass += el.atomic_weight                                                                                  # Accumulate mass.

        if flag == 0:                                                                                                   # For compound alone and compound-hydrogen systems.
            if index >= num_atoms:                                                                                      # If the current atom is not in the compound, add 1 to the total charge.
                total_charge += 1
                print("Calculating charge on placed H atoms")

            else:                                                                                                       # If the current atom is in the compound.
                oxidation_state = oxidation_states.get(element_symbol, 0)
                total_charge += oxidation_state                                                                         # Calculate charge using the oxidation state.

        if flag == 1:                                                                                                   # For hydrogen alone system.
            total_charge += 1                                                                                           # Add one to the total charge for every atom.

    return total_mass, total_charge

