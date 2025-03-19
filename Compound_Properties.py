import numpy as np

from mendeleev import element


def get_spin(element_list):
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
        count = int(count) if count else 1  # Default to 1 if no number is given
        if element in unpaired_electrons:
            total_electrons += unpaired_electrons[element] * count  # Accumulate unpaired electrons
        else:
            print(f"Warning: Element '{element}' not defined in unpaired electrons.")

    if total_electrons % 2 == 0:
        total_unpaired = 0
    else:
        total_unpaired = 1

    return total_unpaired


def node_edge_features(centered_xyz, edge_indices, oxidation_states, num_fixed, flag):
    node_features = []
    coordinates = []

    # Define fixed atoms
    fixed_atoms = set(centered_xyz[:num_fixed])

    # Bond strength dictionary
    bond_strengths = {}

    # Extract node features
    for index, current_node in enumerate(centered_xyz):
        node_list = []
        parts = current_node.split()

        # Extract element and coordinates
        symbol = parts[0]
        x_coord = float(parts[1])
        y_coord = float(parts[2])
        z_coord = float(parts[3])

        coordinates.append([x_coord, y_coord, z_coord])  # Store coordinates

        # Get proton, neutron, and electron counts
        mass_number, protons, neutrons, electrons = atom_config(current_node)

        if flag == 0:
            if index >= num_fixed:
                electrons = 1
            else:
                if symbol in oxidation_states:
                    electrons -= int(oxidation_states[symbol])  # Adjust electron count for oxidation state

        if flag == 1:
            electrons = 1

        # Add basic features
        node_list.extend([x_coord, y_coord, z_coord])
        node_list.extend([mass_number, protons, neutrons, electrons])

        node_features.append(node_list)

    edge_features = []
    for edge in edge_indices:

        atom1_index, atom2_index = edge
        atom1 = centered_xyz[atom1_index].split()[0]
        atom2 = centered_xyz[atom2_index].split()[0]

        if flag == 0:

            # Determine if bond is within or outside the compound
            within_compound = centered_xyz[atom1_index] in fixed_atoms and centered_xyz[atom2_index] in fixed_atoms
            bond_key = (atom1, atom2, "within") if within_compound else (atom1, atom2, "outside")

            # Ask for user input only if this bond type hasn't been entered before
            if bond_key not in bond_strengths and (atom2, atom1, bond_key[2]) not in bond_strengths:
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

            # Retrieve bond strength
            bond_strength = bond_strengths.get(bond_key, bond_strengths.get((atom2, atom1, bond_key[2]), 0))

        else:

            # Check if bond strength for this pair has already been recorded, if not, prompt for input
            if (atom1, atom2) not in bond_strengths and (atom2, atom1) not in bond_strengths:
                print(f"For the bond between {atom1} and {atom2} -")
                # Allow user to manually assign the bond strength based on the atom pair
                while True:
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

        # Append edge features
        edge_features.append([distance, bond_strength])

    return node_features, edge_features


def atom_config(node_coords):
    """
        Extracts proton, neutron, and electron counts based on the element in the input string.

        Args:
            input_str (str): A string in the format 'Element x y z', where
                             Element is the chemical symbol (e.g., 'Li'),
                             and x, y, z are coordinates (ignored in calculations).

        Returns:
            dict: A dictionary containing proton, neutron, and electron counts.
        """
    # Parse the input string
    parts = node_coords.split()
    symbol = parts[0]  # Extract element symbol (e.g., 'Li')

    # Get the element's data from the periodic table
    el = element(symbol)

    # Extract proton and neutron counts
    protons = el.atomic_number
    neutrons = round(el.atomic_weight) - el.atomic_number  # Using most common isotope approximation

    # Assume a neutral atom, so electrons = protons
    electrons = protons

    mass_number = protons + neutrons

    return [mass_number, protons, neutrons, electrons]


def edge_lengths(edge, coordinates):

    node1, node2 = edge
    coord1 = coordinates[node1]
    coord2 = coordinates[node2]

    # Calculate Euclidean distance
    distance = ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2 + (coord1[2] - coord2[2]) ** 2) ** 0.5

    return distance


def calculate_bond_angle(coord_a, coord_center, coord_b):

    vec1 = np.array(coord_a) - np.array(coord_center)
    vec2 = np.array(coord_b) - np.array(coord_center)
    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))  # Ensure cos_theta is in valid range


def mass_and_charge(compound_reo_xyz, oxidation_states, num_atoms, flag):
    """
    Calculate the total mass and charge of a compound considering oxidation states.

    Parameters:
        compound_reo_xyz (list of str): A list where each entry contains atomic information in the format:
                                        'Element x y z'.
        oxidation_states (dict): A dictionary where keys are element symbols and values are their oxidation states.

    Returns:
        tuple: (total_mass, total_charge), where:
               - total_mass (float) is the sum of the atomic masses.
               - total_charge (float) is the sum of the charges based on oxidation states.
    """
    total_mass = 0
    total_charge = 0

    for index, atom_info in enumerate(compound_reo_xyz):
        # Extract element symbol (first part of the string)
        element_symbol = atom_info.split()[0]

        # Use mendeleev to fetch atomic properties
        el = element(element_symbol)

        # Accumulate mass
        total_mass += el.atomic_weight

        if flag == 0:
            if index >= num_atoms:
                total_charge += 1
                print("Calculating charge on placed H atoms")

            else:
                # Calculate charge using the oxidation state if provided
                oxidation_state = oxidation_states.get(element_symbol, 0)
                total_charge += oxidation_state

        if flag == 1:
            total_charge += 1

    return total_mass, total_charge

