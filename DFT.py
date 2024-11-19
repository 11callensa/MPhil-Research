from pyscf import gto, dft
from collections import Counter


def get_spin(element_list):

    unpaired_electrons = {
        'H': 1, 'Li': 1, 'Be': 0, 'B': 1, 'C': 0,
        'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
        'La': 1, 'Ni': 2, 'Mg': 0
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


def calculate_energy(atoms):

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    mol = gto.M(                                                                                                        # Define the molecule in PySCF
        verbose=0,
        atom=atoms,
        basis='def2-svp',
        unit='Angstrom',
        spin=get_spin(element_count)
    )

    mf = dft.RKS(mol)                                                                                                   # Run DFT calculation
    mf.xc = 'b3lyp'                                                                                                     # Choose an exchange-correlation functional
    energy = mf.kernel()

    return 27.2114 * energy, element_count                                                                              # Convert from Hartree to eV
