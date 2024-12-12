from pyscf import gto, dft, lib
import numpy as np
import ctypes
from collections import Counter


_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)


def get_spin(element_list):

    unpaired_electrons = {
        'H': 1, 'Li': 1, 'Be': 0, 'B': 1, 'C': 0,
        'N': 3, 'O': 2, 'F': 1, 'Ne': 0, 'Na': 1,
        'Al': 1, 'La': 1, 'Ni': 2, 'Mg': 0, 'V': 3,
        'Zr': 2, 'Ca': 0, 'Ti': 2, 'Fe': 4
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


def dftd3(mf, mol):
    """
        Takes a setup molecule system and calculates total energy whilst taking into account
        attractive and repulsive forces.

        :param mf: DFT setup molecule system.
        :param mol: Molecule system.
        :return: Energy with forces taken into account.
    """

    coords = mol.atom_coords()
    itype = np.zeros(mol.natm, dtype=np.int32)
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        itype[ia] = lib.parameters.NUC[symb]

    func = 'b3-lyp'.encode()  # Encoding required for ctypes
    version = 4
    tz = 0
    edisp = np.zeros(1)
    grad = np.zeros((mol.natm, 3))

    # Call the libdftd3 wrapper
    libdftd3.wrapper(
        ctypes.c_int(mol.natm),
        coords.ctypes.data_as(ctypes.c_void_p),
        itype.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_char_p(func),
        ctypes.c_int(version),
        ctypes.c_int(tz),
        edisp.ctypes.data_as(ctypes.c_void_p),
        grad.ctypes.data_as(ctypes.c_void_p),
    )

    # Add dispersion correction to the SCF energy
    energy_with_dispersion = (mf.kernel() + edisp[0]) * 27.2114  # eV

    return energy_with_dispersion


def calculate_energy(atoms):
    """
        Takes in the XYZ format of the 3D coordinates of the material and computes
        the total energy.

        :param atoms: 3D coordinates of a material.
        :return: The total energy in eV.
    """

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    mol = gto.M(                                                                                                        # Define the molecule in PySCF
        verbose=0,
        atom=atoms,
        basis='def2-svp',
        unit='Angstrom',
        spin=get_spin(element_count)
    )

    mf = dft.RKS(mol)                                                                                                   # Setup system for DFT calculation

    corrected_energy = dftd3(mf, mol)                                                                                   # Call the dftd3 function

    return corrected_energy                                                                                             # Convert from Hartree to eV


# def calculate_energy(atoms):
#
#     symbols = [line.split()[0] for line in atoms]
#     element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]
#
#     mol = gto.M(                                                                                                        # Define the molecule in PySCF
#         verbose=0,
#         atom=atoms,
#         basis='def2-svp',
#         unit='Angstrom',
#         spin=get_spin(element_count)
#     )
#
#     mf = dft.RKS(mol)                                                                                                   # Run DFT calculation
#     mf.xc = 'b3lyp'                                                                                                     # Choose an exchange-correlation functional
#     energy = mf.kernel()
#
#     return 27.2114 * energy