
import numpy as np
import ctypes

from pyscf import gto, dft, lib
from collections import Counter

from Compound_Properties import get_spin


_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)


xe_funcs = ['b3-lyp', 'pbe', 'b97-d']
select_xe = 0


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

    func = xe_funcs[select_xe].encode()  # Encoding required for ctypes
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
