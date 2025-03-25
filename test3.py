import os

os.environ["OMP_NUM_THREADS"] = "9"
os.environ["VECLIB_MAXIMUM_THREADS"] = "9"
os.environ["NUMEXPR_MAX_THREADS"] = "9"

import numpy as np
import ctypes
import time

from collections import Counter
from pyscf import lib
from pyscf import gto, dft
# from pyscf import df
# import pyscf.df

# from mpi4pyscf import dft

lib.num_threads = 9

_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)

xe_funcs = ['b3-lyp', 'pbe', 'b97-d', 'SVWN', 'b-lyp']
select_xe = 0


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

    func = xe_funcs[select_xe].encode()

    version = 4
    tz = 0
    edisp = np.zeros(1)
    grad = np.zeros((mol.natm, 3))

    print("Before wrapper")

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

    forces = -grad * 27.2114  # Convert to eV/Angstrom

    return energy_with_dispersion, forces


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
        verbose=5,
        atom=atoms,
        basis='6-31G',
        unit='Angstrom',
        spin=get_spin(element_count)
    )

    mf = dft.RKS(mol).density_fit().to_gpu()                                                                            # Setup system for DFT calculation

    mf.grids.level = 2  # Default is 3; increase for more accuracy, decrease for speed
    mf.direct_scf = True

    mf.diis = True  # Use DIIS for SCF convergence acceleration

    mf.set(default='dftd3')  # Use the 'dftd3' engine, or another depending on your method

    mf = mf.to_cpu()

    start_time3 = time.time()
    corrected_energy, forces = dftd3(mf, mol)                                                                           # Call the dftd3 function
    end_time3 = time.time()
    print("Time for dftd3: ", end_time3 - start_time3)

    return corrected_energy


centered_xyz_Li4H4_alone = ['Li -1.0043096959 1.0043096959 1.0043096959', 'Li -1.0043096959 -1.0043096959 -1.0043096959', 'Li 1.0043096959 -1.0043096959 1.0043096959', 'Li 1.0043096959 1.0043096959 -1.0043096959', 'H -1.0043096959 -1.0043096959 1.0043096959', 'H -1.0043096959 1.0043096959 -1.0043096959', 'H 1.0043096959 1.0043096959 1.0043096959', 'H 1.0043096959 -1.0043096959 -1.0043096959']

# centered_xyz_TiO2_alone = ['Ti 0.0000000000 0.0000000000 0.0000000002', 'Ti 0.0000000000 -1.8912698030 2.4037554262', 'Ti 0.0000000000 1.8912698030 2.4037554262', 'Ti -1.8912698030 -1.8912698030 -4.8075108528', 'Ti -1.8912698030 -1.8912698030 4.8075108532', 'Ti -1.8912698030 1.8912698030 -4.8075108528', 'Ti -1.8912698030 1.8912698030 4.8075108532', 'Ti 1.8912698030 -1.8912698030 -4.8075108528', 'Ti 1.8912698030 -1.8912698030 4.8075108532', 'Ti 1.8912698030 1.8912698030 -4.8075108528', 'Ti 1.8912698030 1.8912698030 4.8075108532', 'Ti -1.8912698030 0.0000000000 -2.4037554268', 'Ti 1.8912698030 0.0000000000 -2.4037554268', 'O -1.8912698030 0.0000000000 -0.4119832958', 'O 1.8912698030 0.0000000000 -0.4119832958', 'O 0.0000000000 0.0000000000 1.9917721302', 'O 0.0000000000 -1.8912698030 0.4119832962', 'O 0.0000000000 1.8912698030 0.4119832962', 'O -1.8912698030 -1.8912698030 2.8157387222', 'O -1.8912698030 1.8912698030 2.8157387222', 'O 1.8912698030 -1.8912698030 2.8157387222', 'O 1.8912698030 1.8912698030 2.8157387222', 'O 0.0000000000 -1.8912698030 4.3955275562', 'O 0.0000000000 1.8912698030 4.3955275562', 'O -1.8912698030 -1.8912698030 -2.8157387228', 'O -1.8912698030 1.8912698030 -2.8157387228', 'O 1.8912698030 -1.8912698030 -2.8157387228', 'O 1.8912698030 1.8912698030 -2.8157387228', 'O -1.8912698030 0.0000000000 -4.3955275568', 'O 1.8912698030 0.0000000000 -4.3955275568', 'O 0.0000000000 0.0000000000 -1.9917721298', 'O 0.0000000000 -3.7825396060 1.9917721302', 'O 0.0000000000 3.7825396060 1.9917721302', 'O -3.7825396060 -1.8912698030 -5.2194941488', 'O -1.8912698030 -3.7825396060 -4.3955275568', 'O -1.8912698030 -1.8912698030 -6.7992829828', 'O 0.0000000000 -1.8912698030 -5.2194941488', 'O -3.7825396060 -1.8912698030 4.3955275562', 'O -1.8912698030 -3.7825396060 5.2194941492', 'O -1.8912698030 -1.8912698030 6.7992829832', 'O -1.8912698030 0.0000000000 5.2194941492', 'O -3.7825396060 1.8912698030 -5.2194941488', 'O -1.8912698030 1.8912698030 -6.7992829828', 'O -1.8912698030 3.7825396060 -4.3955275568', 'O 0.0000000000 1.8912698030 -5.2194941488', 'O -3.7825396060 1.8912698030 4.3955275562', 'O -1.8912698030 1.8912698030 6.7992829832', 'O -1.8912698030 3.7825396060 5.2194941492', 'O 1.8912698030 -3.7825396060 -4.3955275568', 'O 1.8912698030 -1.8912698030 -6.7992829828', 'O 3.7825396060 -1.8912698030 -5.2194941488', 'O 1.8912698030 -3.7825396060 5.2194941492', 'O 1.8912698030 -1.8912698030 6.7992829832', 'O 1.8912698030 0.0000000000 5.2194941492', 'O 3.7825396060 -1.8912698030 4.3955275562', 'O 1.8912698030 1.8912698030 -6.7992829828', 'O 1.8912698030 3.7825396060 -4.3955275568', 'O 3.7825396060 1.8912698030 -5.2194941488', 'O 1.8912698030 1.8912698030 6.7992829832', 'O 1.8912698030 3.7825396060 5.2194941492', 'O 3.7825396060 1.8912698030 4.3955275562', 'O -3.7825396060 0.0000000000 -1.9917721298', 'O 3.7825396060 0.0000000000 -1.9917721298']

energy_crystal = calculate_energy(centered_xyz_Li4H4_alone)

print(energy_crystal)