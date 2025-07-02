import time
import os
import numpy as np
import ctypes
import cupy as cp

from collections import Counter
from pyscf import pbc
from pyscf.pbc import gto, dft


# from pyscf import lib, gto, dft
from pyscf.geomopt.geometric_solver import optimize
# from gpu4pyscf import dft
# from mpi4pyscf import dft

from Compound_Properties import get_spin
from External_Saving import save_opt_xyz, save_opt_csv

_loaderpath = 'libdftd3-master/lib'
# libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)
libdftd3 = None


init_guess = ['hcore', 'minao', 'atom']

bases = ['6-31G', 'STO-6G', 'def2-svp', 'aug-cc-pVDZ']
density_fit_bases = ['weigend', 'def2-svp']

wrapper_xe_funcs = ['b3-lyp', 'pbe', 'b97-d', 'SVWN', 'b-lyp']
select_wrap_xe = 1


def dftd3_wrapper(mf, mol):
    """
        Takes a setup molecule system and calculates total energy whilst taking into account
        attractive and repulsive forces.

        :param mf: Mean-field object.
        :param mol: Molecule system.
        :return: System energy and forces.
    """

    coords = mol.atom_coords()
    itype = np.zeros(mol.natm, dtype=np.int32)
    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        itype[ia] = lib.parameters.NUC[symb]

    func = wrapper_xe_funcs[select_wrap_xe].encode()

    version = 4
    tz = 0
    edisp = np.zeros(1)
    grad = np.zeros((mol.natm, 3))

    libdftd3.wrapper(
        ctypes.c_int(mol.natm),
        coords.ctypes.data_as(ctypes.c_void_p),
        itype.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_char_p(func),
        ctypes.c_int(version),
        ctypes.c_int(tz),
        edisp.ctypes.data_as(ctypes.c_void_p),
        grad.ctypes.data_as(ctypes.c_void_p),
    )                                                                                                                   # Call the libdftd3 wrapper.

    energy_with_dispersion = (mf.kernel() + edisp[0]) * 27.2114                                                         # Add dispersion correction to the SCF energy (in eV).

    forces = -grad * 27.2114                                                                                            # Convert to eV/Angstrom.

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

    mol = gto.M(
        verbose=4,
        atom=atoms,
        basis='cc-pvdz',
        ecp='cc-pvdz',
        unit='Angstrom',
        spin=get_spin(element_count)
    )                                                                                                                   # Define the molecule in PySCF.

    start_time = time.time()

    # mf = dft.RKS(mol).density_fit(auxbasis='def2-svp')
    mf = dft.RKS(mol)
    mf.conv_tol = 1e-4
    mf.level_shift = 0.5

    mf.xc = 'pbe'
    mf.init_guess = 'atom'
    # mf.disp = 'd3bj'

    mf.direct_scf = True
    mf.diis = True
    # mf.with_df._cderi_to_disk = True# Use DIIS for SCF convergence acceleration.                                                                                           # Use the 'dftd3' engine.

    # corrected_energy, _ = dftd3_wrapper(mf, mol)                                                                           # Call the dftd3 function.

    corrected_energy = mf.kernel() * 27.2114

    print('Energy HF: ', corrected_energy/27.2114)

    end_time = time.time()

    print("Total time taken for energy calculation: ", end_time - start_time)

    return corrected_energy


def optimise_geometry(atoms, num_fixed, name):
    """
    Perform the geometry optimization with rigid body constraints for the first `num_fixed` atoms.
    """

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    mol = gto.M(
        verbose=5,
        atom=atoms,
        basis='STO-6G',
        unit='Angstrom',
        spin=get_spin(element_count)
    )

    mf = dft.RKS(mol).density_fit(auxbasis='def2-svp')

    mf.direct_scf = True
    mf.diis = True
    mf.with_df._cderi_to_disk = True

    mf.xc = 'pbe'
    mf.init_guess = 'atom'
    mf.disp = 'd3bj'

    mf.grids.level = 4
    mf.level_shift = 0.5
    mf.conv_tol = 1e-4

    def write_constraints(num_fixed, name):
        os.makedirs("../Constraints", exist_ok=True)  # Ensure folder exists
        filename = f"Constraints/constraints_{name}.txt"

        with open(filename, "w") as f:
            f.write("$freeze\n")
            # f.write(f"rotation 1-{num_fixed}\n")  # Freezes rotation

            for i in range(1, num_fixed + 1):
                for j in range(i + 1, num_fixed + 1):  # Avoid repeating pairs (i, j) and (j, i)
                    f.write(f"distance {i} {j}\n")

        return filename

    constraint_path = write_constraints(num_fixed, name)

    params = {
        "constraints": constraint_path,
        "convergence_energy": 5e-1,
        "other": ["--conmethod", "1"]
    }

    start_time = time.time()

    opt = optimize(mf, **params)
    optimized_coords = opt.atom_coords()

    end_time = time.time()

    print('Optimisation time: ', end_time - start_time)

    atom_symbols = [atom[0] for atom in opt._atom]

    ANGSTROM_TO_BOHR = 1/1.8897259886
    optimized_coords_bohr = np.array(optimized_coords) * ANGSTROM_TO_BOHR

    optimized_atoms = [
        f"{symbol} {x:.6f} {y:.6f} {z:.6f}"
        for symbol, (x, y, z) in zip(atom_symbols, optimized_coords_bohr)
    ]

    save_opt_csv(optimized_coords_bohr, f'Optimised Coordinates/{name}_optimised_1_csv')
    save_opt_xyz(atom_symbols, optimized_coords_bohr, f"Optimised Coordinates/{name}_optimised_1_coords")

    return optimized_atoms



def calculate_new_energy(atoms):
    """
        Takes in the XYZ format of the 3D coordinates of the material and computes
        the total energy.

        :param atoms: 3D coordinates of a material.
        :return: The total energy in eV.
    """

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    cell = gto.Cell(
        verbose=4
    )                                                                                                                   # Define the molecule in PySCF.

    a = 3.58  # lattice constant in Angstrom

    cell.a = np.array([
        [a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, 0.0, a]
    ])

    cell.atom = atoms
    cell.mesh = [30, 30, 30]  # adjust depending on desired accuracy

    start_time = time.time()

    cell.build()

    # mf = dft.RKS(mol).density_fit(auxbasis='def2-svp')
    mf = dft.RKS(cell)  # K-point restricted Kohn-Sham DFT

    mf.conv_tol = 1e-4
    mf.level_shift = 0.5
    mf.grids.level = 0

    mf.xc = 'pbe'
    mf.init_guess = 'minao'
    # mf.disp = 'd3bj'

    mf.direct_scf = True
    mf.diis = True
    # mf.with_df._cderi_to_disk = True# Use DIIS for SCF convergence acceleration.                                                                                           # Use the 'dftd3' engine.

    # corrected_energy, _ = dftd3_wrapper(mf, mol)                                                                           # Call the dftd3 function.

    corrected_energy = mf.kernel() * 27.2114

    print('Energy HF: ', corrected_energy/27.2114)

    end_time = time.time()

    print("Total time taken for energy calculation: ", end_time - start_time)

    return corrected_energy


# coords = ['H -3.873662 -3.185369 5.797057', 'H -3.675897 -2.671708 4.414634', 'H -3.128901 0.125655 4.523488',
#           'H -3.659485 0.097794 3.023886', 'H -10.388724 1.929783 7.899367', 'H -9.014472 1.815038 7.360618',
#           'H -0.349124 -1.402653 1.993698', 'H 0.623614 -0.576550 1.957935', 'H -1.550132 2.505346 4.139707',
#           'H -0.589543 2.086406 3.297534', 'H 4.748285 -1.339658 5.055409', 'H 5.272041 -1.674399 6.433237',
#           'H 9.010782 1.846897 7.542184', 'H 10.411638 1.927485 8.010425']

# coords = ['H 3.06934903 4.02311175 4.08980207', 'H 1.98111175 5.11134903 4.08980207', 'O 2.104795 4.146795 4.193271']


def xyz_to_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip first two lines (usually number of atoms and comment)
    atom_lines = lines[2:]

    result = []
    for line in atom_lines:
        parts = line.strip().split()
        element = parts[0]
        x, y, z = parts[1], parts[2], parts[3]
        # Keep the string as is, no rounding
        result.append(f"{element} {x} {y} {z}")

    return result


coords = xyz_to_list('Cu_alone.xyz')
print(coords)

coords = ['Cu 1 0 0']

energy = calculate_new_energy(coords)
print('Energy eV: ', energy)

