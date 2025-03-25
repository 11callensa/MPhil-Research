import os
import numpy as np
import ctypes

from collections import Counter
from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.dft import libxc
from pyscf.geomopt import berny_solver
from functools import partial

from Compound_Properties import get_spin

_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)

xe_funcs = ['b3-lyp', 'pbe', 'b97-d', 'SVWN', 'b-lyp']
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

    func = xe_funcs[select_xe].encode()

    functionals = libxc.available_libxc_functionals()

    # # Print all functional names
    # print("Available Libxc functionals:")
    # for name, func_id in functionals.items():
    #     print(f"{name} (ID: {func_id})")
    #
    # # Check for LDA-related functionals
    # lda_functionals = {name: func_id for name, func_id in functionals.items() if "LDA" in name}
    # print("\nLDA-related functionals:")
    # for name, func_id in lda_functionals.items():
    #     print(f"{name} (ID: {func_id})")
    # print("Correlation Function: ", xe_funcs[select_xe])

    # func = xe_funcs[select_xe].encode()  # Encoding required for ctypes
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

    # print("Forces: ", forces)

    force_magnitude = np.linalg.norm(forces[-1])
    direction = forces[-1] / force_magnitude  # Normalized direction vector

    # print("Force magnitude: ", force_magnitude)
    # print("Force direction: ", direction)

    return energy_with_dispersion, forces


def mf_grad_with_dftd3(geom, mf_grad_scan, fixed_coords, num_fixed):

    e_tot, g_rhf = mf_grad_scan(geom)
    mol = mf_grad_scan.mol
    func = 'pbe0'.encode()
    version = 4
    tz = 0
    coords = mol.atom_coords()
    itype = np.zeros(mol.natm, dtype=np.int32)

    for ia in range(mol.natm):
        symb = mol.atom_pure_symbol(ia)
        itype[ia] = lib.parameters.NUC[symb]

    edisp = np.zeros(1)
    grad = np.zeros((mol.natm, 3))

    # Run DFT-D3 dispersion corrections
    libdftd3.wrapper(ctypes.c_int(mol.natm),
                     coords.ctypes.data_as(ctypes.c_void_p),
                     itype.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_char_p(func),
                     ctypes.c_int(version),
                     ctypes.c_int(tz),
                     edisp.ctypes.data_as(ctypes.c_void_p),
                     grad.ctypes.data_as(ctypes.c_void_p))

    # **Fix the Li4H4 crystal in space**
    coords[:num_fixed] = fixed_coords  # Reset positions of first 8 atoms to their original locations
    grad[:num_fixed] = np.zeros((num_fixed, 3))  # Zero out forces so they don’t experience movement

    # Return the energy and gradients
    e_tot += edisp[0]
    g_rhf += grad

    return e_tot, g_rhf


def setup_compound(atoms):

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    mol = gto.M(  # Define the molecule in PySCF
        verbose=4,
        atom=atoms,
        basis='aug-cc-pvdz',
        unit='Angstrom',
        spin=get_spin(element_count)
    )

    # mol.symmetry = 0

    # Default settings
    level_shift = 0.5
    grid_level = 3
    damp = 0.2
    scf_convergence = 1e-4
    force_convergence = 1e-3

    # Ask user if they want to edit settings
    response = input("Would you like to edit the optimisation settings? "
                     "(Default: Level Shift = 0.5, Grid Level = 3, SCF Convergence = 1e-4, Force Convergence = 1e-3) "
                     "(y/n): ").strip().lower()

    if response == 'y':
        try:
            level_shift = float(input(f"Enter Level Shift (Default: {level_shift}): ") or level_shift)
            grid_level = int(input(f"Enter Grid Level (Default: {grid_level}): ") or grid_level)
            damp = float(input(f"Enter Damp Level (Default: {damp}): " or damp))
            scf_convergence = float(input(f"Enter SCF Convergence (Default: {scf_convergence}): ") or scf_convergence)
            force_convergence = float(
                input(f"Enter Force Convergence (Default: {force_convergence}): ") or force_convergence)

        except ValueError:
            print("Invalid input detected, using default settings.")

    mf_grad_scan = dft.RKS(mol).nuc_grad_method().as_scanner()
    mf_grad_scan.base = scf.addons.remove_linear_dep_(mf_grad_scan.base)
    mf_grad_scan.base.verbose = 5
    mf_grad_scan.base.xc = 'pbe0'
    mf_grad_scan.base.init_guess = 'hcore'

    mf_grad_scan.base.level_shift = level_shift
    mf_grad_scan.base.grids.level = grid_level
    mf_grad_scan.base.damp = damp
    mf_grad_scan.base.conv_tol = scf_convergence
    mf_grad_scan.base.conv_tol_grad = force_convergence

    mf_grad_scan.base.grids.prune = dft.gen_grid.treutler_prune
    mf_grad_scan.grid_response = False

    os.environ["OMP_NUM_THREADS"] = "4"  # Use 4 CPU cores (adjust as needed)

    coords_init = mol.atom_coords()

    print("Initial coordinates: ", coords_init)

    return mol, mf_grad_scan, coords_init


def optimiser(mol, mf_grad_scan, coords, num_compound):

    fixed_coords = coords[:num_compound].copy()

    mf = berny_solver.as_pyscf_method(mol, partial(mf_grad_with_dftd3, mf_grad_scan=mf_grad_scan, fixed_coords=fixed_coords, num_fixed=num_compound))

    print("mf created")

    _, mol = berny_solver.kernel(mf, maxsteps=30)

    print("Final Coordinates: ", mol.atom_coords() * lib.param.BOHR)

    coords = mol.atom_coords() * lib.param.BOHR

    return coords


centered_xyz = ['Li 0.0000000000 0.0000000000 0.0000000000', 'Li 0.0000000000 -2.0086193919 -2.0086193919', 'Li 2.0086193919 -2.0086193919 0.0000000000', 'Li 2.0086193919 0.0000000000 -2.0086193919', 'H 0.0000000000 -2.0086193919 0.0000000000', 'H 0.0000000000 0.0000000000 -2.0086193919', 'H 2.0086193919 0.0000000000 0.0000000000', 'H 2.0086193919 -2.0086193919 -2.0086193919', 'H 1.0887637941 -1.0887637941 1.5000000000', 'H 1.5893953951 -1.5893953951 1.5000000000', 'H 0.4192239968 -0.4192239968 3.0000000000', 'H 0.9198555978 -0.9198555978 3.0000000000', 'H -1.5000000000 -0.9198555978 -0.9198555978', 'H -1.5000000000 -0.4192239968 -0.4192239968', 'H -3.0000000000 -1.5893953951 -1.5893953951', 'H -3.0000000000 -1.0887637941 -1.0887637941', 'H 0.4192239968 1.5000000000 -0.4192239968', 'H 0.9198555978 1.5000000000 -0.9198555978', 'H 1.0887637941 3.0000000000 -1.0887637941', 'H 1.5893953951 3.0000000000 -1.5893953951', 'H 1.5893953951 -3.5086193919 -0.4192239968', 'H 1.0887637941 -3.5086193919 -0.9198555978', 'H 0.9198555978 -5.0086193919 -1.0887637941', 'H 0.4192239968 -5.0086193919 -1.5893953951', 'H 3.5086193919 -1.0887637941 -0.9198555978', 'H 3.5086193919 -1.5893953951 -0.4192239968', 'H 5.0086193919 -0.4192239968 -1.5893953951', 'H 5.0086193919 -0.9198555978 -1.0887637941', 'H 0.4192239968 -1.5893953951 -3.5086193919', 'H 0.9198555978 -1.0887637941 -3.5086193919', 'H 1.0887637941 -0.9198555978 -5.0086193919', 'H 1.5893953951 -0.4192239968 -5.0086193919']

mol_opt, mf_grad_scan_opt, initial_opt_coords = setup_compound(centered_xyz)
optimised_xyz = optimiser(mol_opt, mf_grad_scan_opt, initial_opt_coords, 8)

mf = dft.RKS(mol_opt)

combined_energy = dftd3(mf, mol_opt)

print("Combined energy: ", combined_energy)
