import os
import numpy as np
import ctypes

from collections import Counter
from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.geomopt import berny_solver
from functools import partial
import resource
import psutil

from Compound_Properties import get_spin
from Mol_Geometry import (find_centroid, find_direction, find_distances, find_translation, find_rotation,
                          apply_translation, apply_rotation)

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


def mf_grad_with_dftd3(geom, mf_grad_scan):
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

    # Return the energy and gradients (no modification to forces)
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

    # Default settings
    level_shift = 0.5
    grid_level = 3
    damp = 0.2
    scf_convergence = 1e-4
    force_convergence = 1e-3

    # Ask user if they want to edit the optimisation settings
    response = input("Would you like to edit the optimisation settings? "
                     "(Default: Level Shift = 0.5, Grid Level = 3 (1 - 9), SCF Convergence = 1e-4, Force Convergence = 1e-3) "
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

    print("Initial coordinates: ", coords_init * lib.param.BOHR)

    return mol, mf_grad_scan, coords_init


def optimiser(mol, mf_grad_scan, coords, num_compound, maxsteps=100, force_change_convergence=1e-09):
    prev_coords = coords[:num_compound].copy()
    prev_centroid = find_centroid(prev_coords)
    prev_direction = find_direction(prev_coords)
    # prev_distances = find_distances(prev_coords, prev_centroid)

    mf = berny_solver.as_pyscf_method(mol, partial(mf_grad_with_dftd3, mf_grad_scan=mf_grad_scan))

    prev_forces_array = None  # Initialize to None

    for step in range(maxsteps):
        print(f"OPTIMISATION STEP {step + 1}")

        print("Pre optimisation step coords: ", mol.atom_coords() * lib.param.BOHR)

        # Perform optimization using the Berny solver
        _, mol = berny_solver.kernel(mf, maxsteps=6)
        current_coords = mol.atom_coords()

        print("Post optimised coords: ", mol.atom_coords() * lib.param.BOHR)

        current_fixed_coords = current_coords[:num_compound]
        trans_vec = find_translation(current_fixed_coords, prev_centroid)
        rotation_axis, angle = find_rotation(current_fixed_coords, prev_direction)
        transformed_fixed_coords = apply_translation(prev_coords, trans_vec)

        if rotation_axis is not None:
            transformed_fixed_coords = apply_rotation(transformed_fixed_coords, rotation_axis, angle)

        combined_coords = np.vstack((transformed_fixed_coords, current_coords[num_compound:]))
        mol.set_geom_(combined_coords, unit="Bohr")

        print("Post adjusted coords: ", mol.atom_coords() * lib.param.BOHR)

        # Calculate forces
        e_tot, grad_values = mf_grad_with_dftd3(mol.atom_coords(), mf_grad_scan)

        forces_array = grad_values  # Forces at the current step

        # Initialize force_change for the first iteration
        force_change = 0

        if prev_forces_array is not None:
            # Compute the change in forces by comparing current forces with previous ones
            force_change = np.linalg.norm(forces_array - prev_forces_array)

            print("Change in forces: ", force_change)
            print("Force change convergence threshold: ", force_change_convergence)

            if force_change < force_change_convergence:
                print(f"Convergence reached at step {step + 1} with force change {force_change:.5e}")
                break

        # Update the previous forces for the next iteration
        prev_forces_array = forces_array.copy()

        print(f"Step completed. Force change: {force_change:.5e}")

    coords_optimized = mol.atom_coords()
    return coords_optimized


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

    corrected_energy, forces = dftd3(mf, mol)                                                                           # Call the dftd3 function

    return corrected_energy, forces
