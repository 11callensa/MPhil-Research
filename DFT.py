import os

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_MAX_THREADS"] = "1"

import time
import numpy as np
import ctypes

from collections import Counter
from pyscf import lib
from pyscf import scf, gto
from gpu4pyscf import dft, df
from pyscf.geomopt import berny_solver
from functools import partial
# from mpi4pyscf import scf, dft

from Compound_Properties import get_spin
from Mol_Geometry import (find_centroid, find_direction, find_distances, find_translation, find_rotation,
                          apply_translation, apply_rotation)
from External_Saving import save_optimised_coords


_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)

xe_funcs = ['b3-lyp', 'pbe', 'b97-d', 'SVWN', 'b-lyp']
select_xe = 0


def dftd3(mf, mol):
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

    func = xe_funcs[select_xe].encode()

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


def mf_grad_with_dftd3(geom, mf_grad_scan):
    """
        Connects the mean-field object with dftd3.

        :param geom: The geometry of the system.
        :param mf_grad_scan: Parameters of the mean-field object.
        :return: Energy and gradient of energy (Force).
    """

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

    libdftd3.wrapper(ctypes.c_int(mol.natm),
                     coords.ctypes.data_as(ctypes.c_void_p),
                     itype.ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_char_p(func),
                     ctypes.c_int(version),
                     ctypes.c_int(tz),
                     edisp.ctypes.data_as(ctypes.c_void_p),
                     grad.ctypes.data_as(ctypes.c_void_p))                                                              # Run DFT-D3 dispersion corrections.

    e_tot += edisp[0]                                                                                                   # Return the energy and gradients (no modification to forces).
    g_rhf += grad

    return e_tot, g_rhf


def setup_compound(atoms):
    """
        Sets up the compound and parameters of the DFT calculation.

        :param atoms: 3D coordinates of the system.
        :return: The packaged compound, calculation parameters and initial coordinates for the optimisation.
    """

    symbols = [line.split()[0] for line in atoms]
    element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

    mol = gto.M(
        verbose=4,
        atom=atoms,
        basis='aug-cc-pvdz',
        unit='Angstrom',
        spin=get_spin(element_count)
    )                                                                                                                   # Define the molecule in PySCF.

    level_shift = 0.5                                                                                                   # Default settings.
    grid_level = 3
    damp = 0.2
    scf_convergence = 1e-4
    force_convergence = 1e-3

    response = input("Would you like to edit the optimisation settings? "
                     "(Default: Level Shift = 0.5, Grid Level = 3 (1 - 9), SCF Convergence = 1e-4, "
                     "Force Convergence = 1e-3) (y/n): ").strip().lower()                                               # Ask user if they want to edit the optimisation settings.

    if response == 'y':                                                                                                 # Get user inputs for settings.
        try:
            level_shift = float(input(f"Enter Level Shift (Default: {level_shift}): ") or level_shift)
            grid_level = int(input(f"Enter Grid Level (Default: {grid_level}): ") or grid_level)
            damp = float(input(f"Enter Damp Level (Default: {damp}): " or damp))
            scf_convergence = float(input(f"Enter SCF Convergence (Default: {scf_convergence}): ") or scf_convergence)
            force_convergence = float(
                input(f"Enter Force Convergence (Default: {force_convergence}): ") or force_convergence)

        except ValueError:
            print("Invalid input detected, using default settings.")

    mf_grad_scan = dft.RKS(mol).density_fit(auxbasis='weigend').nuc_grad_method().as_scanner()
    mf_grad_scan.to_gpu()
    mf_grad_scan.base = scf.addons.remove_linear_dep_(mf_grad_scan.base)
    mf_grad_scan.base.verbose = 5
    mf_grad_scan.base.xc = 'pbe0'
    mf_grad_scan.base.init_guess = 'hcore'

    mf_grad_scan.base.level_shift = level_shift
    mf_grad_scan.base.grids.level = grid_level
    mf_grad_scan.base.damp = damp
    mf_grad_scan.base.conv_tol = scf_convergence
    mf_grad_scan.base.conv_tol_grad = force_convergence

    mf_grad_scan.grid_response = False

    coords_init = mol.atom_coords()                                                                                     # Extract the initial coordinates.

    print("Initial coordinates: ", coords_init * lib.param.BOHR)

    return mol, mf_grad_scan, coords_init


def optimiser(mol, mf_grad_scan, coords, num_compound, maxsteps=1, force_change_convergence=1e-09):
    """
        Runs the optimisation that models how hydrogen adsorbs to the compound. Rigid body rules are enforced
        on the compound throughout the optimisation to avoid compound atoms being pulled out of the compound.

        :param mol: Compound setup as a molecule.
        :param mf_grad_scan: Optimisation parameters.
        :param coords: Initial coordinates.
        :param num_compound: Number of atoms in the compound.
        :param maxsteps: Sets the maximum number of steps for the optimisation.
        :param force_change_convergence: Sets the convergence in force that ends the optimisation.
        :return: Optimised coordinates.
    """

    prev_coords = coords[:num_compound].copy()                                                                          # Sets the initial coordinates of the compound.
    prev_centroid = find_centroid(prev_coords)                                                                          # Find the original centroid of the compound.
    prev_direction = find_direction(prev_coords)                                                                        # Find the original direction vectors between all of the compounds' atoms.
    # prev_distances = find_distances(prev_coords, prev_centroid)

    start_time2 = time.time()
    mf = berny_solver.as_pyscf_method(mol, partial(mf_grad_with_dftd3, mf_grad_scan=mf_grad_scan))                      # Run optimisation using mf_grad_with_dftd3 as the method, using the parameters in mf_grad_scan.
    end_time2 = time.time()
    print('Set method time: ', end_time2-start_time2)

    mf.direct_scf = True
    mf.diis = False                                                                                                      # Use DIIS for SCF convergence acceleration.

    prev_forces_array = None                                                                                            # Initialize to None.

    start_time = time.time()

    for step in range(maxsteps):
        print(f"OPTIMISATION STEP {step + 1}")
        print("Pre optimisation step coords: ", mol.atom_coords() * lib.param.BOHR)

        start_time3 = time.time()
        _, mol = berny_solver.kernel(mf, maxsteps=6)                                                                    # Perform optimization using the Berny solver - take in mf object and update the coordinates using force, to update mol.
        end_time3 = time.time()
        print('Berny solver 6 iterations time: ', end_time3-start_time3)
        input('PAUSE: ')

        current_coords = mol.atom_coords()                                                                              # Extract the current coordinates from mol.

        print("Post optimised coords: ", mol.atom_coords() * lib.param.BOHR)

        current_fixed_coords = current_coords[:num_compound]                                                            # Rigid body enforcement: Extract the atoms in the compound.
        trans_vec = find_translation(current_fixed_coords, prev_centroid)                                               # Find the translation vector of the centroid as it moves.
        rotation_axis, angle = find_rotation(current_fixed_coords, prev_direction)                                      # Find the rotation vector of the compound based on the previous direction  vector between its atoms.
        transformed_fixed_coords = apply_translation(prev_coords, trans_vec)                                            # Translate the compound back to its original place.

        if rotation_axis is not None:
            transformed_fixed_coords = apply_rotation(transformed_fixed_coords, rotation_axis, angle)                   # Rotate the compound back to its original orientation.

        combined_coords = np.vstack((transformed_fixed_coords, current_coords[num_compound:]))                          # Append the hydrogen atoms to the 'fixed' compound atoms.
        mol.set_geom_(combined_coords, unit="Bohr")                                                                     # Convert units to Bohr.

        print("Post adjusted coords: ", mol.atom_coords() * lib.param.BOHR)

        e_tot, grad_values = mf_grad_with_dftd3(mol.atom_coords(), mf_grad_scan)                                        # Calculate energy and forces.

        forces_array = grad_values                                                                                      # Forces at the current step.

        force_change = 0                                                                                                # Initialize force_change.

        if prev_forces_array is not None:
            force_change = np.linalg.norm(forces_array - prev_forces_array)                                             # Compute the change in forces by comparing current forces with previous ones.

            print("Change in forces: ", force_change)
            print("Force change convergence threshold: ", force_change_convergence)

            if force_change < force_change_convergence:
                print(f"Convergence reached at step {step + 1} with force change {force_change:.5e}")
                break

        prev_forces_array = forces_array.copy()                                                                         # Update the previous forces for the next iteration.

        print(f"Step completed. Force change: {force_change:.5e}")

    end_time = time.time()

    print("Optimisation time: ", end_time - start_time)

    coords_optimized = mol.atom_coords()                                                                                # Extract optimised coordinates.

    print("Coords Optimised: ", coords_optimized)

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

    mol = gto.M(
        verbose=5,
        atom=atoms,
        basis='6-31G',
        unit='Angstrom',
        spin=get_spin(element_count)
    )                                                                                                                   # Define the molecule in PySCF.

    start_time = time.time()

    mf = dft.RKS(mol).density_fit(auxbasis='weigend')
    mf.conv_tol = 1e-4
    mf.level_shift = 0.5
    mf.xc = 'pbe'
    mf.init_guess = 'atom'
    mf.direct_scf = True
    mf.diis = True
    mf.with_df._cderi_to_disk = True# Use DIIS for SCF convergence acceleration.                                                                                           # Use the 'dftd3' engine.

    corrected_energy, forces = dftd3(mf, mol)                                                                           # Call the dftd3 function.

    end_time = time.time()

    print("Total time taken for energy calculation: ", start_time - end_time)

    return corrected_energy, forces

####################################################

# coordinates = ['Li -1.0043096959 1.0043096959 1.0043096959', 'Li -1.0043096959 -1.0043096959 -1.0043096959', 'Li 1.0043096959 -1.0043096959 1.0043096959', 'Li 1.0043096959 1.0043096959 -1.0043096959', 'H -1.0043096959 -1.0043096959 1.0043096959', 'H -1.0043096959 1.0043096959 -1.0043096959', 'H 1.0043096959 1.0043096959 1.0043096959', 'H 1.0043096959 -1.0043096959 -1.0043096959']
#
# energy, _ = calculate_energy(coordinates)
#
# print("Crystal alone energy: ", energy)

#####################################################

# num_atoms = 8
# name = 'Li4H4-Trial'
# combined_xyz_Li4H4 = ['Li -1.0043096959 1.0043096959 1.0043096959', 'Li -1.0043096959 -1.0043096959 -1.0043096959', 'Li 1.0043096959 -1.0043096959 1.0043096959', 'Li 1.0043096959 1.0043096959 -1.0043096959', 'H -1.0043096959 -1.0043096959 1.0043096959', 'H -1.0043096959 1.0043096959 -1.0043096959', 'H 1.0043096959 1.0043096959 1.0043096959', 'H 1.0043096959 -1.0043096959 -1.0043096959', 'H -0.5850856992 -1.7543096959 -0.5850856992', 'H -0.0844540981 -1.7543096959 -0.0844540981', 'H 0.0844540981 -2.3793096959 0.0844540981', 'H 0.5850856992 -2.3793096959 0.5850856992', 'H 0.0844540981 -0.0844540981 1.7543096959', 'H 0.5850856992 -0.5850856992 1.7543096959', 'H -0.5850856992 0.5850856992 2.3793096959', 'H -0.0844540981 0.0844540981 2.3793096959', 'H 1.7543096959 -0.5850856992 0.5850856992', 'H 1.7543096959 -0.0844540981 0.0844540981', 'H 2.3793096959 0.0844540981 -0.0844540981', 'H 2.3793096959 0.5850856992 -0.5850856992', 'H -1.7543096959 -0.5850856992 -0.5850856992', 'H -1.7543096959 -0.0844540981 -0.0844540981', 'H -2.3793096959 0.0844540981 0.0844540981', 'H -2.3793096959 0.5850856992 0.5850856992', 'H -0.5850856992 1.7543096959 0.5850856992', 'H -0.0844540981 1.7543096959 0.0844540981', 'H 0.0844540981 2.3793096959 -0.0844540981', 'H 0.5850856992 2.3793096959 -0.5850856992', 'H -0.0844540981 -0.0844540981 -1.7543096959', 'H -0.5850856992 -0.5850856992 -1.7543096959', 'H 0.5850856992 0.5850856992 -2.3793096959', 'H 0.0844540981 0.0844540981 -2.3793096959']

# num_atoms = 63
# name = 'TiO2'
# combined_xyz_TiO2 = ['Ti 0.0000000000 0.0000000000 0.0000000002', 'Ti 0.0000000000 -1.8912698030 2.4037554262', 'Ti 0.0000000000 1.8912698030 2.4037554262', 'Ti -1.8912698030 -1.8912698030 -4.8075108528', 'Ti -1.8912698030 -1.8912698030 4.8075108532', 'Ti -1.8912698030 1.8912698030 -4.8075108528', 'Ti -1.8912698030 1.8912698030 4.8075108532', 'Ti 1.8912698030 -1.8912698030 -4.8075108528', 'Ti 1.8912698030 -1.8912698030 4.8075108532', 'Ti 1.8912698030 1.8912698030 -4.8075108528', 'Ti 1.8912698030 1.8912698030 4.8075108532', 'Ti -1.8912698030 0.0000000000 -2.4037554268', 'Ti 1.8912698030 0.0000000000 -2.4037554268', 'O -1.8912698030 0.0000000000 -0.4119832958', 'O 1.8912698030 0.0000000000 -0.4119832958', 'O 0.0000000000 0.0000000000 1.9917721302', 'O 0.0000000000 -1.8912698030 0.4119832962', 'O 0.0000000000 1.8912698030 0.4119832962', 'O -1.8912698030 -1.8912698030 2.8157387222', 'O -1.8912698030 1.8912698030 2.8157387222', 'O 1.8912698030 -1.8912698030 2.8157387222', 'O 1.8912698030 1.8912698030 2.8157387222', 'O 0.0000000000 -1.8912698030 4.3955275562', 'O 0.0000000000 1.8912698030 4.3955275562', 'O -1.8912698030 -1.8912698030 -2.8157387228', 'O -1.8912698030 1.8912698030 -2.8157387228', 'O 1.8912698030 -1.8912698030 -2.8157387228', 'O 1.8912698030 1.8912698030 -2.8157387228', 'O -1.8912698030 0.0000000000 -4.3955275568', 'O 1.8912698030 0.0000000000 -4.3955275568', 'O 0.0000000000 0.0000000000 -1.9917721298', 'O 0.0000000000 -3.7825396060 1.9917721302', 'O 0.0000000000 3.7825396060 1.9917721302', 'O -3.7825396060 -1.8912698030 -5.2194941488', 'O -1.8912698030 -3.7825396060 -4.3955275568', 'O -1.8912698030 -1.8912698030 -6.7992829828', 'O 0.0000000000 -1.8912698030 -5.2194941488', 'O -3.7825396060 -1.8912698030 4.3955275562', 'O -1.8912698030 -3.7825396060 5.2194941492', 'O -1.8912698030 -1.8912698030 6.7992829832', 'O -1.8912698030 0.0000000000 5.2194941492', 'O -3.7825396060 1.8912698030 -5.2194941488', 'O -1.8912698030 1.8912698030 -6.7992829828', 'O -1.8912698030 3.7825396060 -4.3955275568', 'O 0.0000000000 1.8912698030 -5.2194941488', 'O -3.7825396060 1.8912698030 4.3955275562', 'O -1.8912698030 1.8912698030 6.7992829832', 'O -1.8912698030 3.7825396060 5.2194941492', 'O 1.8912698030 -3.7825396060 -4.3955275568', 'O 1.8912698030 -1.8912698030 -6.7992829828', 'O 3.7825396060 -1.8912698030 -5.2194941488', 'O 1.8912698030 -3.7825396060 5.2194941492', 'O 1.8912698030 -1.8912698030 6.7992829832', 'O 1.8912698030 0.0000000000 5.2194941492', 'O 3.7825396060 -1.8912698030 4.3955275562', 'O 1.8912698030 1.8912698030 -6.7992829828', 'O 1.8912698030 3.7825396060 -4.3955275568', 'O 3.7825396060 1.8912698030 -5.2194941488', 'O 1.8912698030 1.8912698030 6.7992829832', 'O 1.8912698030 3.7825396060 5.2194941492', 'O 3.7825396060 1.8912698030 4.3955275562', 'O -3.7825396060 0.0000000000 -1.9917721298', 'O 3.7825396060 0.0000000000 -1.9917721298', 'H 3.2987979595 2.7074108767 5.6937699773', 'H 2.7919819949 3.0243402422 6.0731871030', 'H -2.9942296175 3.7352120481 -6.0371396807', 'H -3.3111589830 3.2283960834 -6.4165568063', 'H -3.1289880900 -3.0700756895 5.6066579079', 'H -2.9617918644 -2.6616754295 6.1602991724', 'H -3.4672171994 -3.3541306416 -6.1263950571', 'H -2.8381714011 -3.6094774899 -6.3273014299', 'H -2.7919819949 3.0243402422 6.0731871030', 'H -3.2987979595 2.7074108767 5.6937699773', 'H 2.9942296175 3.7352120481 -6.0371396807', 'H 3.3111589830 3.2283960834 -6.4165568063', 'H 3.1803984587 -2.9177165529 -5.7830253538', 'H 2.5513526604 -3.1730634013 -5.9839317265', 'H 3.3541306418 -3.4672171993 6.1263950571', 'H 3.6094774901 -2.8381714009 6.3273014297', 'H -2.7748619468 0.8236588835 -7.0384422251', 'H -3.2301403199 0.4371876519 -6.6581455118', 'H -3.8059586000 -0.4371876519 -6.9915185835', 'H -4.2612369731 -0.8236588835 -6.6112218702', 'H -2.8095400884 3.9249294924 -1.5530389461', 'H -3.2945062248 3.4399633560 -1.3773127582', 'H -3.8819050943 3.7364479630 1.3773127583', 'H -4.3668712307 3.2514818266 1.5530389461', 'H -0.3009980212 4.5325396060 -1.0609247698', 'H -0.9598485141 4.5325396060 -1.3201158731', 'H 0.9598485141 5.1575396060 2.1440824655', 'H 0.3009980212 5.1575396060 1.8848913622', 'H 2.9783275735 3.7561420073 -1.8034862452', 'H 3.1257187396 3.6087508412 -1.1268654590', 'H 3.8819050943 3.7364479630 1.3773127583', 'H 4.3668712307 3.2514818266 1.5530389461', 'H -4.5325396060 -0.3009980212 -1.8848913622', 'H -4.5325396060 -0.9598485141 -2.1440824654', 'H -5.1575396060 0.9598485141 1.3201158728', 'H -5.1575396060 0.3009980212 1.0609247696', 'H -3.4429691664 -3.2915004144 -1.6367911677', 'H -3.9219236820 -2.8125458987 -1.8428715985', 'H -3.2544876370 -4.3638654203 1.8428715983', 'H -3.7334421526 -3.8849109047 1.6367911675', 'H -0.9598485141 -4.5325396060 2.1440824655', 'H -0.3009980212 -4.5325396060 1.8848913622', 'H 0.3009980212 -5.1575396060 -1.0609247698', 'H 0.9598485141 -5.1575396060 -1.3201158731', 'H -3.9117561788 0.8530068119 5.4442058766', 'H -3.5713331432 0.4078397234 5.8768748264', 'H -3.7725231801 -0.4078397234 6.6319251742', 'H -3.4321001444 -0.8530068119 7.0645941240', 'H -0.2764232677 3.6329244010 6.3216975909', 'H -0.9844232677 3.6329244010 6.3216975909', 'H 0.6304232677 3.1314879376 7.5549079109', 'H 0.6304232677 3.6748611001 7.1010250997', 'H 0.2764232677 3.7415446609 -5.6605403519', 'H 0.9844232677 3.7415446609 -5.6605403519', 'H -0.6304232677 3.3834164765 -7.1264698175', 'H -0.6304232677 3.8212068477 -6.5700494810', 'H 4.5325396060 0.3009980212 1.0609247696', 'H 4.5325396060 0.9598485141 1.3201158728', 'H 5.1575396060 -0.9598485141 -2.1440824654', 'H 5.1575396060 -0.3009980212 -1.8848913622', 'H 3.1257187397 -3.6087508411 2.0781417759', 'H 2.9783275734 -3.7561420073 1.4015209898', 'H 3.8849109047 -3.7334421526 -1.6367911677', 'H 4.3638654203 -3.2544876370 -1.8428715985', 'H 3.4052852145 -0.4371876519 -6.5118459471', 'H 3.8605635875 -0.8236588835 -6.1315492338', 'H 3.1314879376 0.6304232677 -7.5549079105', 'H 3.6748611001 0.6304232677 -7.1010250993', 'H -0.6304232677 -0.9844232677 -7.5492829828', 'H -0.6304232677 -0.2764232677 -7.5492829828', 'H 0.2764232677 0.6304232677 -8.1742829828', 'H 0.9844232677 0.6304232677 -8.1742829828', 'H 0.8530068119 -3.9117561788 -5.4442058770', 'H 0.4078397234 -3.5713331430 -5.8768748268', 'H -0.4078397234 -3.7725231799 -6.6319251744', 'H -0.8530068119 -3.4321001442 -7.0645941241', 'H 3.7415446610 -0.9844232677 5.6605403515', 'H 3.7415446610 -0.2764232677 5.6605403515', 'H 3.8212068478 0.6304232677 6.5700494808', 'H 3.3834164767 0.6304232677 7.1264698174', 'H -0.3801074671 -0.3801074671 7.5492829832', 'H -0.8807390682 -0.8807390682 7.5492829832', 'H 0.8807390682 0.8807390682 8.1742829832', 'H 0.3801074671 0.3801074671 8.1742829832', 'H 0.2764232677 -3.6329244010 6.3216975909', 'H 0.9844232677 -3.6329244010 6.3216975909', 'H -0.8236588835 -3.1755353324 7.5181148619', 'H -0.4371876519 -3.6308137054 7.1378181486']
# num_atoms = len(combined_xyz_TiO2)
#
# print('Num atoms: ', num_atoms)
#
# start_time1 = time.time()
# mol, mf_grad_scan, initial_coordinates = setup_compound(combined_xyz_TiO2)
# end_time1 = time.time()
# print('Setup compound time: ', end_time1-start_time1)
#
# raw_optimised_xyz = optimiser(mol, mf_grad_scan, initial_coordinates, num_atoms)
file = f'Optimised Coordinates/{name}_optimised_coords'

save_optimised_coords(raw_optimised_xyz, file)

######################################################

optimised_xyz = ['Ti 0.0000000000 0.0000000000 0.0000000002', 'Ti 0.0000000000 -1.8912698030 2.4037554262', 'Ti 0.0000000000 1.8912698030 2.4037554262', 'Ti -1.8912698030 -1.8912698030 -4.8075108528', 'Ti -1.8912698030 -1.8912698030 4.8075108532', 'Ti -1.8912698030 1.8912698030 -4.8075108528', 'Ti -1.8912698030 1.8912698030 4.8075108532', 'Ti 1.8912698030 -1.8912698030 -4.8075108528', 'Ti 1.8912698030 -1.8912698030 4.8075108532', 'Ti 1.8912698030 1.8912698030 -4.8075108528', 'Ti 1.8912698030 1.8912698030 4.8075108532', 'Ti -1.8912698030 0.0000000000 -2.4037554268', 'Ti 1.8912698030 0.0000000000 -2.4037554268', 'O -1.8912698030 0.0000000000 -0.4119832958', 'O 1.8912698030 0.0000000000 -0.4119832958', 'O 0.0000000000 0.0000000000 1.9917721302', 'O 0.0000000000 -1.8912698030 0.4119832962', 'O 0.0000000000 1.8912698030 0.4119832962', 'O -1.8912698030 -1.8912698030 2.8157387222', 'O -1.8912698030 1.8912698030 2.8157387222', 'O 1.8912698030 -1.8912698030 2.8157387222', 'O 1.8912698030 1.8912698030 2.8157387222', 'O 0.0000000000 -1.8912698030 4.3955275562', 'O 0.0000000000 1.8912698030 4.3955275562', 'O -1.8912698030 -1.8912698030 -2.8157387228', 'O -1.8912698030 1.8912698030 -2.8157387228', 'O 1.8912698030 -1.8912698030 -2.8157387228', 'O 1.8912698030 1.8912698030 -2.8157387228', 'O -1.8912698030 0.0000000000 -4.3955275568', 'O 1.8912698030 0.0000000000 -4.3955275568', 'O 0.0000000000 0.0000000000 -1.9917721298', 'O 0.0000000000 -3.7825396060 1.9917721302', 'O 0.0000000000 3.7825396060 1.9917721302', 'O -3.7825396060 -1.8912698030 -5.2194941488', 'O -1.8912698030 -3.7825396060 -4.3955275568', 'O -1.8912698030 -1.8912698030 -6.7992829828', 'O 0.0000000000 -1.8912698030 -5.2194941488', 'O -3.7825396060 -1.8912698030 4.3955275562', 'O -1.8912698030 -3.7825396060 5.2194941492', 'O -1.8912698030 -1.8912698030 6.7992829832', 'O -1.8912698030 0.0000000000 5.2194941492', 'O -3.7825396060 1.8912698030 -5.2194941488', 'O -1.8912698030 1.8912698030 -6.7992829828', 'O -1.8912698030 3.7825396060 -4.3955275568', 'O 0.0000000000 1.8912698030 -5.2194941488', 'O -3.7825396060 1.8912698030 4.3955275562', 'O -1.8912698030 1.8912698030 6.7992829832', 'O -1.8912698030 3.7825396060 5.2194941492', 'O 1.8912698030 -3.7825396060 -4.3955275568', 'O 1.8912698030 -1.8912698030 -6.7992829828', 'O 3.7825396060 -1.8912698030 -5.2194941488', 'O 1.8912698030 -3.7825396060 5.2194941492', 'O 1.8912698030 -1.8912698030 6.7992829832', 'O 1.8912698030 0.0000000000 5.2194941492', 'O 3.7825396060 -1.8912698030 4.3955275562', 'O 1.8912698030 1.8912698030 -6.7992829828', 'O 1.8912698030 3.7825396060 -4.3955275568', 'O 3.7825396060 1.8912698030 -5.2194941488', 'O 1.8912698030 1.8912698030 6.7992829832', 'O 1.8912698030 3.7825396060 5.2194941492', 'O 3.7825396060 1.8912698030 4.3955275562', 'O -3.7825396060 0.0000000000 -1.9917721298', 'O 3.7825396060 0.0000000000 -1.9917721298', 'H 3.2987979595 2.7074108767 5.6937699773', 'H 2.7919819949 3.0243402422 6.0731871030', 'H -2.9942296175 3.7352120481 -6.0371396807', 'H -3.3111589830 3.2283960834 -6.4165568063', 'H -3.1289880900 -3.0700756895 5.6066579079', 'H -2.9617918644 -2.6616754295 6.1602991724', 'H -3.4672171994 -3.3541306416 -6.1263950571', 'H -2.8381714011 -3.6094774899 -6.3273014299', 'H -2.7919819949 3.0243402422 6.0731871030', 'H -3.2987979595 2.7074108767 5.6937699773', 'H 2.9942296175 3.7352120481 -6.0371396807', 'H 3.3111589830 3.2283960834 -6.4165568063', 'H 3.1803984587 -2.9177165529 -5.7830253538', 'H 2.5513526604 -3.1730634013 -5.9839317265', 'H 3.3541306418 -3.4672171993 6.1263950571', 'H 3.6094774901 -2.8381714009 6.3273014297', 'H -2.7748619468 0.8236588835 -7.0384422251', 'H -3.2301403199 0.4371876519 -6.6581455118', 'H -3.8059586000 -0.4371876519 -6.9915185835', 'H -4.2612369731 -0.8236588835 -6.6112218702', 'H -2.8095400884 3.9249294924 -1.5530389461', 'H -3.2945062248 3.4399633560 -1.3773127582', 'H -3.8819050943 3.7364479630 1.3773127583', 'H -4.3668712307 3.2514818266 1.5530389461', 'H -0.3009980212 4.5325396060 -1.0609247698', 'H -0.9598485141 4.5325396060 -1.3201158731', 'H 0.9598485141 5.1575396060 2.1440824655', 'H 0.3009980212 5.1575396060 1.8848913622', 'H 2.9783275735 3.7561420073 -1.8034862452', 'H 3.1257187396 3.6087508412 -1.1268654590', 'H 3.8819050943 3.7364479630 1.3773127583', 'H 4.3668712307 3.2514818266 1.5530389461', 'H -4.5325396060 -0.3009980212 -1.8848913622', 'H -4.5325396060 -0.9598485141 -2.1440824654', 'H -5.1575396060 0.9598485141 1.3201158728', 'H -5.1575396060 0.3009980212 1.0609247696', 'H -3.4429691664 -3.2915004144 -1.6367911677', 'H -3.9219236820 -2.8125458987 -1.8428715985', 'H -3.2544876370 -4.3638654203 1.8428715983', 'H -3.7334421526 -3.8849109047 1.6367911675', 'H -0.9598485141 -4.5325396060 2.1440824655', 'H -0.3009980212 -4.5325396060 1.8848913622', 'H 0.3009980212 -5.1575396060 -1.0609247698', 'H 0.9598485141 -5.1575396060 -1.3201158731', 'H -3.9117561788 0.8530068119 5.4442058766', 'H -3.5713331432 0.4078397234 5.8768748264', 'H -3.7725231801 -0.4078397234 6.6319251742', 'H -3.4321001444 -0.8530068119 7.0645941240', 'H -0.2764232677 3.6329244010 6.3216975909', 'H -0.9844232677 3.6329244010 6.3216975909', 'H 0.6304232677 3.1314879376 7.5549079109', 'H 0.6304232677 3.6748611001 7.1010250997', 'H 0.2764232677 3.7415446609 -5.6605403519', 'H 0.9844232677 3.7415446609 -5.6605403519', 'H -0.6304232677 3.3834164765 -7.1264698175', 'H -0.6304232677 3.8212068477 -6.5700494810', 'H 4.5325396060 0.3009980212 1.0609247696', 'H 4.5325396060 0.9598485141 1.3201158728', 'H 5.1575396060 -0.9598485141 -2.1440824654', 'H 5.1575396060 -0.3009980212 -1.8848913622', 'H 3.1257187397 -3.6087508411 2.0781417759', 'H 2.9783275734 -3.7561420073 1.4015209898', 'H 3.8849109047 -3.7334421526 -1.6367911677', 'H 4.3638654203 -3.2544876370 -1.8428715985', 'H 3.4052852145 -0.4371876519 -6.5118459471', 'H 3.8605635875 -0.8236588835 -6.1315492338', 'H 3.1314879376 0.6304232677 -7.5549079105', 'H 3.6748611001 0.6304232677 -7.1010250993', 'H -0.6304232677 -0.9844232677 -7.5492829828', 'H -0.6304232677 -0.2764232677 -7.5492829828', 'H 0.2764232677 0.6304232677 -8.1742829828', 'H 0.9844232677 0.6304232677 -8.1742829828', 'H 0.8530068119 -3.9117561788 -5.4442058770', 'H 0.4078397234 -3.5713331430 -5.8768748268', 'H -0.4078397234 -3.7725231799 -6.6319251744', 'H -0.8530068119 -3.4321001442 -7.0645941241', 'H 3.7415446610 -0.9844232677 5.6605403515', 'H 3.7415446610 -0.2764232677 5.6605403515', 'H 3.8212068478 0.6304232677 6.5700494808', 'H 3.3834164767 0.6304232677 7.1264698174', 'H -0.3801074671 -0.3801074671 7.5492829832', 'H -0.8807390682 -0.8807390682 7.5492829832', 'H 0.8807390682 0.8807390682 8.1742829832', 'H 0.3801074671 0.3801074671 8.1742829832', 'H 0.2764232677 -3.6329244010 6.3216975909', 'H 0.9844232677 -3.6329244010 6.3216975909', 'H -0.8236588835 -3.1755353324 7.5181148619', 'H -0.4371876519 -3.6308137054 7.1378181486']

energy_comb, _ = calculate_energy(optimised_xyz)
print("Combined optimised energy: ", energy_comb)

#######################################################

# H_opt_xyz =

# energy_H, _ = calculate_energy(H_opt_xyz)
# print("H optimised energy: ", energy_H)