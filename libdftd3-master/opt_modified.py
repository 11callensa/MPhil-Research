
import numpy as np
import ctypes

from Compound_Properties import get_spin
from collections import Counter

from pyscf import lib
from pyscf import gto, scf, dft
from pyscf.dft import libxc
from pyscf.geomopt import berny_solver

import os


_loaderpath = 'libdftd3-master/lib'
libdftd3 = np.ctypeslib.load_library('libdftd3.so', _loaderpath)


atoms1 = ['Li 0.0000000000 0.0000000000 0.0000000000',
          'Li 0.0000000000 -2.0086193919 -2.0086193919',
          'Li 2.0086193919 -2.0086193919 0.0000000000',
          'Li 2.0086193919 0.0000000000 -2.0086193919',
          'H 0.0000000000 -2.0086193919 0.0000000000',
          'H 0.0000000000 0.0000000000 -2.0086193919',
          'H 2.0086193919 0.0000000000 0.0000000000',
          'H 2.0086193919 -2.0086193919 -2.0086193919',
          'H 0.12577068745972128 -0.10080110493304462 4.0']

name = 'Li4H4_Retry'

# atoms1 = ['Ti 1.8912698030 -4.8075108530 -3.7825396060', 'Ti 1.8912698030 -7.2112662790 -5.6738094090',
#           'Ti 1.8912698030 -7.2112662790 -1.8912698030', 'Ti 0.0000000000 0.0000000000 -5.6738094090',
#           'Ti 0.0000000000 -9.6150217060 -5.6738094090', 'Ti 0.0000000000 0.0000000000 -1.8912698030',
#           'Ti 0.0000000000 -9.6150217060 -1.8912698030', 'Ti 3.7825396060 0.0000000000 -5.6738094090',
#           'Ti 3.7825396060 -9.6150217060 -5.6738094090', 'Ti 3.7825396060 0.0000000000 -1.8912698030',
#           'Ti 3.7825396060 -9.6150217060 -1.8912698030', 'Ti 0.0000000000 -2.4037554260 -3.7825396060',
#           'Ti 3.7825396060 -2.4037554260 -3.7825396060', 'O 0.0000000000 -4.3955275570 -3.7825396060',
#           'O 3.7825396060 -4.3955275570 -3.7825396060', 'O 1.8912698030 -6.7992829830 -3.7825396060',
#           'O 1.8912698030 -5.2194941490 -5.6738094090', 'O 1.8912698030 -5.2194941490 -1.8912698030',
#           'O 0.0000000000 -7.6232495750 -5.6738094090', 'O 0.0000000000 -7.6232495750 -1.8912698030',
#           'O 3.7825396060 -7.6232495750 -5.6738094090', 'O 3.7825396060 -7.6232495750 -1.8912698030',
#           'O 1.8912698030 -9.2030384090 -5.6738094090', 'O 1.8912698030 -9.2030384090 -1.8912698030',
#           'O 0.0000000000 -1.9917721300 -5.6738094090', 'O 0.0000000000 -1.9917721300 -1.8912698030',
#           'O 3.7825396060 -1.9917721300 -5.6738094090', 'O 3.7825396060 -1.9917721300 -1.8912698030',
#           'O 0.0000000000 -0.4119832960 -3.7825396060', 'O 3.7825396060 -0.4119832960 -3.7825396060',
#           'O 1.8912698030 -2.8157387230 -3.7825396060', 'O 1.8912698030 -6.7992829830 -7.5650792120',
#           'O 1.8912698030 -6.7992829830 0.0000000000', 'O -1.8912698030 0.4119832960 -5.6738094090',
#           'O 0.0000000000 -0.4119832960 -7.5650792120', 'O 0.0000000000 1.9917721300 -5.6738094090',
#           'O 1.8912698030 0.4119832960 -5.6738094090', 'O -1.8912698030 -9.2030384090 -5.6738094090',
#           'O 0.0000000000 -10.0270050020 -7.5650792120', 'O 0.0000000000 -11.6067938360 -5.6738094090',
#           'O 0.0000000000 -10.0270050020 -3.7825396060', 'O -1.8912698030 0.4119832960 -1.8912698030',
#           'O 0.0000000000 1.9917721300 -1.8912698030', 'O 0.0000000000 -0.4119832960 0.0000000000',
#           'O 1.8912698030 0.4119832960 -1.8912698030', 'O -1.8912698030 -9.2030384090 -1.8912698030',
#           'O 0.0000000000 -11.6067938360 -1.8912698030', 'O 0.0000000000 -10.0270050020 0.0000000000',
#           'O 3.7825396060 -0.4119832960 -7.5650792120', 'O 3.7825396060 1.9917721300 -5.6738094090',
#           'O 5.6738094090 0.4119832960 -5.6738094090', 'O 3.7825396060 -10.0270050020 -7.5650792120',
#           'O 3.7825396060 -11.6067938360 -5.6738094090', 'O 3.7825396060 -10.0270050020 -3.7825396060',
#           'O 5.6738094090 -9.2030384090 -5.6738094090', 'O 3.7825396060 1.9917721300 -1.8912698030',
#           'O 3.7825396060 -0.4119832960 0.0000000000', 'O 5.6738094090 0.4119832960 -1.8912698030',
#           'O 3.7825396060 -11.6067938360 -1.8912698030', 'O 3.7825396060 -10.0270050020 0.0000000000',
#           'O 5.6738094090 -9.2030384090 -1.8912698030', 'O -1.8912698030 -2.8157387230 -3.7825396060',
#           'O 5.6738094090 -2.8157387230 -3.7825396060']

# name = 'TiO2'

symbols = [line.split()[0] for line in atoms1]
element_count = [(atom, str(count)) for atom, count in Counter(symbols).items()]

mol = gto.M(                                                                                                        # Define the molecule in PySCF
        verbose=5,
        atom=atoms1,
        basis='def2-svp',
        unit='Angstrom',
        spin=get_spin(element_count),
        symmetry=0,
    )

mf_grad_scan = dft.RKS(mol).nuc_grad_method().as_scanner()                                                              # Sets up restricted Kohn-Sham methods, nuclear grad adds a force calculation and as.scanner finds the lowest energy structure
mf_grad_scan.base = scf.addons.remove_linear_dep_(mf_grad_scan.base)
mf_grad_scan.base.level_shift = 0.5  # Increase from 0.2 if necessary
# Or change Berny specific settings for step size
mf_grad_scan.base.verbose = 4

##################### LEVERS FOR PERFORMANCE ########################

mf_grad_scan.base.grids.level = 3  # Reduce grid size
mf_grad_scan.base.damp = 0.3
# mf_grad_scan.base.grids.atom_grid = {"Ti": (40, 90), "O": (30, 75)}  # Custom grid
# mf_grad_scan.base.grids.prune = dft.gen_grid.treutler_prune  # Less dense pruning
mf_grad_scan.base.grids.prune = dft.gen_grid.nwchem_prune

mf_grad_scan.base.conv_tol = 1e-5  # Looser SCF convergence
mf_grad_scan.base.conv_tol_grad = 1e-4  # Looser force convergence

# mf_grad_scan.base.max_cycle = 30  # Limit SCF iterations
# mf_grad_scan.base.diis_space = 6  # Reduce DIIS memory

os.environ["OMP_NUM_THREADS"] = "4"  # Use 4 CPU cores (adjust as needed)

########################################################################

mf_grad_scan.base.xc = 'pbe0'
mf_grad_scan.grid_response = False


def mf_grad_with_dftd3(geom):

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
    grad = np.zeros((mol.natm,3))
    libdftd3.wrapper(ctypes.c_int(mol.natm),
             coords.ctypes.data_as(ctypes.c_void_p),
             itype.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_char_p(func),
             ctypes.c_int(version),
             ctypes.c_int(tz),
             edisp.ctypes.data_as(ctypes.c_void_p),
             grad.ctypes.data_as(ctypes.c_void_p))
    lib.logger.info(mf_grad_scan,"* Disp Energy [au]: %12.8f" % edisp[0])
    lib.logger.info(mf_grad_scan,"* Disp Gradients [au]:")
    atmlst = range(mol.natm)
    for k, ia in enumerate(atmlst):
        symb = mol.atom_pure_symbol(ia)
        lib.logger.info(mf_grad_scan,"* %d %s %12.8f %12.8f %12.8f" \
        % (ia, symb, grad[k,0], grad[k,1], grad[k,2]))
    e_tot += edisp[0]
    g_rhf += grad
    lib.logger.info(mf_grad_scan,"* Total Energy [au]: %12.8f" % e_tot)
    lib.logger.info(mf_grad_scan,"* Total Gradients [au]:")
    atmlst = range(mol.natm)
    for k, ia in enumerate(atmlst):
        symb = mol.atom_pure_symbol(ia)
        lib.logger.info(mf_grad_scan,"* %d %s %12.8f %12.8f %12.8f" \
        % (ia, symb, g_rhf[k,0], g_rhf[k,1], g_rhf[k,2]))
    return e_tot, g_rhf


mf = berny_solver.as_pyscf_method(mol, mf_grad_with_dftd3)
_, mol = berny_solver.kernel(mf, maxsteps=40)

xyzfile = name + '_d3_opt.xyz'
fspt = open(xyzfile,'w')
coords = mol.atom_coords()*lib.param.BOHR
fspt.write('%d \n' % mol.natm)
fspt.write('%d %d\n' % (mol.charge, (mol.spin+1)))

for ia in range(mol.natm):
    symb = mol.atom_pure_symbol(ia)
    fspt.write('%s  %12.6f  %12.6f  %12.6f\n' % (symb, \
    coords[ia][0],coords[ia][1], coords[ia][2]))
