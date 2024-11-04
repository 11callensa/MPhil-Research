from pyscf import gto, dft
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole

from collections import Counter

IPythonConsole.drawOptions.addAtomIndices = True


def create_molecule(smiles, bond_lengths=None, add_explicit_h=True, filename=None):
    """
    Creates a molecule from a SMILES string, optionally adding explicit hydrogens and
    setting specified bond lengths, then exports it to an XYZ file.

    Parameters:
    - smiles (str): The SMILES representation of the molecule.
    - bond_lengths (list of floats): Bond lengths to set (must match the number of bonds).
    - add_explicit_h (bool): Whether to add explicit hydrogens to the molecule.
    - filename (str): The filename for the XYZ output.

    Returns:
    - mol: The generated RDKit molecule object.
    """

    mol = Chem.MolFromSmiles(smiles)

    if add_explicit_h:
        mol = Chem.AddHs(mol)                                                                                           # Add explicit hydrogens if required

    Chem.AllChem.EmbedMolecule(mol)

    if bond_lengths:
        for i in range(len(bond_lengths)):
            if mol.GetNumAtoms() > i + 1:  # Ensure we have enough atoms
                Chem.AllChem.SetBondLength(mol.GetConformer(0), i, i + 1, bond_lengths[i])

    if filename:
        Chem.MolToXYZFile(mol, filename)

    return mol


def calculate_energy(atom):

    mol = gto.M(
            verbose=0,
            atom=atom,
            basis='sto-3g',
            unit='Angstrom',
        )

    mf = dft.RKS(mol)                                                                                                   # Run DFT calculation
    mf.xc = 'b3lyp'                                                                                                     # Choose an exchange-correlation functional
    energy = mf.kernel()

    return 27.2114*energy
