import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def set_train_materials():
    """
        Sets the adsorbent and adsorber that will be added to the training file.

        :return: Hydrogen and the compound to be adsorbed to.
    """

    hydrogen = 'H2'

    # DONE compound = {'TiO2-A': 'mp-390'}
    # DONE compound = {'TiO2-R': 'mp-1041565'}
    # DONE compound = {'Ni': 'mp-23'}
    # DONE compound = {'Pd': 'mp-2'}
    # DONE compound = {'PdO': 'mp-1336'}
    # DONE compound = {'MgO': 'mp-1265'}
    # DONE compound = {'Al2O3': 'mp-1143'}
    # DONE compound = {'LaCrO3': 'mp-18841'}
    # DONE compound = {'LaNiO3': 'mp-1075921'}
    # DONE compound = {'LaFeO3': 'mp-552676'}
    # DONE compound = {'GaN': 'mp-830'}
    # DONE compound = {'WC': 'mp-13136'}
    # DONE compound = {'TiC': 'mp-631'}
    # DONE compound = {'ZnO': 'mp-1986'}
    # DONE compound = {'ZrO2': 'mp-2858'}
    # DONE compound = {'Y2O3': 'mp-673247'}
    # DONE compound =  {'Cr2O3': 'mp-776873'}
    # DONE compound = {'W': 'mp-91'}
    # DONE compound = {'CuCr2O4': 'mp-504573'}
    # DONE compound = {'Fe': 'mp-150'}
    # DONE compound = {'Au': 'mp-81'}
    # DONE compound = {'Cu': 'mp-30'}
    # DONE compound = {'Ag': 'mp-124'}
    # DONE compound = {'SiC':'mp-7631'}
    # DONE compound = {'Ru': 'mp-8639'}
    # DONE compound = {'Ir': 'mp-101'}
    # DONE compound = {'Rh': 'mp-74'}
    # DONE compound = {'Al': 'mp-134'}
    # DONE compound = {'Nb': 'mp-75'}
    # DONE compound = {'GaAs': 'mp-2534'}
    # DONE compound = {'Re': 'mp-8642'}
    # DONE compound = {'SnO2':'mp-856'}

    # PARTIAL compound = {'LaCoO3': 'mp-573180'}
    # PARTIAL compound = {'CoO': 'mp-19079'}
    # PARTIAL compound = {'Ta': 'mp-6986'}
    # PARTIAL compound = {'Co': 'mp-102'}

    return hydrogen, compound


def set_test_materials():
    """
            Sets the adsorbent and adsorber that will be added to the training file.

            :return: Hydrogen and the compound to be adsorbed to.
    """

    hydrogen = 'H2'

    # DONE compound = {'NiO': 'mp-19009'}
    # DONE compound = {'Mo': 'mp-8637'}
    # DONE compound = {'Si': 'mp-1491'}
    # DONE compound = {'In2O3': 'mp-22598'}
    # DONE compound = {'SiO2': 'mp-558891'}
    # DONE compound = {'Pt': 'mp-126'}
    # DONE compound = {'FeTi': 'mp-305'}

    return hydrogen, compound
