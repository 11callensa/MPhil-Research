import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def set_train_materials():
    """
        Sets the adsorbent and adsorber.

        :return: Hydrogen and the compound to be adsorbed to.
    """

    hydrogen = 'H2'

    # DONE compound = {'TiO2-A': 'mp-390'}
    # DONE compound = {'TiO2-R': 'mp-1041565'}
    # compound = {'Pt': 'mp-126'}
    # DONE compound = {'Ni': 'mp-23'}
    # DONE compound = {'Pd': 'mp-2'}
    # DONE compound = {'PdO': 'mp-1336'}
    # DONE compound = {'MgO': 'mp-1265'}
    # DONE compound = {'Al2O3': 'mp-1143'}
    # DONE compound = {'LaCrO3': 'mp-18841'}
    # DONE compound = {'LaNiO3': 'mp-1075921'}
    # DONE compound = {'LaFeO3': 'mp-552676'}
    # compound = {'LaCoO3': 'mp-573180'}
    # DONE compound = {'GaN': 'mp-830'}
    # compound = {'WC': 'mp-1894'}
    # DONE compound = {'TiC': 'mp-631'}
    # compound = {'Co2O4': 'mp-18748'}
    # DONE compound = {'ZnO': 'mp-1986'}
    # DONE compound = {'ZrO2': 'mp-2858'}
    # DONE compound = {'Y2O3': 'mp-673247'}
    # DONE compound =  {'Cr2O3': 'mp-776873'}
    # DONE compound = {'W': 'mp-91'}
    # DONE compound = {'CuCr2O4': 'mp-1103973'}
    # DONE compound = {'Au': 'mp-81'}
    # DONE compound = {'Cu': 'mp-30'}
    # DONE compound = {'Ag': 'mp-124'}
    # DONE compound = {'SiC':'mp-7631'}
    # DONE compound = {'Ru': 'mp-8639'}
    # DONE compound = {'Ir': 'mp-101'}
    # DONE compound = {'Rh': 'mp-74'}
    # DONE compound = {'Al': 'mp-134'}
    compound = {'Ta': 'mp-50'}
    # compound = {'Co': 'mp-102'}
    # DONE compound = {'Nb': 'mp-75'}
    # DONE compound = {'GaAs': 'mp-2534'}
    # compound = {'Re': 'mp-1186901'}
    # DONE compound = {'SnO2':'mp-856'}

    return hydrogen, compound


def set_test_materials():

    hydrogen = 'H2'

    # compound = {'NiO': 'mp-19009',
    #             'Fe': 'mp-13',
    #             'Mo': 'mp-129',
    #             'Si': 'mp-149',
    #             'In2O3': 'mp-22598',
    #             'SiO2': 'mp-7000'}

    compound = {'NiO': 'mp-19009'}

    return hydrogen, compound


def set_chemisorption_materials():

    hydrogen = 'H2'

    # --------- Simple Hydrides ------- #

    # compound_hydrides = {'LiH': ['mp-23703'], 'NaH': ['mp-23870'], 'CaH2': ['mp-23713'], 'TiH2': ['mp-24726'],
    #                      'ZrH2': ['mp-24286'], 'VH2': ['mp-24728']}

    # compound = {'LiH': ['mp-23703', [[0, 4], [0, 6], [0, 5], [4, 2], [4, 1], [6, 2], [6, 3], [5, 3], [5, 1], [7, 2], [7, 3], [7, 1]]]}

    # compound = {'CaH2': ['mp-23713', [[0, 11], [0, 6], [0, 9], [0, 5], [0, 8], [1, 7], [1, 10], [1, 8], [1, 4], [1, 9], [2, 5], [2, 9], [2, 7], [3, 8], [3, 6], [3, 4]]],
    #             'NaH': ['mp-23870', [[0, 4], [0, 6], [0, 5], [4, 2], [4, 1], [6, 2], [6, 3], [5, 3], [5, 1], [7, 2], [7, 3], [7, 1]]],
    #             'MgH2': ['mp-23711', [[[0, 5], [0, 3], [0, 2], [0, 4], [1, 4]]]]}

    # compound = {'LiH': ['mp-23703']}
    # compound = {'NaH': 'mp-23870'}
    # compound = {'CaH2': ['mp-23713']}
    # compound = {'TiH2': ['mp-24726']}
    # compound = {'ZrH2': 'mp-24286'}
    # compound = {'VH2': ['mp-24728']}

    # compound = {'MgH2': ['mp-23710']}                 # Version 1
    # compound = {'MgH2': ['mp-23710', [[0, 5], [0, 3], [0, 2], [0, 4], [1,4]]]}                 # Version 2 (Stable)

    # ---------- Complex Hydrides ----------- #

    # compound_complex = {'NaBH4': ['mp-38725'], 'LiAl4': ['mp-27653'], 'Mg(BH4)2': ['mp-1200811']}

    # compound = {'NaBH4': ['mp-38725']}
    # compound = {'LiAl4': ['mp-27653']}
    # compound = {'Mg(BH4)2': ['mp-1200811']}

    # ---------- Alanates ---------- #

    compound_alanates = {'NaAlH4': ['mp-23704'], 'LiAlH4': ['mp-27653'], 'KAlH4': ['mp-31097']}

    # compound = {'NaAlH4': ['mp-23704']}
    # compound = {'LiAlH4': ['mp-27653']}
    # compound = {'KAlH4': ['mp-31097']}

    # ---------- Intermetallic Compounds --------- #

    compound_intermetallic = {'LaNi5H6': ['mp-1222987'], 'TiFeH2': ['mp-1079106'], 'Mg2NiH3': ['mp-697331']}

    # compound = {'LaNi5H6': ['mp-1222987']}
    # compound = {'TiFeH2': ['mp-1079106']}
    # compound = {'Mg2NiH3': ['mp-697331']}

    # compounds = {'NaH': ['mp-23870'], 'CaH2': ['mp-23713'], 'TiH2': ['mp-24726'],'ZrH2': ['mp-24286'],
    #              'NaAlH4': ['mp-23704'], }

    return True
