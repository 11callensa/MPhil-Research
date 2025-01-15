import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")

def set_materials():

    hydrogen = 'H2'

    # --------- Simple Hydrides ------- #

    # compound_hydrides = {'LiH': ['mp-23703'], 'NaH': ['mp-23870'], 'CaH2': ['mp-23713'], 'TiH2': ['mp-24726'],
    #                      'ZrH2': ['mp-24286'], 'VH2': ['mp-24728']}

    # compound = {'LiH': ['mp-23703', [[0, 4], [0, 6], [0, 5], [4, 2], [4, 1], [6, 2], [6, 3], [5, 3], [5, 1], [7, 2], [7, 3], [7, 1]]],
    #             'NaH': ['mp-23870', [[0, 4], [0, 6], [0, 5], [4, 2], [4, 1], [6, 2], [6, 3], [5, 3], [5, 1], [7, 2], [7, 3], [7, 1]]]}

    compound = {'LiH': ['mp-23703', [[0, 4], [0, 6], [0, 5], [4, 2], [4, 1], [6, 2], [6, 3], [5, 3], [5, 1], [7, 2], [7, 3], [7, 1]]]}

    # compound = {'LiH': ['mp-23703']}
    # compound = {'NaH': 'mp-23870'}
    # compound = {'CaH2': ['mp-23713']}
    # compound = {'TiH2': ['mp-24726']}
    # compound = {'ZrH2': 'mp-24286'}
    # compound = {'VH2': ['mp-24728']}

    # compound = {'MgH2': ['mp-23710']}                 # Version 1
    # compound = {'MgH2': ['mp-23711']}                 # Version 2


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

    return hydrogen, compound
