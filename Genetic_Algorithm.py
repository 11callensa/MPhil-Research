import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def set_materials():

    hydrogen = 'H2'
    compound = {'LiH': ['mp-23703']}

    # compound = {'LaNi5H6': ['mp-1222987']}

    # compound = {'MgH2': ['mp-23710']}

    compound_name = list(compound.keys())[0]
    compound_ID = compound[compound_name]

    return hydrogen, compound_name, compound_ID
