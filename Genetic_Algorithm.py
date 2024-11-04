import os
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def set_materials():

    hydrogen = 'H2'
    compound = {'LiH': ['mp-23703']}

    compound_name = list(compound.keys())[0]
    compound_ID = compound[compound_name]

    return hydrogen, compound_name, compound_ID
