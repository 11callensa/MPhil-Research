import os

from mp_api.client import MPRester
from dotenv import load_dotenv

import Fitness_Define
import Element_Finder
import Search_Filter

# young_mod = input("What Young Modulus is required?: ")

requirements = [2.2 * 10**9, 6]                                                                                         # Typical young modulus and thermal conductivity of NaCl
weights = [0.5, 0.5]

constraints = Search_Filter.filter(requirements)

properties = Element_Finder.material_search(constraints)




fitness_value = Fitness_Define.fitness_function(properties, requirements, weights)

print(requirements)
