import os

from mp_api.client import MPRester
from dotenv import load_dotenv

import Setup
import Fitness_Define
import Extractor

# young_mod = input("What Young Modulus is required?: ")

requirements = [2.2 * 10**9, 6]                                                                                         # Typical young modulus and thermal conductivity of NaCl
weights = [0.5, 0.5]

element_ids, compound_ids, extract_properties, headers = Setup.initialise_variables()

element_database = Extractor.extraction(element_ids, extract_properties, headers)
compound_database = Extractor.extraction(compound_ids, extract_properties, headers)


# fitness_value = Fitness_Define.fitness_function(properties, requirements, weights)
