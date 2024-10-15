import os
import pandas as pd

from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def extraction(materials_ids, properties, headers):

    with MPRester(key) as mpr:

        docs = mpr.materials.summary.search(material_ids=materials_ids, fields=properties)

    database = pd.DataFrame(columns=headers)

    for n in range(len(docs)):

        database.loc[n, "Formula"] = docs[n].formula_pretty
        database.loc[n, "Density"] = docs[n].density
        database.loc[n, "Metal"] = docs[n].is_metal
        database.loc[n, "Magnetic"] = docs[n].is_magnetic

    print(database)
    input("Pause: ")

    return docs