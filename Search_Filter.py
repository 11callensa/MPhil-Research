import os

from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()


def filter(requirements):
    key = os.getenv("MATERIALS_KEY")

    with MPRester(key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=["mp-149", "mp-13", "mp-22526"]
        )

    materials = docs[0]

    return materials