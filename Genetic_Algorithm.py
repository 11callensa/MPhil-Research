import os

from dotenv import load_dotenv

load_dotenv()

key = os.getenv("MATERIALS_KEY")


def set_materials():

    hydrogen = 'H2'
    # compound = {'LiH': ['mp-23703']}

    # compound = {'LaNi5H6': ['mp-1222987']}

    # compound = {'MgH2': ['mp-23710']}

    compound = {'NaBH4': ['mp-38725']}
    # compound = {'NaAlH4': ['mp-23704']}

    compound_name = list(compound.keys())[0]
    compound_ID = compound[compound_name]

    return hydrogen, compound_name, compound_ID


def set_PT():

    temperature = 30
    pressure = 101

    return temperature, pressure


def calculate_spacing(temp_celsius, pressure_kpa):
    # Constants
    k_B = 1.38e-23  # Boltzmann constant in J/K
    conversion_factor = 10 ** 10  # To convert meters to angstroms

    # Convert temperature to Kelvin
    temp_kelvin = temp_celsius + 273.15

    print("Temp kelvin: ", temp_kelvin)

    # Convert pressure to Pascals
    pressure_pa = pressure_kpa * 1e3

    print("Pressure Pa: ", pressure_pa)

    # Calculate intermolecular spacing in meters
    d_meters = (k_B * temp_kelvin / pressure_pa) ** (1/3)

    # Convert spacing to angstroms
    d_angstroms = d_meters * conversion_factor

    print("D Angstroms: ", d_angstroms)

    return d_angstroms
