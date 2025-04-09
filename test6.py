def parse_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Skip the first two lines
    atoms_coordinates = lines[2:]

    result = []

    for line in atoms_coordinates:
        parts = line.split()
        atom = parts[0]
        x, y, z = parts[1], parts[2], parts[3]
        result.append(f"{atom} {x} {y} {z}")

    print(result)


def write_xyz_from_mixed_coords(input_string, output_filename):
    lines = [line.strip() for line in input_string.strip().splitlines() if line.strip()]
    atoms = []

    for line in lines:
        parts = line.split()
        try:
            idx = parts.index("AA")
            element = parts[1]
            x_angstrom = float(parts[idx - 3])
            y_angstrom = float(parts[idx - 2])
            z_angstrom = float(parts[idx - 1])
        except (ValueError, IndexError):
            print(f"⚠️ Skipping line due to parse error: {line}")
            continue

        atoms.append((element, x_angstrom, y_angstrom, z_angstrom))

    with open(output_filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Converted to XYZ format using Ångström coordinates\n")
        for atom in atoms:
            f.write(f"{atom[0]} {atom[1]:.6f} {atom[2]:.6f} {atom[3]:.6f}\n")

    print(f"✅ XYZ file written to '{output_filename}' with {len(atoms)} atoms.")



name = 'TiO2-A'
coords = 0
# write_xyz_from_mixed_coords(coords, f'Optimised Coordinates/{name}_optimised_coords_3.xyz')

# Example usage
file_path = 'Optimised Coordinates/TiO2-A_optimised_coords_3.xyz'  # Replace with your XYZ file path
parse_xyz(file_path)
