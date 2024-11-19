for symbol, pos in zip(atoms, positions):
    xyz_format += f"{symbol} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n"