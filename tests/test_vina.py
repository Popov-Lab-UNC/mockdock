from vina import Vina

# 1. Setup Vina with AD4
v = Vina(sf_name='ad4', cpu=0)

# 2. Load the maps (AutoGrid4 output)
# This replaces set_receptor() and compute_vina_maps()
v.load_maps('2R0U/rec_2r0u')

# 3. Load the ligand
v.set_ligand_from_file('2R0U/lig_0_s0.pdbqt')

# 4. Score or Dock
energy = v.score()
print(f'Initial Score (AD4): {energy[0]:.3f} kcal/mol')

# Docking
v.dock(exhaustiveness=32, n_poses=10)
v.write_poses('results_ad4.pdbqt', n_poses=5, overwrite=True)