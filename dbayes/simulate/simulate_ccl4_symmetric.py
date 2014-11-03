import sys
import itertools
import os
import pymc
import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

n_molecules = 500
traj = md.load("./ccl4.pdb")

out_dir = os.path.join(os.getenv("HOME"), "dat", "dipoles-symmetric-grid")

q0 = pymc.Uniform("q0", 0.0, 1.0)
sigma1 = pymc.Uniform("sigma1", 0.3, 0.4)  # 0.347094140587

model = pymc.Model([q0, sigma1])

temperatures = [280 * u.kelvin, 300 * u.kelvin, 320 * u.kelvin]
pressure = 1.0 * u.atmospheres

model.draw_from_prior()
q0_grid = np.linspace(0.0, 1.0, 10)
sigma_grid = np.linspace(0.3, 0.4, 10)

product_grid = itertools.product(q0_grid, sigma_grid)
for k, (q0_val, sigma_val) in enumerate(product_grid):
    if k != int(sys.argv[1]):
        continue
    print(k, q0_val, sigma_val)
    q0.value = q0_val
    sigma0.value = sigma_val
    for temperature in temperatures:
        dipole = dipoles.Tetrahedron(n_molecules, q0=q0.value, sigma1=sigma1.value)
        try:
            t_new = md.load(dcd_filename, top=traj)[-1]  # Try to load traj from last temperature as starting box.
            dipole.traj = t_new
        except NameError:  # Otherwise use gaff box.
            dipole.traj = traj
        print(dipole)
        values, mu, sigma, dcd_filename = dipoles.simulate_density(dipole, temperature, pressure, out_dir)
