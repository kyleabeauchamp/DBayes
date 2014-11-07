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

n_molecules = 400
traj = md.load("./tetrahedron.pdb")

out_dir = os.path.join(os.getenv("HOME"), "dat", "ccl4")

q0 = pymc.Uniform("q0", 0.2, 1.0)
sigma1 = pymc.Uniform("sigma1", 0.325, 0.375)  # 0.347094140587

model = pymc.Model([q0, sigma1])

temperatures = [280 * u.kelvin, 300 * u.kelvin, 320 * u.kelvin]
pressure = 1.0 * u.atmospheres

q0_grid = np.linspace(q0.parents["lower"], q0.parents["upper"], 10)
sigma_grid = np.linspace(sigma1.parents["lower"], sigma1.parents["upper"], 10)

product_grid = itertools.product(q0_grid, sigma_grid)
for k, (q0_val, sigma_val) in enumerate(product_grid):
    #if k != int(sys.argv[1]):
    #    pass#continue
    print(k, q0_val, sigma_val)
    q0.value = q0_val
    sigma1.value = sigma_val
    for temperature in temperatures:
        dipole = dipoles.Tetrahedron(n_molecules, q0=q0.value, sigma1=sigma1.value)
        try:
            t_new = md.load(dcd_filename, top=traj)[-1]  # Try to load traj from last temperature as starting box.
            dipole.traj = t_new
        except NameError:  # Otherwise use gaff box.
            dipole.traj = traj
        print(dipole)
        dipoles.simulate_density(dipole, temperature, pressure, out_dir)
        energy = dipole.gas_energy(out_dir, temperature)
