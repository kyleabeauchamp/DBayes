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
traj = md.load("./dipoles.pdb")

out_dir = os.path.join(os.getenv("HOME"), "dat", "dipoles-symmetric-grid2")

q0 = pymc.Uniform("q0", 0.4, 0.8)

sigma0 = pymc.Uniform("sigma0", 0.2, 0.3)
sigma1 = 1.0 * sigma0

epsilon0 = pymc.Uniform("epsilon0", 0.2, 1.0, value=0.5, observed=True)
epsilon1 = 1.0 * epsilon0

r0 = pymc.Uniform("r0", 0.05, 0.25, value=0.2, observed=True)

model = pymc.Model([q0, sigma0, epsilon0, sigma1, epsilon1, r0])

temperatures = [280 * u.kelvin, 300 * u.kelvin, 320 * u.kelvin]
pressure = 1.0 * u.atmospheres

q0_grid = np.linspace(q0.parents["lower"], q0.parents["upper"], 10)
sigma_grid = np.linspace(sigma0.parents["lower"], sigma0.parents["upper"], 10)
product_grid = itertools.product(q0_grid, sigma_grid)

for k, (q0_val, sigma_val) in enumerate(product_grid):
    if k != int(sys.argv[1]):
        pass#continue
    print(k, q0_val, sigma_val)
    q0.value = q0_val
    sigma0.value = sigma_val
    for temperature in temperatures:
        dipole = dipoles.Dipole(n_molecules, q0=q0.value, sigma0=sigma0.value, epsilon0=epsilon0.value, sigma1=sigma1.value, epsilon1=epsilon1.value, r0=r0.value)
        try:
            t_new = md.load(dcd_filename, top=traj)[-1]  # Try to load traj from last temperature as starting box.
            dipole.traj = t_new
        except NameError:  # Otherwise use gaff box.
            dipole.traj = traj
        print(dipole)
        #dipoles.simulate_density(dipole, temperature, pressure, out_dir)
        energy = dipole.gas_energy(out_dir, temperature)
