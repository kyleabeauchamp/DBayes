import pymc
import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md


traj = md.load("./dipoles.pdb")
out_dir = "./symmetric_energies/"
q0 = pymc.Uniform("q0", 0.0, 1.0)
sigma0 = pymc.Uniform("sigma0", 0.08, 0.4)
epsilon0 = pymc.Uniform("epsilon0", 0.2, 2.0)
#sigma1 = pymc.Uniform("sigma0", 0.08, 0.4)
#epsilon1 = pymc.Uniform("epsilon0", 0.2, 2.0)
sigma1 = 1.0 * sigma0
epsilon1 = 1.0 * epsilon0
r0 = pymc.Uniform("r0", 0.05, 0.25, value=0.2, observed=True)

model = pymc.Model([q0, sigma0, epsilon0, sigma1, epsilon1, r0])

temperatures = [280 * u.kelvin, 290 * u.kelvin, 300 * u.kelvin, 310 * u.kelvin, 320 * u.kelvin]
pressure = 1.0 * u.atmospheres

data = []

for k in range(10000):
    model.draw_from_prior()
    for q0 in np.linspace(0, 1, 5):
        for temperature in temperatures:
            dipole = dipoles.Dipole(1000, q0=q0, sigma0=sigma0.value, epsilon0=epsilon0.value, sigma1=sigma1.value, epsilon1=epsilon1.value, r0=r0.value)
            print(dipole)
            try:
                values, mu, sigma = dipoles.simulate_density(dipole, traj, temperature, pressure, out_dir, print_frequency=100)
                data.append(dict(q0=q0, sigma0=sigma0.value, epsilon0=epsilon0.value, sigma1=sigma1.value, epsilon1=epsilon1.value, r0=r0.value, density=mu, density_error=sigma))
            except Exception as e:
                print(e)
