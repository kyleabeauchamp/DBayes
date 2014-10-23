import pymc
import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md


q0 = pymc.Uniform("q0", 0.0, 1.)
sigma0 = pymc.Uniform("sigma0", 0.08, 0.5)
epsilon0 = pymc.Uniform("epsilon0", 0.1, 2.0)
sigma1 = pymc.Uniform("sigma0", 0.08, 0.5)
epsilon1 = pymc.Uniform("epsilon0", 0.1, 2.0)
r0 = pymc.Uniform("r0", 0.05, 0.5)

model = pymc.Model([q0, sigma0, epsilon0, sigma1, epsilon1, r0])
model.draw_from_prior()

dipole = dipoles.Dipole(1000)

temperature = 300 * u.kelvin
pressure = 1.0 * u.atmospheres

values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure, print_frequency=25)


data = []
for k in range(10):
    model.draw_from_prior()    
    values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure, print_frequency=25)
    data.append(dict(q0=q0.value, sigma0=sigma0.value, epsilon0=epsilon0.value, sigma1=sigma1.value, epsilon1=epsilon1.value, r0=r0.value, density=mu, density_error=sigma))
