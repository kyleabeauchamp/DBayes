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


traj = md.load("./monopole.pdb")
out_dir = os.path.join(os.getenv("HOME"), "dat", "monopoles")

q0 = pymc.Uniform("q0", 0.0, 1.0, value=0.25, observed=True)

sigma0 = pymc.Uniform("sigma0", 0.1, 0.6)
sigma1 = pymc.Uniform("sigma1", 0.1, 0.6)

epsilon0 = pymc.Uniform("epsilon0", 0.0, 2.0)
epsilon1 = pymc.Uniform("epsilon1", 0.0, 2.0)


model = pymc.Model([q0, sigma0, epsilon0, sigma1, epsilon1])

temperatures = [280 * u.kelvin, 290 * u.kelvin, 300 * u.kelvin, 310 * u.kelvin, 320 * u.kelvin]
pressure = 1.0 * u.atmospheres

model.draw_from_prior()
for temperature in temperatures:
    monopole = dipoles.Monopole(1000, q0=q0.value, sigma0=sigma0.value, epsilon0=epsilon0.value, sigma1=sigma1.value, epsilon1=epsilon1.value)
    traj = monopole.build_box()
    print(monopole)
    try:
        values, mu, sigma = dipoles.simulate_density(monopole, temperature, pressure, out_dir, langevin_tolerance=0.0001, print_frequency=100)
    except Exception as e:
        print(e)
