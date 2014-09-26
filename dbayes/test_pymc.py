import pymbar
import blib
import pymc
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

mass = 12.01078 * u.daltons + 4 * 35.4532 * u.daltons

sigma = pymc.Uniform("sigma", 0.325, 0.375, value=0.35)
epsilon = pymc.Uniform("epsilon", 18.0, 23.0, value=20.0)

atoms_per_dim = 7

measurements = [dict(temperature=298.15 * u.kelvin, pressure=101.325 * u.kilopascals, density=1584.36 * u.kilograms / (u.meter ** 3.))]

ff = app.ForceField("./test.xml")

traj, mmtop = blib.build_top(atoms_per_dim, sigma.value)

@pymc.deterministic
def density(sigma=sigma, epsilon=epsilon):
    print(sigma, epsilon)
    simulation, system, x = blib.build(traj, mmtop, measurements[0]["temperature"], measurements[0]["pressure"], sigma, epsilon)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    return x[t:].mean()

measurement = pymc.Normal("observed_density", mu=density, tau=1.0, value=measurements[0]["density"] / (u.grams / u.milliliter), observed=True)

variables = [density, measurement, sigma, epsilon]
model = pymc.Model(variables)
mcmc = pymc.MCMC(model)

mcmc.sample(15)
mcmc.trace("sigma")[:]
mcmc.trace("epsilon")[:]
