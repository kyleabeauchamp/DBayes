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

sigma = pymc.Uniform("sigma", 0.45, 0.65, value=0.5)
epsilon = pymc.Uniform("epsilon", 10.0, 20.0, value=13.0)

temperature = 298.15 * u.kelvin
pressure = 101.325 * u.kilopascals

observed = 1584.36 * u.kilograms / (u.meter ** 3.)
observed = observed / (u.grams / u.milliliter)

atoms_per_dim = 7

traj, mmtop = blib.build_top(atoms_per_dim, sigma.value)


def calc_density(sigma, epsilon):
    simulation, system, x = blib.build(traj, mmtop, temperature, pressure, sigma, epsilon)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    return x[t:].mean()

density = pymc.Deterministic(calc_density, "Calculates density", "density", dict(sigma=sigma, epsilon=epsilon), dtype='float')

measurement = pymc.Normal("observed_density", mu=density, tau=10.0, value=observed, observed=True)

variables = [density, measurement, sigma, epsilon]
model = pymc.Model(variables)
mcmc = pymc.MCMC(model)

mcmc.sample(10000)
s = mcmc.trace("sigma")[:]
e = mcmc.trace("epsilon")[:]

