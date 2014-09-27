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

sigma = pymc.Uniform("sigma", 0.5, 0.6, value=0.545)
epsilon = pymc.Uniform("epsilon", 10.0, 20.0, value=13.0)

temperature = 298.15 * u.kelvin
pressure = 101.325 * u.kilopascals

observed = 1.58436 * u.grams / u.milliliter
observed = observed / (u.grams / u.milliliter)
error = 0.02

atoms_per_dim = 7

traj, mmtop = blib.build_top(atoms_per_dim, sigma.value)


def calc_density(sigma, epsilon):
    simulation, system, x = blib.build(traj, mmtop, temperature, pressure, sigma, epsilon, nonbondedCutoff=1.2*u.nanometer, stderr_tolerance=error)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    mu = x[t:].mean()
    print("\nmu = %f, target = %f, z = %f" % (mu, observed, (mu - observed) / error))
    return mu

density = pymc.Deterministic(calc_density, "Calculates density", "density", dict(sigma=sigma, epsilon=epsilon), dtype='float')

measurement = pymc.Normal("observed_density", mu=density, tau=error ** -2., value=observed, observed=True)

variables = [density, measurement, sigma, epsilon]
model = pymc.Model(variables)
mcmc = pymc.MCMC(model, db='hdf5', dbname="./out.h5")

mcmc.sample(10000)
s = mcmc.trace("sigma")[:]
e = mcmc.trace("epsilon")[:]

import pymc
db = pymc.database.hdf5.load("./out.h5")
s = db.trace("sigma")[:]
e = db.trace("epsilon")[:]
