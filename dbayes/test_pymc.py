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

sigma = pymc.Uniform("sigma", 0.1, 0.7)
epsilon = pymc.Uniform("epsilon", 1.0, 40.0)

sigma.value = 0.35
epsilon.value = 20.0
atoms_per_dim = 7

measurements = [dict(temperature=298.15 * u.kelvin, pressure=101.325 * u.kilopascals, density=1584.36 * u.kilograms / (u.meter ** 3.))]

ff = app.ForceField("./test.xml")

#xyz = blib.build_box(atoms_per_dim, sigma.value)
traj, mmtop = blib.build_top(atoms_per_dim, sigma.value)

simulation, system, x = blib.build(traj, mmtop, measurements[0]["temperature"], measurements[0]["pressure"], sigma.value, epsilon.value)
#t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)

#x[t:].mean()
#x[t:].std() / N_eff ** 0.5


@pymc.deterministic
def density(sigma=sigma, epsilon=epsilon):
    print(sigma, epsilon)
    system, x = blib.build(traj, mmtop, measurements[0]["temperature"], measurements[0]["pressure"], sigma, epsilon)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    return x[t:].mean()


@pymc.potential
def error(density=density):
    mu = measurements[0]["density"] / (u.grams / u.milliliter)
    sigma = 0.05
    return -((mu - density) / sigma) ** 2.

variables = [density, error, sigma, epsilon]
