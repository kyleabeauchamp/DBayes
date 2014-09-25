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

sigma = pymc.Uniform("sigma", 0.05, 1.0)
epsilon = pymc.Uniform("sigma", 0.0, 100.0)

sigma.value = 0.35
epsilon.value = 20.0

measurements = [dict(temperature=298.15 * u.kelvin, pressure=101.325 * u.kilopascals, density=1584.36 * u.kilograms / (u.meter ** 3.))]
#measurements.append(dict(temperature=313.15 * u.kelvin, pressure=101.325 * u.kilopascals, density=1554.96 * u.kilograms / (u.meter ** 3.)))

ff = app.ForceField("./test.xml")

traj, mmtop = blib.build_top()

#system, x = blib.build(traj, mmtop, measurements[0]["temperature"], measurements[0]["pressure"], sigma, epsilon)
#t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)

#x[t:].mean()
#x[t:].std() / N_eff ** 0.5


@pymc.deterministic
def density(sigma=sigma, epsilon=epsilon):
    system, x = blib.build(traj, mmtop, measurements[0]["temperature"], measurements[0]["pressure"], sigma, epsilon)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    return x[t:].mean()
