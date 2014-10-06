import pymbar
import water_lib
import pymc
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

traj, mmtop, system, box, positions = water_lib.build_top()

qH = pymc.Uniform("qH", 0.0, 0.5, value=0.417)
sigma = pymc.Uniform("sigma", 0.3, 0.33, value=0.3151)
epsilon = pymc.Uniform("epsilon", 0.5, 1.0, value=0.6359)
theta = pymc.Uniform("theta", 1.7, 2.0, value=1.8242)
r0 = pymc.Uniform("r0", 0.09, 0.11, value=0.09572)

data = [dict(temperature=281.15 * u.kelvin, density=0.999848), dict(temperature=301.15 * u.kelvin, density=0.996234), dict(temperature=321.15 * u.kelvin, density=0.988927)]
data = pd.DataFrame(data)

pressure = 1.0 * u.atmospheres
density_error = 0.1

#x = water_lib.build(system, positions, mmtop, 300.0 * u.kelvin, pressure, qH.value, sigma.value, epsilon.value, r0.value, theta.value)

def calc_density(qH, sigma, epsilon, r0, theta, temperature):
    print(qH, sigma, epsilon, r0, theta, temperature)
    x = water_lib.build(system, positions, mmtop, temperature, pressure, qH, sigma, epsilon, r0, theta)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    mu = x[t:].mean()
    return mu

density_estimators = [
pymc.Deterministic(lambda qH, sigma, epsilon, r0, theta: calc_density(qH, sigma, epsilon, r0, theta, temperature=t), "Calculates density %d" % i, "density_estimator %d" % i, dict(qH=qH, sigma=sigma, epsilon=epsilon, r0=r0, theta=theta), dtype='float')
for i, d, t in data.itertuples()
]

measurements = [
pymc.Normal("observed_density %d" % i, mu=density_estimators[i], tau=density_error ** -2., value=d, observed=True)
for i, d, t in data.itertuples()
]

variables = [qH, sigma, epsilon, theta, r0]
variables.extend(density_estimators)
variables.extend(measurements)

model = pymc.Model(variables)
mcmc = pymc.MCMC(model, db='hdf5', dbname="./water.h5")

mcmc.sample(10000)


x = water_lib.build(system, positions, mmtop, temperature, pressure, qH, sigma, epsilon, r0, theta)
