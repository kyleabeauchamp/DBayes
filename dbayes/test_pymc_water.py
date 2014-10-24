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

n_steps = 200000
qH = pymc.Uniform("qH", 0.40, 0.5, value=0.417)
sigma = pymc.Uniform("sigma", 0.31, 0.325, value=0.3147)  # 0.31507524065751241
epsilon = pymc.Uniform("epsilon", 0.5, 1.0, value=0.635968, observed=True)
#sigmaH = pymc.Uniform("sigmaH", 0.06, 0.12, value=0.1004)
sigmaH = pymc.Uniform("sigmaH", 0.06, 0.13, value=0.089)
epsilonH = pymc.Uniform("epsilonH", 0.03, 0.15, value=0.06, observed=True)
theta = pymc.Uniform("theta", 1.7, 2.0, value=1.82421813418, observed=True)
r0 = pymc.Uniform("r0", 0.09, 0.11, value=0.09572, observed=True)

data = [dict(temperature=281.15 * u.kelvin, density=0.999848), dict(temperature=301.15 * u.kelvin, density=0.996234), dict(temperature=321.15 * u.kelvin, density=0.988927)]
data = pd.DataFrame(data)

pressure = 1.0 * u.atmospheres
#density_error = 0.001
density_error = 0.00075

def calc_density(qH, sigma, epsilon, sigmaH, epsilonH, r0, theta, temperature):
    x = water_lib.build(system, positions, mmtop, temperature * u.kelvin, pressure, qH, sigma, epsilon, sigmaH, epsilonH, r0, theta, stderr_tolerance=density_error, n_steps=n_steps)
    t, g, N_eff = pymbar.timeseries.detectEquilibration_fft(x)
    mu = x[t:].mean()
    print(temperature, mu)
    return mu

temperatures = [pymc.Uniform("temperature_%d" % i, 0.0, 1000.0, value=data.temperature[i] / u.kelvin, observed=True) for i in data.index]

density_estimators = [
pymc.Deterministic(lambda qH, sigma, epsilon, sigmaH, epsilonH, r0, theta, temperature: calc_density(qH, sigma, epsilon, sigmaH, epsilonH, r0, theta, temperature=temperature), 
"Calculates_density_%d" % i, "density_estimator_%d" % i, dict(qH=qH, sigma=sigma, epsilon=epsilon, sigmaH=sigmaH, epsilonH=epsilonH, r0=r0, theta=theta, temperature=temperatures[i]), dtype='float')
for i in data.index
]

measurements = [
pymc.Normal("observed_density_%d" % i, mu=density_estimators[i], tau=density_error ** -2., value=data.density[i], observed=True)
for i in data.index
]

pred = np.array([d.value for d in density_estimators])
expt = data.density.values
z = (pred - expt) / density_error
z
sum(z ** 2)
sum([q.logp for q in measurements])  # This is -1 * chi squared, up to an additive constant



variables = [qH, sigma, epsilon, sigmaH, epsilonH, theta, r0]
variables.extend(temperatures)
variables.extend(density_estimators)
variables.extend(measurements)

#map = pymc.MAP(variables)
#map.fit()

#mcmc = pymc.MCMC(variables, db='hdf5', dbname="./water.h5", verbose=3)
mcmc = pymc.MCMC(variables, verbose=3)

mcmc.use_step_method(pymc.Metropolis, qH, proposal_sd=0.001, proposal_distribution='Normal')
mcmc.use_step_method(pymc.Metropolis, sigma, proposal_sd=0.0005, proposal_distribution='Normal')
mcmc.use_step_method(pymc.Metropolis, sigmaH, proposal_sd=0.0005, proposal_distribution='Normal')

mcmc.isample(10000, tune_throughout=False)

