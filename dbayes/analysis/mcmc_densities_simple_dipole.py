import pymbar
import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]
observables = ["density", "energy", "dielectric"]
data = pd.read_hdf("./symmetric-grid.h5", 'data')

models = {}

for temperature in data.temperature.unique():
    ind = data.temperature == temperature
    for observable in observables:    
        X = data[ind][keys].values
        y = data[ind][observable].values
        model = scipy.interpolate.interp2d(X[:, 0], X[:, 1], y, bounds_error=True)
        models[observable, temperature] = model


models["density", 280.](0.6, 0.25)

q0 = pymc.Uniform("q0", 0.4, 0.8)
sigma0 = pymc.Uniform("sigma0", 0.2, 0.3)

"""
data.pivot_table(index=keys, columns=["temperature"], values=["energy", "density", "dielectric"]).iloc[44]
            temperature
energy      280           -58857.570163
            300           -58709.308689
            320           -59226.840865
density     280                1.321589
            300                1.317470
            320                1.339186
dielectric  280               44.281327
            300               29.744798
            320               29.375216
Name: (0.577778, 0.244444), dtype: float64

"""

measurements = []
measurements.append(dict(temperature=280, observable="density", value=1.321589, error=0.001))
measurements.append(dict(temperature=280, observable="density", value=1.339186, error=0.001))
measurements = pd.DataFrame(measurements)

@pymc.deterministic
def predictions(q0=q0, sigma0=sigma0):
    values = np.zeros(measurements.shape[0])
    for i, row in measurements.iterrows():
        values[i] = models[row.observable, row.temperature](q0, sigma0)
    return values

experiments = pymc.Normal("observed_density", mu=predictions, tau=measurements.error.values ** -2., value=measurements.value.values, observed=True)

variables = [q0, sigma0, predictions, experiments]
mcmc = pymc.MCMC(variables)
mcmc.sample(500000, thin=100, burn=5000)

q = mcmc.trace("q0")[:]
s = mcmc.trace("sigma0")[:]
p = mcmc.trace("predictions")[:]

plot(q)
plot(s)
sns.jointplot(q, s, kind="kde")
