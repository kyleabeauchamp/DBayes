import pymbar
import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]
data = pd.read_hdf("./symmetric.h5", 'data')


models = {}
for temperature in data.temperature.unique():
    ind = data.temperature == temperature
    X = data[ind][keys].values
    y = data[ind]["density"].values
    model = sklearn.gaussian_process.GaussianProcess(theta0=1.0, nugget=data[ind]["density_sigma"].values)
    model.fit(X, y)
    models[temperature] = model
    #interp = scipy.interpolate.interp2d(X[:, 0], X[:, 1], y, bounds_error=True)
    #models[temperature] = interp

predict = lambda q0, sigma0, temperature: np.array(models[temperature].predict([q0, sigma0], eval_MSE=True))[:, 0]
#predict = lambda q0, sigma0, temperature: np.array(models[temperature](q0, sigma0))
predict(sigma0=0.25, q0=0.5, temperature=300)

q0 = pymc.Uniform("q0", 0.4, 0.8)
sigma0 = pymc.Uniform("sigma0", 0.2, 0.3)


temperatures = [280, 300, 320]
#temperatures = [300]
@pymc.deterministic
def predictions(q0=q0, sigma0=sigma0):
    try:
        return [predict(q0=q0, sigma0=sigma0, temperature=t)[0] for t in temperatures]
    except ValueError:
        return [100.] * len(temperatures)


#values = np.array([1.043560])
values = np.array([1.000998, 1.043560, 1.084166])
#0.577386 0.255586  1.000998  1.043560  1.084166
#relative_error = 0.001
relative_error = 0.002
density_error = values * relative_error

q0.value = 0.577386
sigma0.value = 0.255586

experiments = pymc.Normal("observed_density", mu=predictions, tau=density_error ** -2., value=values, observed=True)

variables = [q0, sigma0, predictions, experiments]
mcmc = pymc.MCMC(variables)
mcmc.sample(500000, thin=100, burn=5000)

q = mcmc.trace("q0")[:]
s = mcmc.trace("sigma0")[:]
p = mcmc.trace("predictions")[:]

plot(q)
plot(s)
sns.jointplot(q, s, kind="kde")
