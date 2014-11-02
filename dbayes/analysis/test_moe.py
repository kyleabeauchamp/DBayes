import moe
from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
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
indexed_data = data.set_index(keys + ["temperature"])

q0 = pymc.Uniform("q0", 0.4, 0.8)
sigma0 = pymc.Uniform("sigma0", 0.2, 0.3)

temperatures = [280, 300, 320]

measurements = np.array([1.000998, 1.043560, 1.084166])
relative_error = 0.001

def objective(q0_val, sigma0_val):
    variables = []
    
    q0.value = q0_val
    sigma0.value = sigma0_val
    print(q0.value)
    print(sigma0.value)
    for k, temperature in enumerate(temperatures):
        observed = measurements[k]
        predicted = indexed_data.density.ix[(q0_val, sigma0_val, temperature)]
        tau = (observed * relative_error) ** -2.
        var = pymc.Normal("obs_%d" % k, mu=predicted, tau=tau, observed=True, value=observed)
        print(predicted, observed, tau, var.logp)
        variables.append(var)
    
    model = pymc.MCMC(variables)
    return model.logp

a, b = data[keys].iloc[0].values
logp = objective(a, b)

get_bounds = lambda variable: (variable.parents["lower"], variable.parents["upper"])

experiment_bounds = [get_bounds(q0), get_bounds(sigma0)]
exp = Experiment(experiment_bounds)

for (q0_val, sigma0_val) in data.set_index(keys).index:
    value = objective(q0_val, sigma0_val)
    print(q0_val, sigma0_val, value)
    error = 0.001
    exp.historical_data.append_sample_points([[(q0_val, sigma0_val), value, error]])

next_point_to_sample = gp_next_points(exp, num_points_to_sample=2)

print next_point_to_sample
