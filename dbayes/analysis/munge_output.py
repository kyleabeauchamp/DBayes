import pymbar
import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]

filenames = glob.glob("/home/kyleb/dat/dipoles-symmetric/*.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1, names=["energy", "density"])
    density_ts = x["density"]
    [t0, g, Neff] = pymbar.timeseries.detectEquilibration(density_ts)
    density_ts = density_ts[t0:]
    mu = density_ts.mean()
    sigma = density_ts.std() * Neff ** -0.5
    a, b = os.path.splitext(os.path.split(filename)[-1])[0].split("_")
    temperature = float(b)
    chunks = a.split(",")
    parameters = {}
    for chunk in chunks:
        name, parm = chunk.split("=")
        parm = float(parm)
        parameters[name.lstrip()] = parm
    parameters["density"] = mu
    parameters["density_sigma"] = sigma
    parameters["temperature"] = temperature
    print(parameters)
    data.append(parameters)
    
data = pd.DataFrame(data).dropna()
data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["density"])

data.to_hdf('./symmetric.h5', 'data')
