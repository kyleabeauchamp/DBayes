import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

#keys = ["q0", "sigma0"]
keys = ["q0", "sigma1"]
#data = pd.read_hdf("./symmetric-grid2.h5", 'data')
data = pd.read_hdf("./ccl4-grid.h5", 'data')

Q = data.pivot_table(index=keys, columns=["temperature"], values=["energy", "density", "dielectric", "kappa"])

temperature = 280.
ind = data.temperature == temperature
mydata = data[ind]

figure()
sns.interactplot(keys[0], keys[1], "density", mydata, cmap="coolwarm", filled=True, levels=25);

figure()
sns.interactplot(keys[0], keys[1], "evap", mydata, cmap="coolwarm", filled=True, levels=25);

figure()
sns.interactplot(keys[0], keys[1], "dielectric", mydata, cmap="coolwarm", filled=True, levels=25);


figure()
sns.interactplot(keys[0], keys[1], "kappa", mydata, cmap="coolwarm", filled=True, levels=25);

