import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]
data = pd.read_hdf("./symmetric.h5", 'data')

Q = data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["energy", "density"])
#Q = data.pivot_table(index=["q0", "sigma0"], columns="temperature", values="energy")
#dE0 = (Q[300.] - Q[280.])
#dE0.name = "dE0"
#dE0 = dE0.reset_index()
#dE0.dropna(inplace=True)


temperature = 280.
ind = data.temperature == temperature
mydata = data[ind]

figure()
sns.interactplot("q0", "sigma0", "density", mydata, cmap="coolwarm", filled=True, levels=25);

#figure()
#sns.interactplot("q0", "sigma0", "dE0", dE0, cmap="coolwarm", filled=True)
