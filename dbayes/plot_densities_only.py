import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]

filenames = glob.glob("./simple_*/*.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1, names=["energy", "density"])
    if len(x) < 250:
        print(filename)
        continue
    mu = x.mean()
    a, b = os.path.splitext(os.path.split(filename)[-1])[0].split("_")
    temperature = float(b)
    chunks = a.split(",")
    parameters = {}
    for chunk in chunks:
        name, parm = chunk.split("=")
        parm = float(parm)
        parameters[name.lstrip()] = parm
    parameters["density"] = mu["density"]
    parameters["energy"] = mu["energy"]
    parameters["temperature"] = temperature
    data.append(parameters)
    
data = pd.DataFrame(data).dropna()
Q = data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["energy", "density"])
Q = data.pivot_table(index=["q0", "sigma0"], columns="temperature", values="energy")
dE0 = (Q[300.] - Q[280.])
dE0.name = "dE0"
dE0 = dE0.reset_index()
dE0.dropna(inplace=True)


temperature = 300.
ind = data.temperature == temperature
mydata = data[ind]

figure()
sns.interactplot("q0", "sigma0", "density", mydata, cmap="coolwarm", filled=True, levels=25);

figure()
sns.interactplot("q0", "sigma0", "dE0", dE0, cmap="coolwarm", filled=True, levels=25);
