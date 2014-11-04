import mdtraj as md
import pymbar
import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

keys = ["q0", "sigma0"]

top_filename = "dipoles.pdb"
#top_filename = "tetrahedron.pdb"

filenames = glob.glob("/home/kyleb/dat/dipoles-symmetric-grid/*.csv")
#filenames = glob.glob("/home/kyleb/dat/ccl4/*.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1, names=["energy", "density"])
    trj_filename = os.path.splitext(filename)[0] + ".dcd"
    traj = md.load(trj_filename, top=top_filename)    
    density_ts = x["density"].values
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
    parameters["energy"] = x["energy"].values[t0:].mean()
    parameters["energy_sigma"] = x["energy"].values[t0:].std() * Neff ** -0.5
    parameters["temperature"] = temperature
    q0 = parameters["q0"]
    charges = np.array([q0, -q0])
    #charges = np.array([q0, -q0 / 4., -q0 / 4., -q0 / 4., -q0 / 4.])
    box_charges = np.tile(charges, traj.n_residues)
    dielectric = md.geometry.static_dielectric(traj, box_charges, temperature)    
    parameters["dielectric"] = dielectric
    print(parameters)
    data.append(parameters)
    
data = pd.DataFrame(data).dropna()
data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["density"])

data.to_hdf('./symmetric-grid.h5', 'data')
#data.to_hdf('./ccl4-grid.h5', 'data')
