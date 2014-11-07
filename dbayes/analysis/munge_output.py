import mdtraj as md
import pymbar
import seaborn as sns
import scipy.interpolate
import pymc
import sklearn.gaussian_process
import os
import pandas as pd
import glob

kgas = 8.3144621E-3

#top_filename = "dipoles.pdb"
top_filename = "tetrahedron.pdb"

traj0 = md.load(top_filename)
n_atoms_per_molecule = traj0.n_atoms / traj0.n_residues

#filenames = glob.glob("/home/kyleb/dat/dipoles-symmetric-grid2/*0.csv")
filenames = glob.glob("/home/kyleb/dat/ccl4/*0.csv")
data = []
for filename in filenames:
    x = pd.read_csv(filename, skiprows=1, names=["energy", "density"])
    trj_filename = os.path.splitext(filename)[0] + ".dcd"
    gas_filename = os.path.splitext(filename)[0] + ".gas.csv"
    gas_energies = pd.read_csv(gas_filename, skiprows=1, names=["energy"]).energy
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
    parameters["gas_energy"] = gas_energies.mean()
    parameters["gas_energy_sigma"] = gas_energies.std() * len(gas_energies) ** -0.5
    parameters["energy"] = x["energy"].values[t0:].mean()
    parameters["energy_sigma"] = x["energy"].values[t0:].std() * Neff ** -0.5
    parameters["temperature"] = temperature
    parameters["evap"] = parameters["gas_energy"] - parameters["energy"] / traj0.n_residues + kgas * temperature
    q0 = parameters["q0"]
    charges = -1. * np.ones(n_atoms_per_molecule) / n_atoms_per_molecule
    charges[0] = q0
    box_charges = np.tile(charges, traj.n_residues)
    dielectric = md.geometry.static_dielectric(traj, box_charges, temperature)    
    parameters["dielectric"] = dielectric
    kappa = md.geometry.isothermal_compressability_kappa_T(traj, temperature)
    parameters["kappa"] = kappa
    alpha = md.geometry.thermal_expansion_alpha_P(traj, temperature, x.energy.values[0:traj.n_frames])
    parameters["alpha"] = alpha
    print(parameters)
    data.append(parameters)
    break
    
data = pd.DataFrame(data).dropna()
data.pivot_table(index=["q0", "sigma0"], columns=["temperature"], values=["density"])

#data.to_hdf('./symmetric-grid2.h5', 'data')
data.to_hdf('./ccl4-grid.h5', 'data')
