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

filenames = glob.glob("/home/kyleb/dat/dipoles-symmetric-grid/*.dcd")


filename = filenames[0]
#filename = "/home/kyleb/dat/ccl4/q0=0.200000, sigma0=0.300000, sigma1=0.325000, epsilon0=0.500000, epsilon1=0.500000, r0=0.200000_280.000000.dcd"
filename = "/home/kyleb/dat/ccl4/q0=0.555556, sigma0=0.300000, sigma1=0.347222, epsilon0=0.500000, epsilon1=0.500000, r0=0.200000_280.000000.dcd"
#top_filename = "./dipoles.pdb"
top_filename = "tetrahedron.pdb"

a, b = os.path.splitext(os.path.split(filename)[-1])[0].split("_")
temperature = float(b)
chunks = a.split(",")
parameters = {}
for chunk in chunks:
    name, parm = chunk.split("=")
    parm = float(parm)
    parameters[name.lstrip()] = parm

q0 = parameters["q0"]
#charges = np.array([q0, -q0])
charges = np.array([q0, -q0 / 4., -q0 / 4., -q0 / 4., -q0 / 4.])

traj = md.load(filename, top=top_filename)
box_charges = np.tile(charges, traj.n_residues)

epsilon = md.geometry.static_dielectric(traj, box_charges, 280.)
