import pymc
import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

out_dir = "./monopole/"

temperature = 300 * u.kelvin
pressure = 1.0 * u.atmospheres

n_molecules = 500



monopole = dipoles.Monopole(n_molecules, q0=0.75, sigma0=0.25, sigma1=0.25, epsilon0=0.6, epsilon1=0.6)
traj = monopole.build_box()

values, mu, sigma = dipoles.simulate_density(monopole, temperature, pressure, out_dir, print_frequency=100)

import glob
filename = glob.glob("monopole/*.dcd")[0]
t = md.load(filename, top=traj)
t[-1].save("./monopole.pdb")


#traj = md.load("./dipoles.pdb")
#out_dir = "./symmetric/"


out_dir = "./"
dipole = dipoles.Dipole(n_molecules, q0=0.75, sigma0=0.25, epsilon0=0.6, sigma1=0.25, epsilon1=0.6, r0=0.2)
traj = dipole.build_box()
values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure, out_dir)


