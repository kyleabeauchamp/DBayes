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

temperature = 80 * u.kelvin
pressure = 1.0 * u.atmospheres


monopole = dipoles.Monopole(1000, q0=1.0, sigma0=0.4, epsilon0=0.6, sigma1=0.4, epsilon1=0.6)
traj = monopole.build_box()

values, mu, sigma = dipoles.simulate_density(monopole, temperature, pressure, out_dir, print_frequency=100)

import glob
filename = glob.glob("monopole/*.dcd")[0]
t = md.load(filename, top=traj)
t[-1].save("./monopole.pdb")


#traj = md.load("./dipoles.pdb")
out_dir = "./symmetric/"



dipole = dipoles.Dipole(1000, q0=1.0, sigma0=0.4, epsilon0=0.6, sigma1=0.4, epsilon1=0.6, r0=0.2)
values, mu, sigma = dipoles.simulate_density(dipole, traj, temperature, pressure, out_dir, print_frequency=100)


