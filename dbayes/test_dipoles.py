import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md


dipole = dipoles.Dipole(1000, q0=1.0)

temperature = 300 * u.kelvin
pressure = 1.0 * u.atmospheres

values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure)
