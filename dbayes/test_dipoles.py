import pymbar
import dipoles
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md


dipole = dipoles.Dipole(1000, q0=0.5)

temperature = 300 * u.kelvin
pressure = 1.0 * u.atmospheres

values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure)


nonbondedCutoff=1.1 * u.nanometer
output_frequency=250
print_frequency=1
import os
import tempfile
import sys

positions = traj.openmm_positions(0)

friction = 100.0 / u.picoseconds
timestep = 0.1 * u.femtoseconds
barostat_frequency = 25

path = tempfile.mkdtemp()
csv_filename = os.path.join(path, "density.csv")

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
#system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

simulation = app.Simulation(mmtop, system, integrator)

simulation.reporters.append(app.PDBReporter("out.pdb", 1))
simulation.reporters.append(app.StateDataReporter(sys.stdout, print_frequency, step=True, density=True))
simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True))
simulation.context.setPositions(positions)

print("minimizing")
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
print("done minimizing")
simulation.context.getState(getPositions=True).getPositions()[0:10]
simulation.step(5)
simulation.context.getState(getPositions=True).getPositions()[0:10]
simulation.step(1)
simulation.context.getState(getPositions=True).getPositions()[0:10]
simulation.step(100)
