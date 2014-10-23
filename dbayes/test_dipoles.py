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

values, mu, sigma = dipoles.simulate_density(dipole, temperature, pressure, print_frequency=1)


traj, mmtop = dipole.build_box()
system = dipole.build_system(mmtop)
dipole.set_parameters(system)

positions = traj.openmm_positions(0)

friction = 1.0 / u.picoseconds
timestep = 2.0 * u.femtoseconds
barostat_frequency = 25

import tempfile, os, sys
print_frequency = 50
output_frequency = 50
path = tempfile.mkdtemp()
csv_filename = os.path.join(path, "density.csv")

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

simulation = app.Simulation(mmtop, system, integrator)
simulation.context.setPositions(positions)

print("minimizing")
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)
print("done minimizing")

simulation.context.getIntegrator().setStepSize(timestep / 20.)
simulation.step(250)
simulation.context.getIntegrator().setStepSize(timestep / 10.)
simulation.step(250)
simulation.context.getIntegrator().setStepSize(timestep / 5.)
simulation.step(250)

simulation.reporters.append(app.DCDReporter("out.dcd", output_frequency))
simulation.reporters.append(app.StateDataReporter(sys.stdout, print_frequency, step=True, density=True, potentialEnergy=True))
simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True))

simulation.step(15000)
