import os
import tempfile
import sys
import pymbar
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

# 0.2 * 1000 / 18.015

temperature = 300.0 * u.kelvin
pressure = 1.0 * u.atmospheres
stderr_tolerance = 0.0001
ionicStrength = 2.0 * u.molar

ff = app.ForceField("tip3p.xml", "amber10.xml")

mmtop = app.Topology()
positions = []
modeller = app.Modeller(mmtop, positions)
modeller.addSolvent(ff, boxSize=mm.Vec3(2.2, 2.2, 2.2)*u.nanometers, ionicStrength=ionicStrength)


system = ff.createSystem(modeller.topology, nonbondedMethod=app.CutoffPeriodic)

output_frequency = 125
friction = 1.0 / u.picoseconds
timestep = 2.0 * u.femtoseconds
barostat_frequency = 25
n_steps = 100000

csv_filename = "density_%d_%0.1f.csv" % (temperature / temperature.unit, ionicStrength / ionicStrength.unit)
pdb_filename = "density_%d_%0.1f.pdb" % (temperature / temperature.unit, ionicStrength / ionicStrength.unit)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

simulation = app.Simulation(modeller.topology, system, integrator)

simulation.reporters.append(app.StateDataReporter(sys.stdout, output_frequency, density=True, potentialEnergy=True))
simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True, potentialEnergy=True))
simulation.context.setPositions(modeller.positions)

print("minimizing")
simulation.minimizeEnergy()

positions = simulation.context.getState(getPositions=True).getPositions()

app.PDBFile.writeFile(simulation.topology, positions, open(pdb_filename, 'w'))
print("done minimizing")
simulation.context.setVelocitiesToTemperature(temperature)


converged = False
while not converged:
    simulation.step(n_steps)
    d = pd.read_csv(csv_filename, names=["Energy", "Density"], skiprows=1)
    density_ts = np.array(d.Density)
    [t0, g, Neff] = pymbar.timeseries.detectEquilibration_fft(density_ts)
    density_ts = density_ts[t0:]
    density_mean_stderr = density_ts.std() / np.sqrt(Neff)
    if density_mean_stderr < stderr_tolerance:
        converged = True

