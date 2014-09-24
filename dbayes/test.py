import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

ff = app.ForceField("./test.xml")


n_atoms = 300
top = []
for i in range(n_atoms):
    top.append(dict(serial=(i+1), name="C", element="C", resSeq=(i+1), resName="C", chainID=(i+1)))

top = pd.DataFrame(top)
bonds = np.zeros((0, 2), dtype='int')
top = md.Topology.from_dataframe(top, bonds)
xyz = np.random.normal(size=(n_atoms, 3))
lengths = 4.0 * np.ones((1, 3))
traj = md.Trajectory(xyz, top, unitcell_lengths=lengths)


mmtop = traj.top.to_openmm()
mmtop.setUnitCellDimensions(mm.Vec3(*traj.unitcell_lengths[0]) * u.nanometer)
system = ff.createSystem(mmtop, nonbondedMethod=app.CutoffPeriodic)

output_frequency = 100
temperature = 105 * u.kelvin
friction = 1.0 / u.picoseconds
timestep = 3.0 * u.femtoseconds
pressure = 1.0 * u.atmospheres
barostat_frequency = 25
n_steps = 4000000

out_filename = "./%d.h5" % (temperature / u.kelvin)
csv_filename = "./%d.csv" % (temperature / u.kelvin)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

simulation = app.Simulation(mmtop, system, integrator)
simulation.reporters.append(md.reporters.HDF5Reporter(out_filename, output_frequency))
simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, step=True, potentialEnergy=True, temperature=True, density=True))
simulation.context.setPositions(traj.openmm_positions(0))

simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temperature)

simulation.step(n_steps)

x = pd.read_csv(csv_filename)["Density (g/mL)"]
x[10000:].mean()

"""
095 K: 0.6061323239846188
105 K: 0.5974042439570022
"""
