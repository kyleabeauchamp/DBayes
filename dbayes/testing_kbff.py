import kbforcefield
import simtk.unit as u
from simtk.openmm import app
import simtk.openmm as mm
import mdtraj as md

#ff = kbforcefield.ForceField('ccl4.xml')
ff = app.forcefield.ForceField('ccl4.xml')

temperature = 300 * u.kelvin
friction = 0.3 / u.picosecond
timestep = 0.01 * u.femtosecond

traj = md.load("./lig-0-0.gaff.mol2")
traj.unitcell_angles = np.array([[90., 90., 90.]])
traj.unitcell_lengths = np.array([[2.0, 2.0, 2.0]])

#model = app.modeller.Modeller(traj.top.to_openmm(), traj.openmm_positions(0))
positions = traj.openmm_positions(0)
topology = traj.top.to_openmm(traj=traj)

system = ff.createSystem(topology, nonbondedMethod=kbforcefield.PME, nonbondedCutoff=1.0 * u.nanometers, constraints=kbforcefield.HAngles)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)

simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
print("running")
simulation.step(1)


