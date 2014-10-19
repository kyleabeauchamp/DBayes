import gaff2xml

m0 = gaff2xml.openeye.smiles_to_oemol("ClC(Cl)(Cl)Cl")
charged = gaff2xml.openeye.get_charges(m0)

trajectories, ffxml = gaff2xml.openeye.oemols_to_ffxml([charged])
f = open("./ccl4.xml", 'w')
f.write(ffxml.read())


import simtk.unit as u
from simtk.openmm import app
import simtk.openmm as mm
import mdtraj as md

ff = app.ForceField("./ccl4.xml", "tip3p.xml")

temperature = 300 * u.kelvin
friction = 0.3 / u.picosecond
timestep = 0.01 * u.femtosecond

traj = md.load("./lig-0-0.gaff.mol2")
traj.unitcell_angles = np.array([[90., 90., 90.]])
traj.unitcell_lengths = np.array([[1.0, 1.0, 1.0]])

model = app.modeller.Modeller(traj.top.to_openmm(), traj.openmm_positions(0))
model.addSolvent(ff, padding=1.0 * u.nanometer)

system = ff.createSystem(model.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * u.nanometers, constraints=app.HAngles)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)

simulation = app.Simulation(model.topology, system, integrator)
simulation.context.setPositions(model.positions)
print("running")
simulation.step(1)
