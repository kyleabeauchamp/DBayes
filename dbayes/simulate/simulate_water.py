from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit

pdb = app.PDBFile("./liquid.pdb")
forcefield = app.ForceField('tip3p.xml')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9*unit.nanometers)
integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 25))

simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
print('Equilibrating...')
simulation.step(10000)

simulation.reporters.append(app.DCDReporter('tip3p_300K_1ATM.dcd', 20000))
simulation.reporters.append(app.StateDataReporter("tip3p_300K_1ATM.csv", 20000, step=True, temperature=True, density=True, energy=True, totalSteps=1000, separator=","))

print('Running Production...')
simulation.step(8000000)
print('Done!')


