from __future__ import print_function
from simtk.openmm import app
import simtk.openmm as mm
from simtk import unit
from sys import stdout

pdb = app.PDBFile("./liquid.pdb")
forcefield = app.ForceField('tip3pfb.xml')

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9*unit.nanometers)
integrator = mm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
system.addForce(mm.MonteCarloBarostat(1 * unit.atmospheres, 300 * unit.kelvin, 25))

platform = mm.Platform.getPlatformByName('CUDA')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
print('Equilibrating...')
simulation.step(100)

simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True, temperature=True, totalSteps=1000, separator='\t', density=True))

print('Running Production...')
simulation.step(50000)
print('Done!')
