import sys
import blib
import pymc
import numpy as np
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

ff = blib.ff


def set_parms(f, sigma, epsilon, q=0.0):
    print("\nsigma=%f, epsilon=%f" % (sigma, epsilon))
    for k in range(f.getNumParticles()):
        f.setParticleParameters(k, q * u.elementary_charge, sigma * u.nanometer, epsilon * u.kilojoule_per_mole)


def setup(traj, mmtop, temperature, pressure, sigma, epsilon, nonbondedCutoff=1.4*u.nanometer):
    
    system = ff.createSystem(mmtop, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=nonbondedCutoff)
    f = system.getForce(0)
    set_parms(f, sigma, epsilon)

    friction = 1.0 / u.picoseconds
    timestep = 3.0 * u.femtoseconds
    barostat_frequency = 25

    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    simulation = app.Simulation(mmtop, system, integrator)

    simulation.reporters.append(app.StateDataReporter(sys.stdout, 100, step=True, density=True))
    simulation.context.setPositions(traj.openmm_positions(0))

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.step(10000)
    return simulation


def propagate(simulation, state, temperature, sigma, epsilon):
        
    f = simulation.system.getForce(0)
    set_parms(f, sigma, epsilon)
    
    simulation.context.setState(state)

    simulation.context.setVelocitiesToTemperature(temperature)

    simulation.step(5000)
    
    state = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True)

    return state

mass = 12.01078 * u.daltons + 4 * 35.4532 * u.daltons

#sigma = pymc.Uniform("sigma", 0.53, 0.57, value=0.545)
sigma0 = 0.545
epsilon = 13.0

observed = 1.58436 * u.grams / u.milliliter
observed = observed / (u.grams / u.milliliter)
error = 0.02

temperature = 298.15 * u.kelvin
pressure = 101.325 * u.kilopascals
kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA
kt = kB * temperature

atoms_per_dim = 7
n_atoms = atoms_per_dim ** 3

traj, mmtop = blib.build_top(atoms_per_dim, sigma0)

simulation = setup(traj, mmtop, temperature, pressure, sigma0, epsilon)
state0 = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True)


class Step(object):
    def __init__(self, var):
        self.var = var.name

    def step(self, point):
        new = point.copy()
        #new[self.var] = 10 + np.random.rand() # Normal samples
        state = point['state']
        sigma = point['sigma']
        new[self.var] = propagate(simulation, state, temperature, sigma, epsilon)

        return new

with pymc.Model() as model:
    sigma = pymc.Uniform("sigma", 0.535, 0.565)
    state = pymc.Flat('state')

    step1 = pymc.step_methods.NUTS(vars=[sigma])
    step2 = Step(state) # not sure how to limit this to one variable

    trace = pymc.sample(10, [step1, step2])

pymc.traceplot(trace[:])
show()
