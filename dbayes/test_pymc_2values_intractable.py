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

sigma = pymc.Uniform("sigma", 0.53, 0.57, value=0.545)
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

traj, mmtop = blib.build_top(atoms_per_dim, sigma.value)

simulation = setup(traj, mmtop, temperature, pressure, sigma.value, epsilon)
state = simulation.context.getState(getPositions=True, getParameters=True, getEnergy=True)


@pymc.stochastic(dtype="O")
def y(value=state, sigma=sigma):
    def logp(value, sigma):
        return 1.0
        #return -value.getPotentialEnergy() / kt

    def random(sigma):
        print("stepping y")
        print(sigma)
        return propagate(simulation, state, temperature, sigma, epsilon)

@pymc.stochastic(dtype="O")
def x(value=state, sigma=sigma, y=y):
    def logp(value, sigma, y):
        return -(y.getPotentialEnergy() - value.getPotentialEnergy()) / kt
        #return -value.getPotentialEnergy() / kt

    def random(sigma, y):
        print(sigma)
        return propagate(simulation, state, temperature, sigma, epsilon)

@pymc.potential
def density(y=y):
    return 0.0

#@pymc.deterministic
#def density(y=y):
#    v = y.getPeriodicBoxVolume()
#    return (n_atoms * mass / v) * (1 / (u.grams / u.milliliter)) / (u.AVOGADRO_CONSTANT_NA)

#measurement = pymc.Normal("observed_density", mu=density, tau=error ** -2., value=observed, observed=True)

variables = [sigma, x, y, density]
model = pymc.Model(variables)
mcmc = pymc.MCMC(model)

mcmc.use_step_method(pymc.NoStepper, y)

mcmc.sample(5)

s = mcmc.trace("sigma")[:]
rho = mcmc.trace("density")[:]


