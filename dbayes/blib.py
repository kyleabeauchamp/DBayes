import sys
import pymbar
import pymc as pm
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md

mass = 12.01078 * u.daltons + 4 * 35.4532 * u.daltons

sigma = pm.Uniform("sigma", 0.05, 1.0)
epsilon = pm.Uniform("sigma", 0.0, 100.0)

states = [dict(temperature=95.0, pressure=1.0), dict(temperature=105.0, pressure=1.0)]

measurements = [dict(temperature=298.15 * u.kelvin, pressure=101.325 * u.kilopascals, density=1584.36 * u.kilograms / (u.meter ** 3.))]

ff = app.ForceField("./test.xml")

def build_top(n_atoms=300, scaling=4.0):
    top = []
    for i in range(n_atoms):
        top.append(dict(serial=(i+1), name="C", element="C", resSeq=(i+1), resName="C", chainID=(i+1)))

    top = pd.DataFrame(top)
    bonds = np.zeros((0, 2), dtype='int')
    top = md.Topology.from_dataframe(top, bonds)
    xyz = np.random.normal(size=(n_atoms, 3))
    lengths = scaling * np.ones((1, 3))
    angles = 90.0 * np.ones((1, 3))
    traj = md.Trajectory(xyz, top, unitcell_lengths=lengths, unitcell_angles=angles)
    
    mmtop = traj.top.to_openmm(traj=traj)
    
    return traj, mmtop

def set_parms(f, sigma, epsilon, q=0.0):
    for k in range(f.getNumParticles()):
        f.setParticleParameters(k, q * u.elementary_charge, sigma * u.nanometer, epsilon * u.kilojoule_per_mole)

def build(traj, mmtop, temperature, pressure, sigma, epsilon, stderr_tolerance=0.05):
    system = ff.createSystem(mmtop, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=1.0 * u.nanometer)
    f = system.getForce(0)
    set_parms(f, sigma, epsilon)

    measurement = measurements[0]
    output_frequency = 250
    temperature = measurement["temperature"]
    pressure = measurement["pressure"]
    friction = 1.0 / u.picoseconds
    timestep = 3.0 * u.femtoseconds
    barostat_frequency = 25
    n_steps = 500000

    #out_filename = "./%d.h5" % (temperature / u.kelvin)
    csv_filename = "./%d.csv" % (temperature / u.kelvin)

    #integrator = mm.VariableLangevinIntegrator(temperature, friction, timestep)
    integrator = mm.LangevinIntegrator(temperature, friction, 1E-3)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    simulation = app.Simulation(mmtop, system, integrator)

    simulation.reporters.append(app.StateDataReporter(sys.stdout, 2500, density=True))
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True))
    simulation.context.setPositions(traj.openmm_positions(0))

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    converged = False
    while not converged:
        print("Simulating %d steps" % n_steps)
        simulation.step(n_steps)
        d = pd.read_csv(csv_filename, names=["Density"], skiprows=1)
        density_ts = np.array(d.Density)
        print(density_ts.mean())
        [t0, g, Neff] = pymbar.timeseries.detectEquilibration_fft(density_ts)
        density_ts = density_ts[t0:]
        density_mean_stderr = density_ts.std() / np.sqrt(Neff)
        if density_mean_stderr < stderr_tolerance:
            converged = True
    
    return system, d.Density.values
