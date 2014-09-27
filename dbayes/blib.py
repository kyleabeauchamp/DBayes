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

mass = 12.01078 * u.daltons + 4 * 35.4532 * u.daltons

ff = app.ForceField("./test.xml")

def build_box(atoms_per_dim, sigma):
    x_grid = np.arange(0, atoms_per_dim) * sigma
    xyz = []
    for i in range(atoms_per_dim):
        for j in range(atoms_per_dim):
            for k in range(atoms_per_dim):
                xyz.append(x_grid[[i, j, k]])
    
    xyz = np.array([xyz])
    return xyz

def build_top(atoms_per_dim, sigma):
    
    n_atoms = atoms_per_dim ** 3
    
    top = []
    for i in range(n_atoms):
        top.append(dict(serial=(i+1), name="C", element="C", resSeq=(i+1), resName="C", chainID=(i+1)))

    top = pd.DataFrame(top)
    bonds = np.zeros((0, 2), dtype='int')
    top = md.Topology.from_dataframe(top, bonds)
    
    xyz = build_box(atoms_per_dim, sigma)
    
    box_length = (atoms_per_dim + 1) * sigma

    lengths = box_length * np.ones((1, 3))
    
    angles = 90.0 * np.ones((1, 3))
    traj = md.Trajectory(xyz, top, unitcell_lengths=lengths, unitcell_angles=angles)
    
    mmtop = traj.top.to_openmm(traj=traj)
    
    return traj, mmtop

def set_parms(f, sigma, epsilon, q=0.0):
    print("\nsigma=%f, epsilon=%f" % (sigma, epsilon))
    for k in range(f.getNumParticles()):
        f.setParticleParameters(k, q * u.elementary_charge, sigma * u.nanometer, epsilon * u.kilojoule_per_mole)

def build(traj, mmtop, temperature, pressure, sigma, epsilon, stderr_tolerance=0.05, n_steps=250000, nonbondedCutoff=1.4*u.nanometer, output_frequency=250, print_frequency=None):
    
    if print_frequency is None:
        print_frequency = int(n_steps / 3.)
    
    system = ff.createSystem(mmtop, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=nonbondedCutoff)
    f = system.getForce(0)
    set_parms(f, sigma, epsilon)

    friction = 1.0 / u.picoseconds
    timestep = 3.0 * u.femtoseconds
    barostat_frequency = 25

    path = tempfile.mkdtemp()
    csv_filename = os.path.join(path, "density.csv")

    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    simulation = app.Simulation(mmtop, system, integrator)

    simulation.reporters.append(app.StateDataReporter(sys.stdout, print_frequency, step=True, density=True))
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True))
    simulation.context.setPositions(traj.openmm_positions(0))

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    converged = False
    while not converged:
        simulation.step(n_steps)
        d = pd.read_csv(csv_filename, names=["Density"], skiprows=1)
        density_ts = np.array(d.Density)
        [t0, g, Neff] = pymbar.timeseries.detectEquilibration_fft(density_ts)
        density_ts = density_ts[t0:]
        density_mean_stderr = density_ts.std() / np.sqrt(Neff)
        if density_mean_stderr < stderr_tolerance:
            converged = True
    print("temperature, density mean, stderr = %f, %f, %f" % (temperature / u.kelvin, density_ts.mean(), density_mean_stderr))
    return simulation, system, d.Density.values
