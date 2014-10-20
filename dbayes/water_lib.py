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
import repex

ff = app.ForceField("tip3p.xml")

def build_top(box_edge=2.1 * u.nanometers, nonbondedMethod=app.CutoffPeriodic):
    box = repex.testsystems.WaterBox(box_edge=box_edge, nonbondedMethod=nonbondedMethod)
    system = box.system
    positions = box.positions
    
    n_atoms = len(box.positions)
    
    top = []
    bonds = []
    for i in range(n_atoms / 3):
        j = 3 * i
        top.append(dict(serial=(j + 1), name="O", element="O", resSeq=(i+1), resName="HOH", chainID=(i+1)))
        top.append(dict(serial=(j + 2), name="H", element="H", resSeq=(i+1), resName="HOH", chainID=(i+1)))
        top.append(dict(serial=(j + 3), name="H", element="H", resSeq=(i+1), resName="HOH", chainID=(i+1)))
        bonds.append([j + 0, j + 1])
        bonds.append([j + 0, j + 2])

    top = pd.DataFrame(top)
    bonds = np.array(bonds, dtype='int')
    top = md.Topology.from_dataframe(top, bonds)
    
    xyz = positions / u.nanometers
    
    boxes = box.system.getDefaultPeriodicBoxVectors()
    lengths = boxes[0][0] / u.nanometers * np.ones((1, 3))
    
    angles = 90.0 * np.ones((1, 3))
    traj = md.Trajectory(xyz, top, unitcell_lengths=lengths, unitcell_angles=angles)
    
    mmtop = traj.top.to_openmm(traj=traj)
    
    return traj, mmtop, system, box, positions

def find_forces(system):
    for force in system.getForces():
        if type(force) == mm.NonbondedForce:
            nonbonded_force = force
        if type(force) == mm.HarmonicBondForce:
            bond_force = force
        if type(force) == mm.HarmonicAngleForce:
            angle_force = force
    
    return bond_force, angle_force, nonbonded_force

def set_constraints(system, r0, theta):
    r1 = 2 * r0 * np.sin(theta / 2.)
    n_constraints = system.getNumConstraints()
    n_water = n_constraints / 3

    for i in range(n_water * 2):
        a0, a1, d = system.getConstraintParameters(i)
        system.setConstraintParameters(i, a0, a1, r0 * u.nanometers)

    for i in range(n_water * 2, n_water * 3):
        a0, a1, d = system.getConstraintParameters(i)
        system.setConstraintParameters(i, a0, a1, r1 * u.nanometers)

def set_nonbonded(f_nonbonded, qH, sigma, epsilon, sigmaH, epsilonH):
    qO = -2.0 * qH
    for k in range(f_nonbonded.getNumParticles()):
        if k % 3 == 0:
            f_nonbonded.setParticleParameters(k, qO * u.elementary_charge, sigma * u.nanometer, epsilon * u.kilojoule_per_mole)
        else:
            f_nonbonded.setParticleParameters(k, qH * u.elementary_charge, sigmaH * u.nanometer, epsilonH * u.kilojoule_per_mole)

def set_parms(system, qH, sigma, epsilon, sigmaH, epsilonH, r0, theta):
    print("\nqH=%f, sigma=%f, epsilon=%f sigmaH=%f, epsilonH=%f, r0=%f, theta=%f\n" % (qH, sigma, epsilon, sigmaH, epsilonH, r0, theta))
    f_bond, f_angle, f_nonbonded = find_forces(system)
    
    set_constraints(system, r0, theta)
    set_nonbonded(f_nonbonded, qH, sigma, epsilon, sigmaH, epsilonH)


def build(system, positions, mmtop, temperature, pressure, qH, sigma, epsilon, sigmaH, epsilonH, r0, theta, stderr_tolerance=0.05, n_steps=250000, nonbondedCutoff=1.1 * u.nanometer, output_frequency=250, print_frequency=None):
    
    if print_frequency is None:
        print_frequency = int(n_steps / 3.)
    
    set_parms(system, qH, sigma, epsilon, sigmaH, epsilonH, r0, theta)

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
    simulation.context.setPositions(positions)

    print("minimizing")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    print("done minimizing")

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
    return d.Density.values

