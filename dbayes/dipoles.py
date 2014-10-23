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

def build_box(atoms_per_dim, spacing):
    x_grid = np.arange(0, atoms_per_dim) * spacing
    xyz = []
    for i in range(atoms_per_dim):
        for j in range(atoms_per_dim):
            for k in range(atoms_per_dim):
                xyz.append(x_grid[[i, j, k]])
    
    xyz = np.array([xyz])
    return xyz


class Dipole(object):
    def __init__(self, n_dipoles, q0=0.5, sigma0=0.3, sigma1=0.3, epsilon0=0.5, epsilon1=0.5, r0=0.2, mass=1.0):
        self.n_dipoles = n_dipoles

        self.q0 = q0
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.epsilon0 = epsilon0
        self.epsilon1 = epsilon1
        self.r0 = r0
        
        self.mass = mass
        
    @property
    def q1(self):
        return -1.0 * self.q0

    @property
    def n_atoms(self):
        return self.n_dipoles * 2

    def build_system(self):
        system = mm.System()
        nonbonded = mm.NonbondedForce()
        nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)        
        
        for i in range(self.n_dipoles):
            a = system.addParticle(self.mass)
            b = system.addParticle(self.mass)
            system.addConstraint(a, b, self.r0 * u.nanometers)
            
            nonbonded.addParticle(self.q0 * u.elementary_charge, self.sigma0 * u.nanometer, self.epsilon0 * u.kilojoule_per_mole)
            nonbonded.addParticle(self.q1 * u.elementary_charge, self.sigma1 * u.nanometer, self.epsilon1 * u.kilojoule_per_mole)

        system.addForce(nonbonded)

        return system
    
    def build_box(self):
        top = []
        bonds = []
        for i in range(self.n_dipoles):
            j = 2 * i
            top.append(dict(serial=(j + 1), name="O", element="O", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            top.append(dict(serial=(j + 2), name="H", element="H", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            bonds.append([j + 0, j + 1])

        top = pd.DataFrame(top)
        bonds = np.array(bonds, dtype='int')
        top = md.Topology.from_dataframe(top, bonds)
        
        atoms_per_dim = int(np.ceil(self.n_dipoles ** (1 / 3.)))
        spacing = self.r0 + (self.sigma0 + self.sigma1) * 0.5
        
        centroids = build_box(atoms_per_dim, spacing)
        xyz = np.zeros((1, self.n_atoms, 3))
        
        for i in range(self.n_dipoles):
            a, b = 2 * i, 2 * i + 1
            xyz[0, a] = centroids[0, i]
            xyz[0, b] = centroids[0, i]
            xyz[0, b, 2] += self.r0
            
        box_length = (atoms_per_dim + 1) * spacing

        lengths = box_length * np.ones((1, 3))
        
        angles = 90.0 * np.ones((1, 3))
        traj = md.Trajectory(xyz, top, unitcell_lengths=lengths, unitcell_angles=angles)
        
        mmtop = traj.top.to_openmm(traj=traj)
        
        return traj, mmtop

        
    def set_constraints(self, system):

        for i in range(self.n_dipoles):
            a0, a1 = i * 2, i * 2 + 1
            system.setConstraintParameters(i, a0, a1, self.r0 * u.nanometers)

    def set_nonbonded(self, system):
        f_nonbonded = _find_nonbonded_force(system)
        
        for i in range(self.n_dipoles):
            a, b = 2 * i, 2 * i + 1
            
            f_nonbonded.setParticleParameters(k, self.q0 * u.elementary_charge, self.sigma0 * u.nanometer, self.epsilon0 * u.kilojoule_per_mole)
            f_nonbonded.setParticleParameters(k, self.q1 * u.elementary_charge, self.sigma1 * u.nanometer, self.epsilon1 * u.kilojoule_per_mole)
    
    def set_parmameters(self, system):
        self.set_constraints(system)
        self.set_nonbonded(system)




def simulate_density(dipole, temperature, pressure, stderr_tolerance=0.05, n_steps=250000, nonbondedCutoff=1.1 * u.nanometer, output_frequency=250, print_frequency=None):
    
    if print_frequency is None:
        print_frequency = int(n_steps / 3.)
    
    system = dipole.build_system()
    traj, mmtop = dipole.build_box()
    positions = traj.openmm_positions(0)

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
    return d.Density.values, density_ts.mean(), density_mean_stderr

