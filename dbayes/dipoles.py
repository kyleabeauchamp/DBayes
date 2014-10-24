import os
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
    #signs = []
    #current_sign = 1.0
    for i in range(atoms_per_dim):
        for j in range(atoms_per_dim):
            for k in range(atoms_per_dim):
                xyz.append(x_grid[[i, j, k]])
                #signs.append(current_sign)
                #current_sign *= -1.
    
    xyz = np.array([xyz])
    #signs = np.array(signs)
    return xyz


class Dipole(object):
    def __init__(self, n_dipoles, q0=0.5, sigma0=0.3, sigma1=0.3, epsilon0=0.5, epsilon1=0.5, r0=0.2, mass=16.0):
        self.n_dipoles = n_dipoles

        self.q0 = q0
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.epsilon0 = epsilon0
        self.epsilon1 = epsilon1
        self.r0 = r0
        
        self.mass = mass

    def __repr__(self):
        return "q0=%f, sigma0=%f, sigma1=%f, epsilon0=%f, epsilon1=%f, r0=%f" % (self.q0, self.sigma0, self.sigma1, self.epsilon0, self.epsilon1, self.r0)

    @property
    def q1(self):
        return -1.0 * self.q0

    @property
    def n_atoms(self):
        return self.n_dipoles * 2
        
    def build_system(self, mmtop):
        ff = app.ForceField("./dipole.xml")
        system = ff.createSystem(mmtop, nonbondedMethod=app.PME, constraints=app.AllBonds)
        return system

    def build_box(self):
        top = []
        bonds = []
        for i in range(self.n_dipoles):
            j = 2 * i
            top.append(dict(serial=(j + 1), name="F", element="F", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            top.append(dict(serial=(j + 2), name="H", element="H", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            bonds.append([j + 0, j + 1])

        top = pd.DataFrame(top)
        bonds = np.array(bonds, dtype='int')
        top = md.Topology.from_dataframe(top, bonds)
        
        atoms_per_dim = int(np.ceil(self.n_dipoles ** (1 / 3.)))
        spacing = self.r0 + (self.sigma0 + self.sigma1) * 0.5 * 2 ** (1 / 6.)
        
        centroids = build_box(atoms_per_dim, spacing)
        xyz = np.zeros((1, self.n_atoms, 3))
        
        for i in range(self.n_dipoles):
            a, b = 2 * i, 2 * i + 1
            xyz[0, a] = centroids[0, i]
            xyz[0, b] = centroids[0, i]
            
            ind = np.random.random_integers(0, 2)
            sgn = np.random.random_integers(0, 1) * 2 - 1
            xyz[0, b, ind] += self.r0 * sgn
            
        box_length = xyz.max() + spacing

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
        for force in system.getForces():
            if type(force) == mm.NonbondedForce:
                f_nonbonded = force        
        
        for i in range(self.n_dipoles):
            a, b = 2 * i, 2 * i + 1
            
            f_nonbonded.setParticleParameters(i, self.q0 * u.elementary_charge, self.sigma0 * u.nanometer, self.epsilon0 * u.kilojoule_per_mole)
            f_nonbonded.setParticleParameters(i, self.q1 * u.elementary_charge, self.sigma1 * u.nanometer, self.epsilon1 * u.kilojoule_per_mole)
    
    def set_parameters(self, system):
        self.set_constraints(system)
        self.set_nonbonded(system)




def simulate_density(dipole, traj, temperature, pressure, out_dir, stderr_tolerance=0.05, n_steps=250000, nonbondedCutoff=1.1 * u.nanometer, output_frequency=250, print_frequency=None, timestep=1.0*u.femtoseconds):
    
    if print_frequency is None:
        print_frequency = int(n_steps / 3.)
    
    
    #traj, mmtop = dipole.build_box()
    mmtop = traj.top.to_openmm(traj=traj)
     
    system = dipole.build_system(mmtop)
    dipole.set_parameters(system)
    
    positions = traj.openmm_positions(0)

    friction = 1.0 / u.picoseconds
    barostat_frequency = 25
    
    dcd_filename = os.path.join(out_dir, "%s_%f.dcd" % (str(dipole), temperature / u.kelvin))
    csv_filename = os.path.join(out_dir, "%s_%f.csv" % (str(dipole), temperature / u.kelvin))

    #integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    langevin_tolerance = 0.0005
    integrator = mm.VariableLangevinIntegrator(temperature, friction, langevin_tolerance)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    simulation = app.Simulation(mmtop, system, integrator)
    simulation.context.setPositions(positions)

    print("minimizing")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    print("done minimizing")

    if False:
        simulation.context.getIntegrator().setStepSize(timestep / 30.)
        simulation.step(500)
        simulation.context.getIntegrator().setStepSize(timestep / 20.)
        simulation.step(500)
        simulation.context.getIntegrator().setStepSize(timestep / 10.)
        simulation.step(500)
        simulation.context.getIntegrator().setStepSize(timestep / 5.)
        simulation.step(500)
        simulation.context.getIntegrator().setStepSize(timestep / 2.)
        simulation.step(500)
        simulation.context.getIntegrator().setStepSize(timestep)
        simulation.step(500)

    simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
    simulation.reporters.append(app.StateDataReporter(sys.stdout, print_frequency, step=True, density=True, potentialEnergy=True))
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True))

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

