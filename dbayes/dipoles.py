import os
import sys
import pymbar
import numpy as np
import pandas as pd
import simtk.openmm.app as app
import simtk.openmm as mm
import simtk.unit as u
import mdtraj as md
import simtk.unit as units

kB = u.BOLTZMANN_CONSTANT_kB * u.AVOGADRO_CONSTANT_NA

def GHMCIntegrator(temperature=298.0*u.kelvin, collision_rate=91.0/u.picoseconds, timestep=1.0*u.femtoseconds):
    """
    Create a generalized hybrid Monte Carlo (GHMC) integrator.
    
    Parameters
    ----------
    temperature : np.unit.Quantity compatible with kelvin, default: 298*simtk.unit.kelvin
        The temperature.
    collision_rate : np.unit.Quantity compatible with 1/picoseconds, default: 91.0/simtk.unit.picoseconds
        The collision rate.
    timestep : np.unit.Quantity compatible with femtoseconds, default: 1.0*simtk.unit.femtoseconds
        The integration timestep.
    
    Returns
    -------
    integrator : simtk.openmm.CustomIntegrator
        A GHMC integrator.

    Notes
    -----
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    Metrpolization step to ensure sampling from the appropriate distribution.

    Additional global variables 'ntrials' and  'naccept' keep track of how many trials have been attempted and
    accepted, respectively.

    TODO
    ----
    Move initialization of 'sigma' to setting the per-particle variables.

    Examples
    --------

    Create a GHMC integrator.

    >>> temperature = 298.0 * simtk.unit.kelvin
    >>> collision_rate = 91.0 / simtk.unit.picoseconds
    >>> timestep = 1.0 * simtk.unit.femtoseconds
    >>> integrator = GHMCIntegrator(temperature, collision_rate, timestep)

    References
    ----------
    Lelievre T, Stoltz G, and Rousset M. Free Energy Computations: A Mathematical Perspective
    http://www.amazon.com/Free-Energy-Computations-Mathematical-Perspective/dp/1848162472

    """

    # Initialize constants.
    kT = kB * temperature
    gamma = collision_rate

    # Create a new custom integrator.
    integrator = mm.CustomIntegrator(timestep)

    #
    # Integrator initialization.
    #
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("b", np.exp(-gamma*timestep)) # velocity mixing parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addGlobalVariable("ke", 0) # kinetic energy
    integrator.addPerDofVariable("vold", 0) # old velocities
    integrator.addPerDofVariable("xold", 0) # old positions
    integrator.addGlobalVariable("Eold", 0) # old energy
    integrator.addGlobalVariable("Enew", 0) # new energy
    integrator.addGlobalVariable("accept", 0) # accept or reject
    integrator.addGlobalVariable("naccept", 0) # number accepted
    integrator.addGlobalVariable("ntrials", 0) # number of Metropolization trials
    integrator.addPerDofVariable("x1", 0) # position before application of constraints
    
    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Allow context updating here.
    #
    integrator.addUpdateContextState();

    #
    # Constrain positions.
    #
    integrator.addConstrainPositions();

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();
    
    #
    # Metropolized symplectic step.
    #
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Eold", "ke + energy")
    integrator.addComputePerDof("xold", "x")
    integrator.addComputePerDof("vold", "v")
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m")
    integrator.addComputePerDof("x", "x + v*dt")
    integrator.addComputePerDof("x1", "x")
    integrator.addConstrainPositions();
    integrator.addComputePerDof("v", "v + 0.5*dt*f/m + (x-x1)/dt")
    integrator.addConstrainVelocities();
    integrator.addComputeSum("ke", "0.5*m*v*v")
    integrator.addComputeGlobal("Enew", "ke + energy")
    integrator.addComputeGlobal("accept", "step(exp(-(Enew-Eold)/kT) - uniform)")
    integrator.addComputePerDof("x", "x*accept + xold*(1-accept)")
    integrator.addComputePerDof("v", "v*accept - vold*(1-accept)")

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(b)*v + sqrt(1-b)*sigma*gaussian")
    integrator.addConstrainVelocities();

    #
    # Accumulate statistics.
    #
    integrator.addComputeGlobal("naccept", "naccept + accept")
    integrator.addComputeGlobal("ntrials", "ntrials + 1")   

    return integrator


def build_box(atoms_per_dim, spacing):
    x_grid = np.arange(0, atoms_per_dim) * spacing
    xyz = []
    for i in range(atoms_per_dim):
        for j in range(atoms_per_dim):
            for k in range(atoms_per_dim):
                xyz.append(x_grid[[i, j, k]])    
    xyz = np.array([xyz])
    return xyz

class Molecule(object):

    @property
    def mmtop(self):
        return self.traj.top.to_openmm(traj=self.traj)

    @property
    def traj(self):
        return self._traj

    @traj.setter
    def traj(self, value):
        self._traj = value
    
    def build_system(self):
        ff = app.ForceField("./%s.xml" % self.name)
        system = ff.createSystem(self.mmtop, nonbondedMethod=app.PME, constraints=app.AllBonds)
        return system

    def set_parameters(self, system):
        self.set_constraints(system)
        self.set_nonbonded(system)

    def set_constraints(self, system):
        pass

class Dipole(Molecule):
    def __init__(self, n_molecules, q0=0.5, sigma0=0.3, sigma1=0.3, epsilon0=0.5, epsilon1=0.5, r0=0.2):
        self.n_molecules = n_molecules

        self.q0 = q0
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.epsilon0 = epsilon0
        self.epsilon1 = epsilon1
        self.r0 = r0
        
        self.name = "dipole"

    def __repr__(self):
        return "q0=%f, sigma0=%f, sigma1=%f, epsilon0=%f, epsilon1=%f, r0=%f" % (self.q0, self.sigma0, self.sigma1, self.epsilon0, self.epsilon1, self.r0)

    @property
    def q1(self):
        return -1.0 * self.q0

    @property
    def n_atoms(self):
        return self.n_molecules * 2

    def build_box(self):
        top = []
        bonds = []
        for i in range(self.n_molecules):
            j = 2 * i
            top.append(dict(serial=(j + 1), name="F", element="F", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            top.append(dict(serial=(j + 2), name="H", element="H", resSeq=(i+1), resName="HOH", chainID=(i+1)))
            bonds.append([j + 0, j + 1])

        top = pd.DataFrame(top)
        bonds = np.array(bonds, dtype='int')
        top = md.Topology.from_dataframe(top, bonds)
        
        atoms_per_dim = int(np.ceil(self.n_molecules ** (1 / 3.)))
        spacing = self.r0 + (self.sigma0 + self.sigma1) * 0.5 * 2 ** (1 / 6.)
        
        centroids = build_box(atoms_per_dim, spacing)
        xyz = np.zeros((1, self.n_atoms, 3))
        
        for i in range(self.n_molecules):
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
        self.traj = traj
        
        return traj

        
    def set_constraints(self, system):
        for i in range(self.n_molecules):
            a0, a1 = i * 2, i * 2 + 1
            system.setConstraintParameters(i, a0, a1, self.r0 * u.nanometers)

    def set_nonbonded(self, system):
        for force in system.getForces():
            if type(force) == mm.NonbondedForce:
                f_nonbonded = force        
        
        for i in range(self.n_molecules):
            a, b = 2 * i, 2 * i + 1
            
            f_nonbonded.setParticleParameters(a, self.q0 * u.elementary_charge, self.sigma0 * u.nanometer, self.epsilon0 * u.kilojoule_per_mole)
            f_nonbonded.setParticleParameters(b, self.q1 * u.elementary_charge, self.sigma1 * u.nanometer, self.epsilon1 * u.kilojoule_per_mole)


class Monopole(Molecule):
    def __init__(self, n_molecules, q0=0.5, sigma0=0.3, sigma1=0.3, epsilon0=0.5, epsilon1=0.5, name0="Na", name1="Cl"):
        self.n_molecules = n_molecules

        self.q0 = q0
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.epsilon0 = epsilon0
        self.epsilon1 = epsilon1
        
        self.name0 = name0
        self.name1 = name1
        
        self.name = "monopole"

    def __repr__(self):
        return "q0=%f, sigma0=%f, sigma1=%f, epsilon0=%f, epsilon1=%f" % (self.q0, self.sigma0, self.sigma1, self.epsilon0, self.epsilon1)

    @property
    def q1(self):
        return -1.0 * self.q0

    @property
    def n_atoms(self):
        return self.n_molecules * 2

    def build_box(self):
        top = []
        bonds = []
        for i in range(self.n_molecules):
            a = 2 * i + 1  # One based indexing for PDB / top stuff
            b = 2 * i + 2
            top.append(dict(serial=a, name=self.name0, element=self.name0, resSeq=a, resName=self.name0, chainID=(1)))
            top.append(dict(serial=b, name=self.name1, element=self.name1, resSeq=b, resName=self.name1, chainID=(1)))

        top = pd.DataFrame(top)
        bonds = np.array(bonds, dtype='int')
        top = md.Topology.from_dataframe(top, bonds)
        
        atoms_per_dim = int(np.ceil(self.n_atoms ** (1 / 3.)))
        spacing = (self.sigma0 + self.sigma1) * 0.5 * 2 ** (1 / 6.)
        
        centroids = build_box(atoms_per_dim, spacing)
        xyz = np.zeros((1, self.n_atoms, 3))
        
        for i in range(self.n_atoms):
            xyz[0, i] = centroids[0, i]

        box_length = xyz.max() + spacing

        lengths = box_length * np.ones((1, 3))
        
        angles = 90.0 * np.ones((1, 3))
        traj = md.Trajectory(xyz, top, unitcell_lengths=lengths, unitcell_angles=angles)
        
        self.traj = traj
        
        return traj


    def set_nonbonded(self, system):
        for force in system.getForces():
            if type(force) == mm.NonbondedForce:
                f_nonbonded = force        
        
        for i in range(self.n_molecules):
            a, b = 2 * i, 2 * i + 1

            f_nonbonded.setParticleParameters(a, self.q0 * u.elementary_charge, self.sigma0 * u.nanometer, self.epsilon0 * u.kilojoule_per_mole)
            f_nonbonded.setParticleParameters(b, self.q1 * u.elementary_charge, self.sigma1 * u.nanometer, self.epsilon1 * u.kilojoule_per_mole)
    

def simulate_density(molecule, temperature, pressure, out_dir, stderr_tolerance=0.00005, n_steps=250000, nonbondedCutoff=1.1 * u.nanometer, output_frequency=250, print_frequency=None, langevin_tolerance=0.0005):
    
    if print_frequency is None:
        print_frequency = int(n_steps / 3.)
    
    mmtop = molecule.mmtop
     
    system = molecule.build_system()
    molecule.set_parameters(system)
    
    positions = molecule.traj.openmm_positions(0)

    friction = 1.0 / u.picoseconds
    barostat_frequency = 25
    
    dcd_filename = os.path.join(out_dir, "%s_%f.dcd" % (str(molecule), temperature / u.kelvin))
    csv_filename = os.path.join(out_dir, "%s_%f.csv" % (str(molecule), temperature / u.kelvin))

    #integrator = mm.VariableLangevinIntegrator(temperature, friction, langevin_tolerance / 10.)
    integrator = GHMCIntegrator(temperature, friction, 1.0 * u.femtoseconds)
    
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostat_frequency))

    simulation = app.Simulation(mmtop, system, integrator)
    simulation.context.setPositions(positions)

    print("minimizing")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)
    print("done minimizing")
    #simulation.step(2000)
    #integrator.setErrorTolerance(langevin_tolerance)

    simulation.reporters.append(app.DCDReporter(dcd_filename, output_frequency))
    simulation.reporters.append(app.StateDataReporter(sys.stdout, print_frequency, step=True, density=True, potentialEnergy=True))
    simulation.reporters.append(app.StateDataReporter(csv_filename, output_frequency, density=True, potentialEnergy=True))

    converged = False
    while not converged:
        simulation.step(n_steps)
        d = pd.read_csv(csv_filename, skiprows=1, names=["energy", "density"])
        density_ts = np.array(d.density)
        [t0, g, Neff] = pymbar.timeseries.detectEquilibration(density_ts)
        density_ts = density_ts[t0:]
        density_mean_stderr = density_ts.std() / np.sqrt(Neff)
        if density_mean_stderr < stderr_tolerance:
            converged = True
    print("temperature, density mean, stderr = %f, %f, %f" % (temperature / u.kelvin, density_ts.mean(), density_mean_stderr))
    return d.density.values, density_ts.mean(), density_mean_stderr

