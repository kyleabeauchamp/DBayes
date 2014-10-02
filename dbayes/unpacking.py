import simtk.openmm as mm
import simtk.unit as u
import numpy as np

class Unpacker(object):
    def __init__(self, state):
        positions, velocities, forces, time, kinetic_energy, potential_energy, boxes, parameters = self.state_to_tuple(state)
        
        self.kinetic_energy_unit = kinetic_energy.unit
        self.potential_energy_unit = potential_energy.unit
        
        self.forces_unit = forces.unit
        self.positions_unit = positions.unit
        self.velocities_unit = velocities.unit
        #self.parameters_unit = dict((key, val.unit) for key, val in parameters.items())
        self.boxes_unit = boxes.unit
        self.time_unit = time.unit
        
        self.parameters_order = parameters.keys()
        
        self.n_atoms = len(positions)
        self.N = self.n_atoms * 3 * 3  # Positions, velocities, forces
        self.N += 1  # Time
        self.N += 2  # Potential and Kinetic Energy
        self.N += 9  # Box
        self.N += len(parameters)  # Other shit
        
        self.split_indices = [self.n_atoms * 3, self.n_atoms * 3 * 2, self.n_atoms * 3 * 3]
        
        self.split_indices.append(self.n_atoms * 3 * 3 + 1)  # time
        self.split_indices.append(self.n_atoms * 3 * 3 + 2)  # KE
        self.split_indices.append(self.n_atoms * 3 * 3 + 3)  # PE
        self.split_indices.append(self.n_atoms * 3 * 3 + 12)  # boxes
        #self.split_indices.append(self.n_atoms * 3 * 3 + 12 + 1 + len(parameters))  # Don't need an index for the last entries, e.g. parameters
        

    def state_to_tuple(self, state):

        time = state.getTime()

        kinetic_energy = state.getKineticEnergy()
        potential_energy = state.getPotentialEnergy()

        positions = state.getPositions(asNumpy=True)
        velocities = state.getVelocities(asNumpy=True)
        forces = state.getForces(asNumpy=True)        
        boxes =  state.getPeriodicBoxVectors(asNumpy=True)        
        parameters = state.getParameters()

        return positions, velocities, forces, time, kinetic_energy, potential_energy, boxes, parameters
    
    def state_to_array(self, state):
        positions, velocities, forces, time, kinetic_energy, potential_energy, boxes, parameters = self.state_to_tuple(state)
        
        arr = np.zeros(self.N)
        
        positions_arr, velocities_arr, forces_arr, time_arr, kinetic_energy_arr, potential_energy_arr, boxes_arr, parameters_arr = np.split(arr, self.split_indices)
        
        positions_arr[:] = (positions / self.positions_unit).flat
        velocities_arr[:] = (velocities / self.velocities_unit).flat
        forces_arr[:] = (forces / self.forces_unit).flat

        time_arr[:] = time / self.time_unit
        
        kinetic_energy_arr[:] = kinetic_energy / self.kinetic_energy_unit
        potential_energy_arr[:] = potential_energy / self.potential_energy_unit
        
        boxes_arr[:] = (boxes / self.boxes_unit).flat
        
        parameters_arr[:] = parameters["MonteCarloPressure"]  # HACK HARDCODED, fix me later
        
        return arr
        
    def array_to_state(self, array):
        #state = mm.State(self, energy=energy, coordList=None, velList=None, forceList=None, periodicBoxVectorsList=None, paramMap=None)
        positions_arr, velocities_arr, forces_arr, time_arr, kinetic_energy_arr, potential_energy_arr, boxes_arr, parameters_arr = np.split(array, self.split_indices)

        kinetic_energy = kinetic_energy_arr[0]  # * self.kinetic_energy_unit
        potential_energy = potential_energy_arr[0]  # * self.potential_energy_unit
        energy = (kinetic_energy, potential_energy)

        positions = positions_arr.reshape((self.n_atoms, 3))
        #positions = u.Quantity(map(lambda x: tuple(x), positions), self.positions_unit)
        positions = map(lambda x: tuple(x), positions)

        velocities = velocities_arr.reshape((self.n_atoms, 3))
        #velocities = u.Quantity(map(lambda x: tuple(x), velocities), self.velocities_unit)
        velocities = map(lambda x: tuple(x), velocities)
        
        forces = forces_arr.reshape((self.n_atoms, 3))
        #forces = u.Quantity(map(lambda x: tuple(x), forces), self.forces_unit)
        forces = map(lambda x: tuple(x), forces)

        boxes = boxes_arr.reshape((3, 3))
        #boxes = u.Quantity(map(lambda x: tuple(x), boxes), self.boxes_unit)
        boxes = map(lambda x: tuple(x), boxes)
        
        params = dict(MonteCarloPressure=parameters_arr[0])
        
        time = time_arr[0]  # * self.time_unit

        state = mm.State(simTime=time, paramMap=params, energy=energy, coordList=positions, velList=velocities, forceList=forces, periodicBoxVectorsList=boxes)
        return state
