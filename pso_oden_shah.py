"""

PARTICLE SWARM ALRGORITHM

CREATED BY NICK ODEN AND SAMAY SHAH

DATE OF CREATION: 02/26/24
DATE OF LAST MODIFICATION: 02/26/24

"""

# import modules
import numpy as np

# class to create a UAV
class UAV:

    # constructor method to initialize UAV
    def __init__(self, R_dim):

        # initial position and velocity of UAV, in R dim
        self.position = np.random.rand(R_dim)
        self.velocity = np.random.rand(R_dim)

        # personal best location of UAV
        self.pb = self.position.copy()

    # method to update UAV position
    def update_position(self):

        # not time based, so directly add velocity to position
        self.position += self.velocity

class PSO:

    # constructor method for PSO algorithm
    def __init__(self, n_uavs, R_dim, n_iterations):

        # defined by user
        self.n_uavs = n_uavs
        self.R_dim = R_dim
        self.n_iterations = n_iterations

        # initialize all UAVs used for search
        self.uavs = [UAV(R_dim) for _ in range(n_uavs)]

        # global best location of all UAVs in swarm
        self.gb = np.ones(R_dim)

        # define target location
        self.target = np.random.rand(R_dim)

        # set sensor weights for UAV search
        # sound - highest rating for if UAV picks up on movement sounds, target could be in the area
        # visual - lowest rating since trail could be cold and target could be long gone
        # heat - medium rating since heat signatures could be picked up from more than just target
        self.sensor_weights = {'sound': 0.5, 'visual': 0.3, 'heat': 0.4}

        ## add: fake targets (animals) to throw off search
        ## add: taget moves locations, but leaves a trail that dissipates
        ## add: better algorithm to determine this sensor data

    # method for sensing the target
    def sense_target(self, uav):

        # sensing distances
        sound_distance = np.linalg.norm(uav.position - self.target)
        visual_distance = np.linalg.norm(uav.position - self.target)
        heat_distance = np.linalg.norm(uav.position - self.target)
        
        # total distance determined by sensing weights in dictionary
        total_distance = (self.sensor_weights['sound'] * sound_distance +
                          self.sensor_weights['visual'] * visual_distance +
                          self.sensor_weights['heat'] * heat_distance)
        
        return total_distance
    
    # method for updating each UAV personal best
    def update_pb(self, uav):

        # compare current distance to personal best
        if self.sense_target(uav) < self.sense_target(uav.pb):

            # update
            uav.pb = uav.position.copy()

    # method for updating swarm global best
    def update_gb(self):

        # for loop for all UAVs in swarm
        for uav in self.uavs:

            # compare all UAV positions to global best position
            if self.sense_target(uav) < self.sense_target(self.gb):

                # update
                self.gb = uav.position.copy()

    # method for updating UAV velocity
    def update_velocity(self):

        # velocity parameters
        w = 0.5 # inertia
        phi_p = 1.5 # cognitive weight
        phi_g = 1.5 # social weight

        # for loop for all UAVs in swarm
        for uav in self.uavs:

            # calculate new UAV velocity
            cognitive = phi_p * np.random.rand(self.R_dim) * (uav.pb - uav.position)
            social = phi_g * np.random.rand(self.R_dim) * (self.gb - uav.position)
            uav.velocity = w * uav.velocity + cognitive + social

            # update UAV position
            uav.update_position()

    # method to run the PSO algorithm
    def run_sim(self):

        # for loop for all iterations
        for _ in range(self.n_iterations):

            # for loop for all UAVs in swarm
            for uav in self.uavs:

                # update personal best
                self.update_pb(uav)
            
            # update global best
            self.update_gb()

            # update velocity of all UAVs, then repeat
            self.update_velocity()

        return self.gb, self.sense_target(self.gb), self.target
    
# values for PSO sim
n_uavs = 10
R_dim = 2
n_iterations = 100

# run PSO sim
pso = PSO(n_uavs, R_dim, n_iterations)
best_position, sensor_data, target_position = pso.run_sim()

# print
print("Best position:", best_position)
print("Target position:", target_position)
print("Total sensor data:", sensor_data)
