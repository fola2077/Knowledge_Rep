import pybullet as p
import numpy as np

class OilSpillage:
    def __init__(self, position, detection_radius=0.5):
        """
        Represents an oil spillage in the environment.

        :param position: Tuple (x, y) for the spillage location.
        :param detection_radius: Radius within which drones can detect the spillage.
        """
        self.position = np.array(position)  # Spillage location as a NumPy array
        self.detected = False  # Whether the spillage has been detected
        self.detection_radius = detection_radius
        self.visual_id = None  # PyBullet object ID for visualization

    def add_to_simulation(self):
        """
        Adds the spillage to the PyBullet simulation as a sphere.
        """
        self.visual_id = p.loadURDF(
            "sphere2.urdf", 
            [self.position[0], self.position[1], 0.1],  # Slightly above ground
            globalScaling=0.5  # Make the sphere small to represent a spillage
        )

    def check_detection(self, drone_position):
        """
        Checks if a drone has detected the spillage.

        :param drone_position: Position of the drone as a NumPy array (x, y).
        :return: True if detected, False otherwise.
        """
        if not self.detected:
            distance = np.linalg.norm(self.position - drone_position)
            if distance <= self.detection_radius:
                self.detected = True  # Mark the spillage as detected
                return True
        return False

    def reset(self):
        """
        Resets the spillage detection status.
        """
        self.detected = False

class DroneEnvironment:
    def __init__(self, num_drones=3, num_spillages=5):
        """
        Initializes the drone environment with drones and oil spillages.

        :param num_drones: Number of drones in the simulation.
        :param num_spillages: Number of oil spillages in the environment.
        """
        self.num_drones = num_drones
        self.num_spillages = num_spillages
        self.drones = []
        self.spillages = []
        self.client = p.connect(p.GUI)  # Start PyBullet GUI
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load URDF assets
        p.setGravity(0, 0, -9.8)  # Set gravity for the environment
        self.ground = p.loadURDF("plane.urdf")  # Load a flat ground plane
        self.initialize_spillages()
        self.initialize_drones()

    def initialize_drones(self):
        """Initialize drones as small cubes at random positions."""
        for _ in range(self.num_drones):
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            drone = p.loadURDF("cube_small.urdf", [x, y, 1])  # Place 1m above ground
            self.drones.append(drone)

    def initialize_spillages(self):
        """Initialize oil spillages at random positions."""
        for _ in range(self.num_spillages):
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            spillage = OilSpillage((x, y))
            spillage.add_to_simulation()
            self.spillages.append(spillage)

    def reset(self):
        """Reset drones and spillages to random positions."""
        for drone in self.drones:
            x, y = np.random.uniform(0, 5), np.random.uniform(0, 5)
            p.resetBasePositionAndOrientation(drone, [x, y, 1], [0, 0, 0, 1])
        for spillage in self.spillages:
            spillage.reset()

    def step(self, actions):
        """
        Update drone positions based on actions and check spillage detection.

        :param actions: List of actions for each drone.
        :return: State, rewards, done flag, and additional info.
        """
        rewards = []
        for i, action in enumerate(actions):
            pos, _ = p.getBasePositionAndOrientation(self.drones[i])
            x, y, z = pos

            # Move drone based on action
            if action == 1: y += 0.1  # Move Up
            elif action == 2: y -= 0.1  # Move Down
            elif action == 3: x -= 0.1  # Move Left
            elif action == 4: x += 0.1  # Move Right

            # Update drone position in simulation
            p.resetBasePositionAndOrientation(self.drones[i], [x, y, z], [0, 0, 0, 1])

            # Check for spillage detection
            drone_position = np.array([x, y])
            reward = 0
            for spillage in self.spillages:
                if spillage.check_detection(drone_position):
                    reward += 10  # Reward for detecting a spillage
            rewards.append(reward)

        return self.get_state(), rewards, False, {}

    def get_state(self):
        """Return current positions of all drones."""
        return [p.getBasePositionAndOrientation(drone)[0][:2] for drone in self.drones]

    def render(self):
        """Rendering is handled by PyBullet's GUI; no need for additional code here."""
        pass

    def close(self):
        """Disconnect PyBullet."""
        p.disconnect()

