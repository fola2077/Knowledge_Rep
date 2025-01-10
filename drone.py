# drone.py

import math
import random
import numpy as np
import csv
import time
import pygame
from pygame.math import Vector2
from config import WIDTH, HEIGHT, CELL_SIZE
from environment import WATER_LEVEL
from dqn_agent import DQNAgent
from oilspillage import OilSpill, OilSpillage
import logging

class Drone:
    def __init__(self, id, position, environment, weather_system, time_manager, color=(255, 0, 0)):
        """
        Initialize the drone with an ID, position, rotor speed, color, and reference to WeatherSystem.
        """
        self.id = id
        self.position = Vector2(position)  # (x, y) tuple
        self.movement_speed = 10.0  # Movement speed in pixels per frame
        self.color = color
        self.neighbors = []
        self.environment = environment 
        self.oil_spillage_manager = None # To be loaded later
        self.weather_system = weather_system  # Reference to WeatherSystem
        self.time_manager = time_manager
        self.visited_cells = set()
        self.detected_cells = set()
        self.time_since_last_detection = 0
        self.log_file = 'drone_stats.csv'
        self.initialize_logging()

        # Initialize DroneAgent
        state_dim = 6 # As defined in DroneEnv
        action_dim = 5 
        self.agent = DQNAgent(
            drone_id=id,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            buffer_capacity=50000,
            batch_size=128,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=10000
        )

        # For storing previous state
        self.prev_state = None
        self.prev_action = None

        # Set up a module-specific logger
        self.logger = logging.getLogger(f'Drone_{self.id}')
        drone_handler = logging.FileHandler(f'drone_{self.id}.log')
        drone_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        drone_handler.setFormatter(formatter)
        self.logger.addHandler(drone_handler)
        self.logger.propagate = False  

    def reset(self):
        # New attributes for tracking actions
        self.visited_cells.clear()
        self.detected_cells.clear()
        self.scan_mode = False
        self.time_since_last_detection = 0

    def step(self, action, reward, done, next_state):
        """Store transition and train the agent."""
        if self.prev_state is not None:
            self.agent.push_transition(self.prev_state, self.prev_action, reward, next_state, done)
            self.agent.update()
        
        # Update the previous state and action
        self.prev_state = next_state
        self.prev_action = action

    def act(self, state):
        """Select an action based on the current state."""
        return self.agent.select_action(state)

    def update_target_network(self):
        """Update the target network."""
        self.agent.update_target_network()

    def initialize_logging(self):
        """
        Initialize the CSV log file with headers if it doesn't exist.
        """
        try:
            with open(self.log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Drone_ID', 'Position_X', 'Position_Y'])
        except FileExistsError:
            # File already exists
            pass

    def log_action(self, action, details):
        """
        Log drone actions to a CSV file and the drone-specific log file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        if action == "Move":
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Extract position data from details
                try:
                    position_str = details.split("Moved to ")[1].strip("()")
                    pos_x, pos_y = position_str.split(", ")
                    writer.writerow([timestamp, self.id, float(pos_x), float(pos_y)])
                except IndexError:
                    # Handle unexpected format
                    writer.writerow([timestamp, self.id, "N/A", "N/A"])
        elif action == "Oil Detected":
            with open(self.log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, self.id, "Oil Detected", details])

    def to_dict(self):
        """
        Convert the drone's state to a dictionary for serialization.
        """
        return {
            'id': self.id,
            'position': (self.position.x, self.position.y),
            # Add other relevant attributes if necessary
        }

    @classmethod
    def from_dict(cls, data, environment, weather_system, time_manager):
        position = data['position']
        drone = cls(
            id=data['id'],
            position=position,
            environment=environment,
            weather_system=weather_system,
            time_manager=time_manager,
        )
        return drone

    def move(self, dx, dy, drones, collision_radius=10):
        """
        Moves the drone by (dx, dy) while checking for collisions with other drones.
        """

        
        # Calculate the movement vector scaled by speed
        direction = Vector2(dx, dy)
        if direction.length() > 0:
            movement = direction.normalize() * self.movement_speed
        else:
            movement = Vector2(0, 0)
        # Calculate the proposed new position
        proposed_position = self.position + movement

        # Boundary checking: Clamp the proposed position within the screen limits
        proposed_position.x = max(0, min(WIDTH, proposed_position.x))
        proposed_position.y = max(0, min(HEIGHT, proposed_position.y))

        # Collision checking: Ensure no other drone is within the collision radius
        for drone in drones:
            if drone.id != self.id:
                distance = proposed_position.distance_to(drone.position)
                if distance < collision_radius:
                    print(f"Drone {self.id} collision detected with Drone {drone.id}. Movement aborted.")
                    self.log_action("Move Aborted", f"Collision with Drone {drone.id} at position ({drone.position.x:.2f}, {drone.position.y:.2f})")
                    return  # Abort movement if collision is detected

        # No collision detected; update the drone's position
        self.position = proposed_position
        self.log_action("Move", f"Moved to ({self.position.x:.2f}, {self.position.y:.2f})")

        # After moving, reset scan_mode if moving
        self.scan_mode = False


    def update_behavior(self):
        """
        Update drone behavior based on current weather conditions.
        """
        current_weather = self.weather_system.get_current_weather()

        if current_weather is None:
            print(f"Drone {self.id} has no current weather data.")
            return


    def find_neighbors(self, drones, neighbor_radius=100):
        """
        Find drones within neighbor_radius and update neighbors list.
        """
        self.neighbors = []
        for drone in drones:
            if drone.id != self.id:
                distance = math.hypot(self.position.x - drone.position.x, self.position.y - drone.position.y)
                if distance <= neighbor_radius:
                    self.neighbors.append(drone)
        print(f"Drone {self.id} has {len(self.neighbors)} neighbors.")

    def share_information(self):
        if self.neighbors:
            new_info = {
                "detected_cells": self.detected_cells
            }
            for neighbor in self.neighbors:
                neighbor.receive_information(new_info)
            print(f"Drone {self.id} shared detection info with {len(self.neighbors)} neighbors.")

    def receive_information(self, info):
        # Merge the info into our own records
        if "detected_cells" in info:
            for cell in info["detected_cells"]:
                self.detected_cells.add(cell)
        print(f"Drone {self.id} received new detection info from another drone.")

    def get_weather_penalty(self):
        """
        Returns a penalty score based on multiple factors:
        wind speed, precipitation, cloud density, and also
        special conditions (Foggy, Harmattan).
        """
        current_weather = self.weather_system.get_current_weather()
        if not current_weather:
            return 0.0

        # --- Extract Weather Attributes ---
        wind_speed = current_weather.wind_speed
        precipitation_type = current_weather.precipitation_type  # e.g. "Rain", "None", "Thunderstorm"
        cloud_density = current_weather.cloud_density
        weather_name = current_weather.name  # e.g. "Foggy", "Harmattan", "Sunny", "Stormy", etc.

        # --- Converts Precipitation to Numeric ---
        precip_map = {
            'None': 0,
            'Rain': 1,
            'Thunderstorm': 2,
            '': 0  # fallback if blank
        }
        precip_value = precip_map.get(precipitation_type, 0)

        # --- Weighted Sum for Basic Factors ---
        w_wind = 0.3
        w_precip = 0.5
        w_cloud = 0.2

        raw_score = (w_wind * wind_speed) + (w_precip * precip_value) + (w_cloud * cloud_density)

        # --- Additional Penalties for Foggy / Harmattan ---
        if weather_name == "Foggy":
            # Fog can be pretty bad, especially if humidity is high
            raw_score += 1.0
        elif weather_name == "Harmattan":
            # Dust haze
            raw_score += 1.5

        # The higher the raw_score, the worse the weather.
        return raw_score

    @staticmethod
    def logistic(x):
        """
        Simple logistic function: Maps (-∞..∞) -> (0..1).
        """
        return 1 / (1 + math.exp(-x))

    def get_weather_modifier(self):
        """
        Takes the raw weather penalty and converts it to a [0..1] multiplier
        using a logistic curve. Higher penalty => smaller multiplier.
        """
        raw_score = self.get_weather_penalty()

        # factor = logistic(-a * (raw_score - b))
        # 'a' is slope, 'b' is offset
        a = 1.0  # slope
        b = 3.0  # offset

        factor = Drone.logistic(-a * (raw_score - b))
        return factor
    
    def thickness_factor(self, oil_concentration):
        """
        Non-linear scale for oil thickness from 0..1 => factor 0..1
        Example: logistic or exponential approach 
        so that very low concentration yields a big penalty.
        """
        return 1.0 - math.exp(-5 * oil_concentration)


    def get_sensor_readings(self):
        """
        Simulates sensor readings for oil detection at the drone's current position.
        Returns a dictionary of sensor data.
        """
        sensor_data = {
            'oil_detected': False,
            'oil_concentration': 0.0
        }

        # Ensure the oil_spillage_manager is available
        if self.oil_spillage_manager is None:
            print("Oil spillage manager not loaded.")
            return sensor_data

        # Assume the drone has an oil detection sensor
        grid_x = int(self.position.x // CELL_SIZE)
        grid_y = int(self.position.y // CELL_SIZE)
        
        if not (0 <= grid_x < self.environment.grid_width and 0 <= grid_y < self.environment.grid_height):
            return sensor_data
        
        oil_concentration_grid = self.oil_spillage_manager.combined_oil_concentration()
        detected_concentration = oil_concentration_grid[grid_x, grid_y]

                # Base probability
        base_probability = 1.0  
        weather_factor = self.get_weather_modifier()      # logistic-based
        thickness_factor = self.thickness_factor(detected_concentration)

        # Combine them multiplicatively 
        detection_probability = base_probability * weather_factor * thickness_factor

        # clamp to [0..1]
        detection_probability = max(0.0, min(1.0, detection_probability))

        if random.random() < detection_probability:
            sensor_data['oil_detected'] = detected_concentration > 0
            sensor_data['oil_concentration'] = detected_concentration

        return sensor_data

    def get_environmental_sensors(self):
        """
        Simulates environmental sensors such as wind speed and visibility.
        Returns a dictionary of environmental sensor data.
        """
        sensor_data = {}
        current_weather = self.weather_system.get_current_weather()
        
        # Simulate sensor readings with potential noise
        sensor_data['wind_speed'] = current_weather.wind_speed + random.gauss(0, 0.5)
        sensor_data['visibility'] = current_weather.visibility + random.gauss(0, 0.05)
        sensor_data['precipitation'] = current_weather.precipitation_type
        
        return sensor_data  
    
    def reset_sensors(self):
        """
        Resets sensor readings to their initial state. Useful when resetting the environment.
        """
        # Reset any sensor state if necessary
        pass  # If your sensors don't maintain state, this can be left empty

    def report_status(self):
        """
        Report current status.
        """
        status = {
            "id": self.id,
            "position": self.position,
            "weather_forecast": self.weather_system.current_state.name if self.weather_system.current_state else None,
            "neighbors": [drone.id for drone in self.neighbors]
        }
        print(f"Drone {self.id} Status: {status}")
        # self.log_action("Status Report", f"{status}")
        return status

    def load_environment(self, environment, oil_spillage_manager):
        """
        Load and interact with environment data.
        """
        self.environment = environment
        self.oil_spillage_manager = oil_spillage_manager
        print(f"Drone {self.id} loaded environment data.")
        # self.log_action("Environment Load", f"Loaded environment data")

    def get_environment_info(self):
        """
        Retrieve and log information from environment.
        """
        if self.environment is not None:
            grid = self.environment.get_grid()
            buildings = self.environment.get_buildings()
            env_info = {
                "grid_min": grid.min(),
                "grid_max": grid.max(),
                "grid_mean": grid.mean(),
                "total_land_cells": np.sum(grid > WATER_LEVEL),
                "total_buildings": np.sum(buildings > 0)
            }
            print(f"Drone {self.id} Environment Info: {env_info}")
            # self.log_action("Environment Info Retrieved", f"{env_info}")
            return env_info
        else:
            print(f"Drone {self.id} has no environment data loaded.")
            # self.log_action("Environment Info Retrieval Failed", "No environment data loaded")
            return None

    def detect_oil(self, detection_threshold=0.01, dynamic_radius=True):
        sensor_data = self.get_sensor_readings()
        if sensor_data['oil_detected']:
            grid_x = int(self.position.x // CELL_SIZE)
            grid_y = int(self.position.y // CELL_SIZE)
            self.detected_cells.add((grid_x, grid_y))
            detection_radius = self.calculate_dynamic_radius() if dynamic_radius else 2
            detection_found = False
            current_time = self.time_manager.get_current_total_minutes()

            detection_delay = self.oil_spillage_manager.get_time_since_spill(self.position)
            if detection_delay is None:
                detection_delay = 0.0

            self.oil_spillage_manager.mark_cell_detected(grid_x, grid_y, current_time)
            detection_found = True
            for dx in range(-detection_radius, detection_radius + 1):
                for dy in range(-detection_radius, detection_radius + 1):
                    gx = grid_x + dx
                    gy = grid_y + dy

                    if (0 <= gx < self.environment.grid_width and 
                        0 <= gy < self.environment.grid_height):
                        
                        concentration = self.oil_spillage_manager.combined_oil_concentration()[gx, gy]
                        
                        if concentration > detection_threshold:
                            self.oil_spillage_manager.mark_cell_detected(gx, gy, current_time)
            
            if detection_found:
                self.log_action('Oil Detected', f"Detected within radius {detection_radius}")
                return True , detection_delay
        return False, None

    def calculate_dynamic_radius(self):
        # Example: Increase radius under adverse weather
        return 2
    
    def scan_for_oil(self, detection_threshold=0.01, dynamic_radius=True, frames=4):
        """
        Initiates scanning mode for oil detection.
        """
        self.scan_mode = True
        oil_detected = False
        scan_radius = 2 if dynamic_radius else 1
        
        for frame in range(frames):
            #Perform a single scan per frame
            detected = self.perform_scan(detection_threshold, scan_radius)
            oil_detected = oil_detected or detected
    

    def perform_scan(self, detection_threshold, scan_radius):
        """
        Actual logic for scanning oil without causing recursion.
        """
        oil_detected = False
        x, y = int(self.position.x), int(self.position.y)

        # Define scan boundaries
        x_min = max(x - scan_radius, 0)
        x_max = min(x + scan_radius + 1, self.environment.grid_width)
        y_min = max(y - scan_radius, 0)
        y_max = min(y + scan_radius + 1, self.environment.grid_height)

        # Extract the scan area
        scan_area = self.oil_spillage_manager.combined_oil_concentration()[x_min:x_max, y_min:y_max]

        # Create coordinate grids
        dx = np.arange(x_min, x_max) - x
        dy = np.arange(y_min, y_max) - y
        dx_grid, dy_grid = np.meshgrid(dx, dy, indexing='ij')
        distance = np.sqrt(dx_grid**2 + dy_grid**2)

        # Create mask for circular scan
        mask = distance <= scan_radius

        # Apply mask and detection threshold
        detected = (scan_area > detection_threshold) & mask

        # Find detected cell indices
        detected_indices = np.argwhere(detected)

        # Add detected cells to detected_cells set
        for idx in detected_indices:
            gx = x_min + idx[0]
            gy = y_min + idx[1]
            self.detected_cells.add((gx, gy))
            oil_detected = True

        # Logging
        if oil_detected:
            print(f"Drone {self.id} detected oil at {[tuple(cell) for cell in detected_indices + [x_min, y_min]]}")

        return oil_detected
