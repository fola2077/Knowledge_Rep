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
        self.log_file = 'drone_stats.csv'
        self.initialize_logging()

        # Set up a module-specific logger
        self.logger = logging.getLogger(f'Drone_{self.id}')
        drone_handler = logging.FileHandler(f'drone_{self.id}.log')
        drone_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        drone_handler.setFormatter(formatter)
        self.logger.addHandler(drone_handler)
        self.logger.propagate = False  # Prevent log messages from being duplicated in main.log

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
        # self.logger.info(f"{action} - {details}")

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
        # Calculate the movement vector scaled by rotor speed
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

    def update_behavior(self):
        """
        Update drone behavior based on current weather conditions.
        """
        current_weather = self.weather_system.get_current_weather()

        if current_weather is None:
            print(f"Drone {self.id} has no current weather data.")
            # self.log_action("Behavior Update Failed", "No current weather data")
            return

        # Adjust rotor speed based on wind speed and precipitation
        wind_speed = current_weather.wind_speed
        precipitation = current_weather.precipitation_type
        intensity = current_weather.intensity

            
        # Adjust behavior based on adverse weather
        if self.is_weather_adverse():
            # For instance, the drone might decide to return to land
            self.return_to_land()


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
        # self.log_action("Neighbor Update", f"Found {len(self.neighbors)} neighbors")

    def share_information(self):
        """
        Share information with neighbors.
        """
        if self.neighbors:
            info = f"Drone {self.id}: Position {self.position}"
            for neighbor in self.neighbors:
                neighbor.receive_information(info)
            print(f"Drone {self.id} shared info with {len(self.neighbors)} neighbors.")
            self.log_action("Information Sharing", f"Shared info with {len(self.neighbors)} neighbors")
        else:
            print(f"Drone {self.id} has no neighbors to share information with.")
            # self.log_action("Information Sharing", "No neighbors to share information with")

    def receive_information(self, info):
        """
        Receive information from another drone.
        """
        print(f"Drone {self.id} received info: {info}")
        # self.log_action("Information Received", f"Received info: {info}")

    def is_weather_adverse(self):
        """
        Determines if the current weather conditions are adverse for the drone.
        Returns True if the weather is adverse, False otherwise.
        """
        current_weather = self.weather_system.get_current_weather()
        wind_speed = current_weather.wind_speed
        visibility = current_weather.visibility
        # Define thresholds (adjust these values based on your environment)
        high_wind_speed = 30.0  # Wind speed above which it's considered adverse
        low_visibility = 0.25   # Visibility below which it's considered adverse
        if wind_speed > high_wind_speed or visibility < low_visibility:
            return True
        return False

    def find_nearest_land_position(self):
        """
        Finds the nearest land cell to the drone's current position.
        Returns a Vector2 position representing the nearest land cell's world coordinates.
        """
        if self.environment is None:
            print("Environment not loaded.")
            return None

        grid_x = int(self.position.x // CELL_SIZE)
        grid_y = int(self.position.y // CELL_SIZE)

        min_distance_sq = None
        nearest_land_cell = None

        for x in range(self.environment.grid_width):
            for y in range(self.environment.grid_height):
                if self.environment.grid[x, y] > WATER_LEVEL:
                    dx = x - grid_x
                    dy = y - grid_y
                    distance_sq = dx * dx + dy * dy  # Squared distance
                    if min_distance_sq is None or distance_sq < min_distance_sq:
                        min_distance_sq = distance_sq
                        nearest_land_cell = (x, y)

        if nearest_land_cell is not None:
            # Convert grid cell back to world coordinates
            land_x = (nearest_land_cell[0] + 0.5) * CELL_SIZE
            land_y = (nearest_land_cell[1] + 0.5) * CELL_SIZE
            return Vector2(land_x, land_y)
        else:
            print("No land found in the environment.")
            return None

    def move_towards(self, target_position, max_distance):
        """
        Moves the drone towards the target_position by up to max_distance.
        Parameters:
            target_position: Vector2 representing the destination.
            max_distance: The maximum distance the drone can move in this step.
        """
        direction = target_position - self.position
        distance = direction.length()
        if distance > 0:
            movement = direction.normalize() * min(max_distance, distance)
            self.position += movement
        else:
            print(f"Drone {self.id} is already at the target position.")
            # Optionally, implement what should happen if the drone has reached land

    def return_to_land(self):
        """
        Moves the drone towards the nearest land position.
        """
        nearest_land_position = self.find_nearest_land_position()
        if nearest_land_position:
            # Define the maximum distance the drone can move in one step
            max_distance = 5.0
            self.move_towards(nearest_land_position, max_distance)
            print(f"Drone {self.id} moving towards land at position ({nearest_land_position.x:.2f}, {nearest_land_position.y:.2f}).")
            # Log the action if desired
            # self.log_action("Return to Land", f"Moving towards ({nearest_land_position.x:.2f}, {nearest_land_position.y:.2f})")

    def get_sensor_readings(self):
        """
        Simulates sensor readings for oil detection at the drone's current position.
        Returns a dictionary of sensor data.
        """
        sensor_data = {}

        # Ensure the oil_spillage_manager is available
        if self.oil_spillage_manager is None:
            print("Oil spillage manager not loaded.")
            sensor_data['oil_detected'] = False
            sensor_data['oil_concentration'] = 0.0
            return sensor_data

        # Assume the drone has an oil detection sensor
        grid_x = int(self.position.x // CELL_SIZE)
        grid_y = int(self.position.y // CELL_SIZE)
        
        if 0 <= grid_x < self.environment.grid_width and 0 <= grid_y < self.environment.grid_height:
            oil_concentration_grid = self.oil_spillage_manager.combined_oil_concentration()
            detected_concentration = oil_concentration_grid[grid_x, grid_y]
            
            # Simulate detection probability or sensor noise
            detection_probability = 1.0  # 90% chance to detect oil if present
            if random.random() < detection_probability:
                sensor_data['oil_detected'] = detected_concentration > 0
                sensor_data['oil_concentration'] = detected_concentration
                if sensor_data['oil_detected']:
        #             # Obtain current simulation time
        #             current_time = self.time_manager.current_sim_time
        #             # Mark the cell as detected in the oil spill with detection_time
        #             self.oil_spillage_manager.mark_cell_detected(grid_x, grid_y, current_time)
                    pass
            else:
                sensor_data['oil_detected'] = False
                sensor_data['oil_concentration'] = 0.0
        else:
            sensor_data['oil_detected'] = False
            sensor_data['oil_concentration'] = 0.0
        
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
            detection_radius = self.calculate_dynamic_radius() if dynamic_radius else 2
            detection_found = False
            current_time = self.time_manager.get_current_total_minutes()

            self.oil_spillage_manager.mark_cell_detected(grid_x, grid_y, current_time)
            detection_found = True
            for dx in range(-detection_radius, detection_radius + 1):
                for dy in range(-detection_radius, detection_radius + 1):
                    gx = grid_x + dx
                    gy = grid_y + dy
                    # Skip current cell (marked)
                    if dx == 0 and dy == 0 :
                        continue
                    if (0 <= gx < self.environment.grid_width and 
                        0 <= gy < self.environment.grid_height):
                        
                        concentration = self.oil_spillage_manager.combined_oil_concentration()[gx, gy]
                        
                        if concentration > detection_threshold:
                            self.oil_spillage_manager.mark_cell_detected(gx, gy, current_time)
            
            if detection_found:
                self.log_action('Oil Detected', f"Detected within radius {detection_radius}")
                return True
        return False

    def calculate_dynamic_radius(self):
        # Example: Increase radius under adverse weather
        if self.is_weather_adverse():
            return 3
        return 2