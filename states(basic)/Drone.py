# drone.py

import pygame
import math
import random
import numpy as np
import csv
import time

# Constants
WATER_LEVEL = 0.3  # Ensure consistency with environment.py

class Drone:
    def __init__(self, id, position, weather_system, rotor_speed=1.0, color=(255, 0, 0)):
        """
        Initialize the drone with an ID, position, rotor speed, color, and reference to WeatherSystem.
        """
        self.id = id
        self.position = pygame.math.Vector2(position) # (x, y) tuple
        self.rotor_speed = rotor_speed
        self.color = color
        self.neighbors = []
        self.environment = None  # To be loaded later
        self.weather_system = weather_system  # Reference to WeatherSystem
        self.log_file = 'drone_stats.csv'
        self.initialize_logging()

    def initialize_logging(self):
        """
        Initialize the CSV log file with headers if it doesn't exist.
        """
        try:
            with open(self.log_file, mode='x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Drone_ID', 'Action', 'Details'])
        except FileExistsError:
            # File already exists
            pass

    def log_action(self, action, details):
        """
        Log drone actions to a CSV file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, self.id, action, details])

    def move(self, dx, dy, drones, collision_radius=10):
        """
        Move the drone by dx and dy, ensuring no collision with other drones.
        """
        # Update position with delta movement
        self.position.x += dx * self.rotor_speed
        self.position.y += dy * self.rotor_speed
        
        # Boundary checking to keep drone within screen
        self.position.x = max(0, min(WIDTH, self.position.x))
        self.position.y = max(0, min(HEIGHT, self.position.y)) 


        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Check for collision
        for drone in drones:
            if drone.id != self.id:
                distance = math.hypot(new_x - drone.position[0], new_y - drone.position[1])
                if distance < collision_radius:
                    print(f"Drone {self.id} collision detected with Drone {drone.id}. Movement aborted.")
                    self.log_action("Move Aborted", f"Collision with Drone {drone.id} at position {drone.position}")
                    return  # Abort movement

        # Update position
        self.position = (new_x, new_y)
        print(f"Drone {self.id} moved to {self.position}.")
        self.log_action("Move", f"Moved to {self.position}")

    def adjust_rotor_speed(self, factor):
        """
        Adjust the rotor speed based on a factor.
        """
        old_speed = self.rotor_speed
        self.rotor_speed = max(0.1, min(self.rotor_speed * factor, 5.0))  # Clamp between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted from {old_speed:.2f} to {self.rotor_speed:.2f}.")
        self.log_action("Rotor Speed Adjustment", f"Adjusted from {old_speed:.2f} to {self.rotor_speed:.2f}")

    def update_behavior(self):
        """
        Update drone behavior based on current weather conditions.
        """
        current_weather = self.weather_system.get_current_weather()

        if current_weather is None:
            print(f"Drone {self.id} has no current weather data.")
            self.log_action("Behavior Update Failed", "No current weather data")
            return

        # Adjust rotor speed based on wind speed and precipitation
        wind_speed = current_weather.wind_speed
        precipitation = current_weather.precipitation_type
        intensity = current_weather.intensity

        # Simple logic: Increase rotor speed in high wind or precipitation
        if wind_speed > 20 or precipitation in ["Rain", "Stormy"]:
            self.adjust_rotor_speed(1.1)  # Increase by 10%
        elif wind_speed < 5 and precipitation == "None":
            self.adjust_rotor_speed(0.9)  # Decrease by 10%
        else:
            # Slight adjustment based on intensity
            self.adjust_rotor_speed(1 + 0.05 * (intensity - 0.5))

    def find_neighbors(self, drones, neighbor_radius=100):
        """
        Find drones within neighbor_radius and update neighbors list.
        """
        self.neighbors = []
        for drone in drones:
            if drone.id != self.id:
                distance = math.hypot(self.position[0] - drone.position[0],
                                      self.position[1] - drone.position[1])
                if distance <= neighbor_radius:
                    self.neighbors.append(drone)
        print(f"Drone {self.id} has {len(self.neighbors)} neighbors.")
        self.log_action("Neighbor Update", f"Found {len(self.neighbors)} neighbors")

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
            self.log_action("Information Sharing", "No neighbors to share information with")

    def receive_information(self, info):
        """
        Receive information from another drone.
        """
        print(f"Drone {self.id} received info: {info}")
        self.log_action("Information Received", f"Received info: {info}")

    def report_status(self):
        """
        Report current status.
        """
        status = {
            "id": self.id,
            "position": self.position,
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_system.current_state.name if self.weather_system.current_state else None,
            "neighbors": [drone.id for drone in self.neighbors]
        }
        print(f"Drone {self.id} Status: {status}")
        self.log_action("Status Report", f"{status}")
        return status

    def load_environment(self, environment):
        """
        Load and interact with environment data.
        """
        self.environment = environment
        print(f"Drone {self.id} loaded environment data.")
        self.log_action("Environment Load", f"Loaded environment data")

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
            self.log_action("Environment Info Retrieved", f"{env_info}")
            return env_info
        else:
            print(f"Drone {self.id} has no environment data loaded.")
            self.log_action("Environment Info Retrieval Failed", "No environment data loaded")
            return None
