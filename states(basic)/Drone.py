import math
import random
import numpy as np
import csv


# Constants
WIDTH, HEIGHT = 1200, 800
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
WATER_LEVEL = 0.3
FPS = 60

class Drone:
    def __init__(self, id, position, rotor_speed=1.0, color=(255, 0, 0)):
        """
        Initialize the drone with an ID, initial position, rotor speed, and color.
        Position is a tuple (x, y).
        """
        self.id = id
        self.position = position  # (x, y) tuple
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.neighbors = []  # List of nearby drones for information sharing
        self.color = color  # Color for visualization
        
        # Logging attributes
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
        Move the drone by the given offsets in the x and y directions.
        Prevent collision with other drones by maintaining a minimum distance.
        """
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy
        
        # Check for collisions
        for drone in drones:
            if drone.id != self.id:
                distance = math.hypot(new_x - drone.position[0], new_y - drone.position[1])
                if distance < collision_radius:
                    print(f"Drone {self.id} collision detected with Drone {drone.id}. Movement aborted.")
                    self.log_action("Move Aborted", f"Collision with Drone {drone.id} at position {drone.position}")
                    return  # Abort movement to avoid collision
        
        # Update position
        self.position = (new_x, new_y)
        print(f"Drone {self.id} moved to {self.position}.")
        self.log_action("Move", f"Moved to {self.position}")
    
    def adjust_rotor_speed(self, new_speed):
        """
        Adjust the drone's rotor speed based on weather conditions.
        """
        old_speed = self.rotor_speed
        self.rotor_speed = max(0.1, min(new_speed, 5.0))  # Keep rotor speed between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted from {old_speed} to {self.rotor_speed}.")
        self.log_action("Rotor Speed Adjustment", f"Adjusted from {old_speed} to {self.rotor_speed}")
    
    def set_weather_forecast(self, forecast):
        """
        Update the weather forecast affecting the drone's operations.
        """
        old_forecast = self.weather_forecast
        self.weather_forecast = forecast
        print(f"Drone {self.id} weather forecast updated from '{old_forecast}' to '{self.weather_forecast}'.")
        self.log_action("Weather Update", f"Updated from '{old_forecast}' to '{self.weather_forecast}'")
        
        # Modify rotor speed based on weather
        if self.weather_forecast == "Windy":
            self.adjust_rotor_speed(self.rotor_speed * 1.2)  # Increase rotor speed by 20%
        elif self.weather_forecast == "Calm":
            self.adjust_rotor_speed(self.rotor_speed * 0.8)  # Decrease rotor speed by 20%
        # Add more weather conditions as needed
    
    def find_neighbors(self, drones, neighbor_radius=100):
        """
        Identify and update the list of neighboring drones within a specified radius.
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
        Share information with the nearest neighbors.
        """
        if self.neighbors:
            info = f"Drone {self.id}: Current position {self.position}"
            for neighbor in self.neighbors:
                neighbor.receive_information(info)
            print(f"Drone {self.id} shared information with {len(self.neighbors)} neighbors.")
            self.log_action("Information Sharing", f"Shared info with {len(self.neighbors)} neighbors")
        else:
            print(f"Drone {self.id} has no neighbors to share information with.")
            self.log_action("Information Sharing", "No neighbors to share information with")
    
    def receive_information(self, info):
        """
        Receive shared information from another drone.
        """
        print(f"Drone {self.id} received info: {info}")
        self.log_action("Information Received", f"Received info: {info}")
    
    def report_status(self):
        """
        Report the current status of the drone.
        """
        status = {
            "id": self.id,
            "position": self.position,
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast,
            "neighbors": [drone.id for drone in self.neighbors]
        }
        print(f"Drone {self.id} Status: {status}")
        self.log_action("Status Report", f"{status}")
        return status
    
    def load_environment(self, filename='environment.npy'):
        """
        Load the environment data from a .npy file.
        """
        try:
            environment_data = np.load(filename, allow_pickle=True).item()
            self.environment = environment_data
            print(f"Drone {self.id} loaded environment data from '{filename}'.")
            self.log_action("Environment Load", f"Loaded data from '{filename}'")
        except Exception as e:
            print(f"Drone {self.id} failed to load environment from '{filename}': {e}")
            self.log_action("Environment Load Failed", f"Error: {e}")
            self.environment = None
    
    def get_environment_info(self):
        """
        Retrieve specific information from the loaded environment.
        """
        if self.environment is None:
            print(f"Drone {self.id} has no environment data loaded.")
            self.log_action("Get Environment Info", "No environment data loaded")
            return None
        
        grid = self.environment.get('grid', None)
        buildings = self.environment.get('buildings', None)
        
        if grid is not None and buildings is not None:
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
            print(f"Drone {self.id} environment data incomplete.")
            self.log_action("Environment Info Retrieval Failed", "Incomplete environment data")
            return None
