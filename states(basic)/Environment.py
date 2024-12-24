# ENVIRONMENT CLASS

import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
import sys
import csv
import time
from config import WIDTH, HEIGHT

# Constants
WATER_LEVEL = 0.3 # Ensure consistency with drone.py

# Cell size and Frames Per Second
CELL_SIZE = 10
FPS = 60

# Colors
LAND_COLOR = (34, 139, 34)            # Land
BUILDING_COLOR = (139, 69, 19)        # Buildings
GRID_LINES_COLOR = (200, 200, 200)    # Grid lines
HIGHLIGHT_COLOR = (255, 255, 255)     # Highlight color for buildings
DRONE_COLOR = (255, 0, 0)             # Red color for drones
INFO_COLOR = (255, 255, 255)          # White color for info text
WATER_COLOR = (0, 0, 255)             # Blue color for water

blue_shades = [
    (0, 0, 139),      # Dark Blue
    (0, 0, 205),      # Medium Blue
    (0, 0, 255)       # Blue
]


class Environment:
    def __init__(self, grid_width=120, grid_height=80, num_islands=8, max_radius=10, num_buildings=50):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_islands = num_islands
        self.max_radius = max_radius
        self.num_buildings = num_buildings
        self.grid = self.generate_grid()
        self.land_mask = self.grid > WATER_LEVEL
        self.buildings = self.add_buildings()
        self.log_file = 'environment_log.csv'
        self.initialize_logging()
        self.print_grid_stats()
        self.drones = []
    
    def generate_grid(self):
        grid = np.zeros((self.grid_width, self.grid_height))
    
        # Create islands
        for _ in range(self.num_islands):
            cx, cy = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            radius = random.randint(3, self.max_radius)
            for x in range(cx - radius, cx + radius + 1):
                for y in range(cy - radius, cy + radius + 1):
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if distance < radius:
                            grid[x, y] += random.uniform(0.5, 1.0)
        
        # Force some land for testing
        for i in range(10):
            if i < self.grid_width and i < self.grid_height:
                grid[i, i] = WATER_LEVEL + 0.5  # Ensure these cells are land
    
        # Smooth terrain
        grid = gaussian_filter(grid, sigma=1)
        return grid
    
    def add_buildings(self):
        buildings = np.zeros_like(self.grid)
        land_indices = np.argwhere(self.land_mask)
    
        for _ in range(self.num_buildings):
            if len(land_indices) == 0:
                break
            x_idx, y_idx = random.choice(land_indices)
            buildings[x_idx, y_idx] = random.uniform(1.0, 3.0)
    
        return buildings
    
    def draw(self, screen):
        """
        Drawing logic handled in main.py for centralized rendering.
        """
        pass
    
    def draw_grid_lines(self, screen, cell_size=10):
        """
        Drawing grid lines handled in main.py.
        """
        pass
    
    def save_environment(self,drones, filename='environment.npy'):
        environment_data = {
            'grid': self.grid,
            'buildings': self.buildings,
            'drones': [drone.report_status() for drone in drones]
        }
        np.save(filename, environment_data)
        print(f"Environment saved to '{filename}'.")
        self.log_action('Save Environment', filename)
    
    def load_environment(self,drones, filename='environment.npy'):
        try:
            environment_data = np.load(filename, allow_pickle=True).item()
            self.grid = environment_data['grid']
            self.buildings = environment_data['buildings']
            self.land_mask = self.grid > WATER_LEVEL
            print(f"Environment loaded from '{filename}'.")
            self.log_action("Environment Loaded", f"Loaded from '{filename}'")

            # Load drones
            drone_statuses = environment_data.get('drones', [])
            drones.clear()  # Remove existing drones
            from drone import Drone  # Import here to avoid circular import

            for status in drone_statuses:
                drone = Drone(
                    id=status['id'],
                    position=status['position'],
                    rotor_speed=status['rotor_speed'],
                    color=(255, 0, 0)  # Default color; adjust as needed
                )
                drone.weather_forecast = status['weather_forecast']
                drones.append(drone)
                print(f"Drone {drone.id} loaded at position {drone.position}.")
                drone.log_action("Drone Loaded", f"Loaded at position {drone.position}")

        except Exception as e:
            print(f"Failed to load environment from '{filename}': {e}")
            # If loading fails, generate a new grid
            self.grid = self.generate_grid()
            self.land_mask = self.grid > WATER_LEVEL
            self.buildings = self.add_buildings()
            self.print_grid_stats()
    
    def print_grid_stats(self):
        print("=== Grid Statistics ===")
        print(f"Min elevation: {self.grid.min():.2f}")
        print(f"Max elevation: {self.grid.max():.2f}")
        print(f"Mean elevation: {self.grid.mean():.2f}")
        print(f"Total land cells: {np.sum(self.land_mask)}")
        print(f"Total buildings: {np.sum(self.buildings > 0)}")
        print("========================")
        self.log_action("Grid Stats Printed", "Printed grid statistics to console.")
    
    def initialize_logging(self):
        """
        Initialize the environment CSV log file with headers if it doesn't exist.
        """
        try:
            with open(self.log_file, mode='x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Action', 'Details'])
        except FileExistsError:
            # File already exists
            pass

    def log_action(self, action, details):
        """
        Log environment actions to the CSV file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, action, details])


