# environment.py
import os
import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
import csv
import time
import pygame
from config import WIDTH, HEIGHT
import logging

# Constants
WATER_LEVEL = 0.3  # Consistent with main.py and drone.py

# Colors
LAND_COLOR = (34, 139, 34)            # Land
BUILDING_COLOR = (139, 69, 19)        # Buildings
GRID_LINES_COLOR = (200, 200, 200)    # Grid lines
HIGHLIGHT_COLOR = (255, 255, 255)     # Highlight color for buildings
WATER_COLOR = (0, 0, 255)             # Blue color for water

blue_shades = [
    (0, 0, 139),      # Dark Blue
    (0, 0, 205),      # Medium Blue
    (0, 0, 255)       # Blue
]

class Environment:
    def __init__(self, grid_width=120, grid_height=80, num_islands=8, max_radius=10, num_buildings=50, load_from_file=True):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_islands = num_islands
        self.max_radius = max_radius
        self.num_buildings = num_buildings
        self.cell_size = 10 
        self.oil_spill = None  
        self.oil_spillage_manager = None  

        # Set up a module-specific logger
        self.logger = logging.getLogger('Environment')
        env_handler = logging.FileHandler('environment.log')
        env_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        env_handler.setFormatter(formatter)
        self.logger.addHandler(env_handler)
        self.logger.propagate = False  # Prevent log messages from being duplicated in main.log

        # Initialize log file for CSV
        self.log_file = 'environment_log.csv'
        self.initialize_logging()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(current_dir, "environment.npy")
        
                # Use the load_from_file parameter
        if load_from_file:
            loaded = self.load_environment(filename)
        else:
            loaded = None  # Force generation of a new environment

        if loaded is None:
            # If loading fails, generate a new grid
            self.grid = self.generate_grid()
            self.land_mask = self.grid > WATER_LEVEL  # Define land_mask before adding buildings
            self.buildings = self.add_buildings()
            self.save_environment([], filename='environment.npy')  # Save initial environment
        else:
            self.grid = loaded.get('grid', self.generate_grid())
            self.land_mask = self.grid > WATER_LEVEL  # Define land_mask before adding buildings
            self.buildings = loaded.get('buildings', self.add_buildings())

        self.print_grid_stats()

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

        # Forces some land for testing
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

    def get_grid(self):
        return self.grid

    def get_buildings(self):
        return self.buildings
    
    def add_oil_spill(self, oil_spill):
        """
        Add an oil spill to the environment.

        Parameters:
            oil_spill: An instance of OilSpillage.
        """
        self.oil_spill = oil_spill
        self.logger.info("Oil spill added to the environment.")

    def save_environment(self, drones, filename='environment.npy'):
        """
        Save the environment grid, buildings, and drone statuses to a NumPy file.

        Parameters:
            drones : List of Drone instances.
            filename : Name of the file to save the environment data.
        """
        environment_data = {
            'grid': self.grid,
            'buildings': self.buildings,
            'drones': [drone.to_dict() for drone in drones]
        }
        np.save(filename, environment_data)
        print(f"Environment saved to '{filename}'.")
        self.log_action('Save Environment', filename)

    def load_environment(self, filename='environment.npy'):
            environment_data = np.load(filename, allow_pickle=True).item()

            self.grid = environment_data.get('grid', self.generate_grid())
            self.land_mask = self.grid > WATER_LEVEL  # Define land_mask
            self.buildings = environment_data.get('buildings', self.add_buildings())


            print(f"Environment loaded from '{filename}'.")
            self.log_action("Environment Loaded", f"Loaded from '{filename}'")
            self.print_grid_stats()
            return environment_data  

    def set_oil_spillage_manager(self, oil_spillage_manager):
        self.oil_spillage_manager = oil_spillage_manager

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
        Log environment actions to the CSV file and the environment-specific log file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, action, details])
        self.logger.info(f"{action} - {details}")

    def draw(self, screen):
        """
        Draw the environment grid onto the provided Pygame screen.

        Parameters:
            screen : Pygame surface to draw the environment.
        """
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                elevation = self.grid[x, y]
                if elevation > WATER_LEVEL:
                    pygame.draw.rect(screen, LAND_COLOR, rect)
                    if self.buildings[x, y] > 0:
                        pygame.draw.rect(screen, BUILDING_COLOR, rect)
                else:
                    pygame.draw.rect(screen, WATER_COLOR, rect)


        for x in range(self.grid_width + 1):
            pygame.draw.line(
                screen, 
                GRID_LINES_COLOR, 
                (x * self.cell_size, 0), 
                (x * self.cell_size, self.grid_height * self.cell_size)
            )
        for y in range(self.grid_height + 1):
            pygame.draw.line(
                screen, 
                GRID_LINES_COLOR, 
                (0, y * self.cell_size), 
                (self.grid_width * self.cell_size, y * self.cell_size)
            )

    def handle_event(self, event):
        """
        Handle user interactions related to the environment.

        Parameters:
            event : Pygame event to handle.
        """
        if hasattr(self, 'editing') and self.editing:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                grid_x = mouse_x // self.cell_size
                grid_y = mouse_y // self.cell_size

                if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                    if self.land_mask[grid_x, grid_y]:
                        if self.buildings[grid_x, grid_y] == 0:
                            # Add a building
                            self.buildings[grid_x, grid_y] = random.uniform(1.0, 3.0)
                            print(f"Building added at ({grid_x}, {grid_y}).")
                            self.log_action("Add Building", f"Location: ({grid_x}, {grid_y})")
                        else:
                            # Remove the building
                            self.buildings[grid_x, grid_y] = 0.0
                            print(f"Building removed at ({grid_x}, {grid_y}).")
                            self.log_action("Remove Building", f"Location: ({grid_x}, {grid_y})")

    def update(self):
        """
        Update environment-related states if necessary.
        """
        pass

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        # Regenerate the grid and buildings
        self.grid = self.generate_grid()
        self.land_mask = self.grid > WATER_LEVEL
        self.buildings = self.add_buildings()
        self.print_grid_stats()
        print("Environment has been reset.")
        self.log_action("Environment Reset", "Reset to initial state.")

    def enable_editing(self):
        """
        Enable environment editing mode with a Pygame UI.
        """
        self.editing = True
        pygame.init()
        screen = pygame.display.set_mode((self.grid_width * self.cell_size, self.grid_height * self.cell_size))
        pygame.display.set_caption("Environment Editor")
        clock = pygame.time.Clock()
        
        running = True
        while running:
            dt = clock.tick(60) / 1000  # Delta time in seconds

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    self.handle_event(event)

            # Rendering
            screen.fill((0, 0, 0))  # Black background
            self.draw(screen)
            pygame.display.flip()

        self.save_environment([])
        pygame.quit()

# Execution Entry Point
if __name__ == "__main__":
    print("Running environment.py directly...")
    env = Environment()
    env.enable_editing()
