# ! This is an important comment (red)
# ? Why is this function here? (blue)
# TODO: Add a parameter for user input (orange)
# * Remember to test the API response (green)
# This is a regular comment (grey)
def example_function():
    pass


import pygame
import random
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter

class Drone:
    def __init__(self, id, position, battery=100, rotor_speed=1.0):
        """
        Initialize the drone with an ID, initial position, battery level, and rotor speed.
        """
        self.id = id
        self.position = position  # (x, y, z) tuple
        self.battery = battery  # Battery percentage
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.depth = 0  # Depth for potential underwater operations (optional)
        self.neighbors = []  # List of nearby drones for information sharing

    def move(self, dx, dy, dz):
        """
        Move the drone by the given offsets in the x, y, and z directions.
        """
        if self.battery <= 0:
            print(f"Drone {self.id} cannot move. Battery depleted!")
            return
        self.position = (
            self.position[0] + dx,
            self.position[1] + dy,
            self.position[2] + dz
        )
        self.battery -= self.calculate_battery_usage(dx, dy, dz)
        print(f"Drone {self.id} moved to {self.position}. Battery: {self.battery}%")

    def calculate_battery_usage(self, dx, dy, dz):
        """
        Calculate battery consumption based on distance moved and rotor speed.
        """
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        usage = distance * self.rotor_speed * 0.5  # Arbitrary battery usage rate
        return min(usage, self.battery)

    def adjust_rotor_speed(self, new_speed):
        """
        Adjust the drone's rotor speed.
        """
        self.rotor_speed = max(0.1, min(new_speed, 5.0))  # Keep rotor speed between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted to {self.rotor_speed}.")

    def set_weather_forecast(self, forecast):
        """
        Update the weather forecast affecting the drone's operations.
        """
        self.weather_forecast = forecast
        print(f"Drone {self.id} weather forecast updated to {forecast}.")

    def share_information(self):
        """
        Share information with the nearest neighbors. (Commented out for simplicity)
        """
        # if self.neighbors:
        #     for neighbor in self.neighbors:
        #         neighbor.receive_information(f"Drone {self.id}: Current position {self.position}.")
        print(f"Drone {self.id} would share information with neighbors: {self.neighbors}")

    def receive_information(self, info):
        """
        Receive shared information from another drone.
        """
        print(f"Drone {self.id} received info: {info}")

    def report_status(self):
        """
        Report the current status of the drone.
        """
        return {
            "id": self.id,
            "position": self.position,
            "battery": self.battery,
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast,
            "depth": self.depth
        }


class GridSimulation:
    def __init__(self, width=800, height=800, cell_size=10, num_islands=8, max_radius=10, num_buildings=50):
        """
        Initialize the simulation with grid dimensions, terrain features, and randomization settings.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid_width = width // cell_size
        self.grid_height = height // cell_size
        self.num_islands = num_islands
        self.max_radius = max_radius
        self.num_buildings = num_buildings
        self.water_level = 0.3  # Default water level
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Grid Simulation")


        # Default grid and features
        self.grid = None
        self.buildings = None
        self.land_mask = None

        # Generate grid and features
        # self.grid = self.generate_grid()
        # self.land_mask = self.grid > self.water_level
        # self.buildings = self.add_buildings()

    def generate_grid(self):
        """
        Generate a terrain grid with islands using Gaussian smoothing.
        """
        grid = np.zeros((self.grid_width, self.grid_height))

        # Uncomment this block to enable random island generation
        for _ in range(self.num_islands):
            cx, cy = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            radius = random.randint(3, self.max_radius)
            for x in range(self.grid_width):
                for y in range(self.grid_height):
                    if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < radius:
                        grid[x, y] += random.uniform(0.5, 1.0)

        # Smooth terrain
        grid = gaussian_filter(grid, sigma=1)
        return grid

    def add_buildings(self):
        """
        Add buildings to land areas in the grid.
        """
        buildings = np.zeros_like(self.grid)
        land_indices = np.argwhere(self.land_mask)

        for _ in range(self.num_buildings):
            if len(land_indices) == 0:
                break
            x, y = random.choice(land_indices)
            buildings[x, y] = random.uniform(1.0, 3.0)

        return buildings
    
    def save_environment(self, filename="environment.npy"):
        """
        Save the current grid to a file for reuse.
        """
        np.save(filename, self.grid)
        print(f"Environment saved to {filename}.")

    def load_environment(self, filename="environment.npy"):
        """
        Load a fixed grid from a file.
        """
        self.grid = np.load(filename)
        self.land_mask = self.grid > self.water_level
        self.buildings = self.add_buildings()
        print(f"Environment loaded from {filename}.")


    def draw_grid(self):
        """
        Draw the terrain grid, including water, land, and buildings.
        """
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                elevation = self.grid[x, y]

                if elevation > self.water_level:
                    pygame.draw.rect(self.screen, (34, 139, 34), rect)  # Land color
                    if self.buildings[x, y] > 0:
                        building_rect = pygame.Rect(
                            x * self.cell_size + self.cell_size // 4,
                            y * self.cell_size + self.cell_size // 4,
                            self.cell_size // 2,
                            self.cell_size // 2
                        )
                        pygame.draw.rect(self.screen, (139, 69, 19), building_rect)  # Building color
                elif elevation > self.water_level - 0.1:
                    pygame.draw.rect(self.screen, (100, 149, 237), rect)  # Shallow water color
                else:
                    pygame.draw.rect(self.screen, (70, 130, 180), rect)  # Deep water color

    def draw_grid_lines(self):
        """
        Draw grid lines for visual clarity.
        """
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y))

    def run(self, use_random=True, env_file="environment.npy"):
        """
        Run the main simulation loop, either generating a random environment or loading a fixed one.
        """
        clock = pygame.time.Clock()
        running = True

        if use_random:
            print("Generating a random environment...")
            self.grid = self.generate_grid()
            self.land_mask = self.grid > self.water_level
            self.buildings = self.add_buildings()
            self.save_environment(env_file)  # Save the random environment
        else:
            self.load_environment(env_file)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((0, 0, 0))
            self.draw_grid()
            self.draw_grid_lines()

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()




if __name__ == "__main__":
    simulation = GridSimulation()
    
    # Change use_random to False to load the saved environment
    simulation.run(use_random=True)  # Set to False to use fixed environment
