# IMPORTING LIBRARIES

import pygame
import random
import math
import time
import numpy as np
from scipy.ndimage import gaussian_filter
import csv
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Niger Delta Weather Simulation for Drone Oil Spill Detection")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors for Different Terrain and Drones
DAY_COLOR = (135, 206, 235)
NIGHT_COLOR = (25, 25, 112)
CLOUD_COLOR = (220, 220, 220)
RAIN_COLOR = (0, 0, 255)
FOG_COLOR = (180, 180, 180, 80)
STORM_COLOR = (50, 50, 50)
LIGHTNING_COLOR = (255, 255, 255)
HARMATTAN_COLOR = (210, 180, 140, 100)
DRONE_COLOR = (255, 0, 0)  # Red color for drones
DRONE_RADIUS = 3  # Radius of drone representation

# Sky Colors for Day-Night Cycle
SKY_COLORS = {
    "sunrise": (255, 223, 186),
    "day": DAY_COLOR,
    "sunset": (255, 140, 0),
    "night": NIGHT_COLOR
}

# Fonts
FONT = pygame.font.Font(None, 24)

# Frame Rate
FPS = 60

# Seasons
SEASONS = ["Rainy", "Dry"]

class Drone:
    def __init__(self, id, position, rotor_speed=1.0):
        """
        Initialize the drone with an ID, initial position, and rotor speed.
        Battery is maintained but not decremented.
        """
        self.id = id
        self.position = position  # (x, y) tuple
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.neighbors = []  # List of nearby drones for information sharing

    def move(self, dx, dy):
        """
        Move the drone by the given offsets in the x and y directions.
        """
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Ensure drone stays within grid bounds
        new_x = max(0, min(new_x, WIDTH))
        new_y = max(0, min(new_y, HEIGHT))

        # Update position
        self.position = (new_x, new_y)
        print(f"Drone {self.id} moved to {self.position}.")

    def adjust_rotor_speed(self, new_speed):
        """
        Adjust the drone's rotor speed.
        """
        self.rotor_speed = max(0.1, min(new_speed, 5.0))  # Keep rotor speed between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted to {self.rotor_speed:.2f}.")

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
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast
        }

class Drone:
    def __init__(self, id, position, rotor_speed=1.0):
        """
        Initialize the drone with an ID, initial position, and rotor speed.
        Battery is maintained but not decremented.
        """
        self.id = id
        self.position = position  # (x, y) tuple
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.neighbors = []  # List of nearby drones for information sharing

    def move(self, dx, dy):
        """
        Move the drone by the given offsets in the x and y directions.
        """
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Ensure drone stays within grid bounds
        new_x = max(0, min(new_x, WIDTH))
        new_y = max(0, min(new_y, HEIGHT))

        # Update position
        self.position = (new_x, new_y)
        print(f"Drone {self.id} moved to {self.position}.")

    def adjust_rotor_speed(self, new_speed):
        """
        Adjust the drone's rotor speed.
        """
        self.rotor_speed = max(0.1, min(new_speed, 5.0))  # Keep rotor speed between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted to {self.rotor_speed:.2f}.")

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
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast
        }

class GridSimulation:
    def __init__(self, environment_file='environment.npy', cell_size=10, num_drones=5):
        """
        Initialize the simulation with grid dimensions, load environment from a file, and initialize drones.
        """
        self.cell_size = cell_size
        self.num_drones = num_drones

        # Load environment
        self.load_environment(environment_file)

        # Initialize drones
        self.drones = self.initialize_drones()

    def load_environment(self, filename):
        """
        Load a fixed grid and buildings from a .npy file.
        """
        try:
            environment_data = np.load(filename, allow_pickle=True).item()
            self.grid = environment_data['grid']
            self.buildings = environment_data['buildings']
            self.land_mask = self.grid > 0.3  # Assuming water_level was 0.3 during saving
            print(f"Environment loaded from '{filename}'.")
        except Exception as e:
            print(f"Failed to load environment from '{filename}': {e}")
            # Initialize empty grid if loading fails
            self.grid = np.zeros((WIDTH // self.cell_size, HEIGHT // self.cell_size))
            self.buildings = np.zeros_like(self.grid)
            self.land_mask = self.grid > 0.3

    def initialize_drones(self):
        """
        Initialize drones with predefined positions on land.
        """
        drones = []
        land_indices = np.argwhere(self.land_mask)

        for i in range(1, self.num_drones + 1):
            # Define specific starting positions or choose from land_indices
            # For simplicity, select randomly from land_indices
            if len(land_indices) == 0:
                print("No land available to place drones.")
                break
            x_idx, y_idx = random.choice(land_indices)
            # Convert grid indices to pixel positions
            x = x_idx * self.cell_size + self.cell_size // 2
            y = y_idx * self.cell_size + self.cell_size // 2

            # Ensure drones are not placed on buildings
            if self.buildings[x_idx, y_idx] > 0:
                print(f"Drone {i} placement on building at ({x}, {y}) avoided.")
                continue

            drone = Drone(id=i, position=(x, y))
            drones.append(drone)
            print(f"Drone {i} initialized at position {drone.position}.")

        return drones

    def draw_grid(self):
        """
        Draw the terrain grid, including water, land, and buildings.
        """
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                elevation = self.grid[x, y]

                if elevation > 0.3:
                    pygame.draw.rect(self.screen, (34, 139, 34), rect)  # Land color
                    if self.buildings[x, y] > 0:
                        building_rect = pygame.Rect(
                            x * self.cell_size + self.cell_size // 4,
                            y * self.cell_size + self.cell_size // 4,
                            self.cell_size // 2,
                            self.cell_size // 2
                        )
                        pygame.draw.rect(self.screen, (139, 69, 19), building_rect)  # Building color
                elif elevation > 0.2:
                    pygame.draw.rect(self.screen, (100, 149, 237), rect)  # Shallow water color
                else:
                    pygame.draw.rect(self.screen, (70, 130, 180), rect)  # Deep water color

    def draw_grid_lines(self):
        """
        Draw grid lines for visual clarity.
        """
        for x in range(0, WIDTH, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (WIDTH, y))

    def update_drones(self):
        """
        Update all drones in the simulation.
        """
        for drone in self.drones:
            # Example Movement: Random walk
            dx = random.choice([-self.cell_size, 0, self.cell_size])
            dy = random.choice([-self.cell_size, 0, self.cell_size])

            # Predict new position
            new_x = drone.position[0] + dx
            new_y = drone.position[1] + dy

            # Ensure new position is within bounds
            new_x = max(0, min(new_x, WIDTH))
            new_y = max(0, min(new_y, HEIGHT))

            # Move drone
            drone.move(dx, dy)

    def draw_drones(self):
        """
        Draw all drones on the grid.
        """
        for drone in self.drones:
            x, y = drone.position
            pygame.draw.circle(self.screen, DRONE_COLOR, (int(x), int(y)), DRONE_RADIUS)
            # Optionally, display drone ID
            id_text = FONT.render(str(drone.id), True, (255, 255, 255))
            self.screen.blit(id_text, (x - DRONE_RADIUS, y - DRONE_RADIUS))

    def log_drone_stats(self, csv_writer):
        """
        Log drone statistics to the CSV file.
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for drone in self.drones:
            csv_writer.writerow([
                timestamp,
                drone.id,
                round(drone.position[0], 2),
                round(drone.position[1], 2),
                # Battery is maintained but not decremented
                drone.battery  # Remains at initial value
            ])

    def run(self):
        """
        Run the main simulation loop, updating and rendering all components.
        """
        running = True

        # Initialize CSV Logging
        try:
            csv_file = open('drone_stats.csv', mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'Timestamp', 'Drone ID', 'Position X', 'Position Y', 'Battery (%)'
            ])
        except IOError as e:
            print(f"Failed to open CSV file: {e}")
            csv_writer = None

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False

            # Update simulation components
            self.update_drones()

            # Render everything
            self.screen.fill((0, 0, 0))  # Clear screen with black
            self.draw_grid()
            self.draw_grid_lines()
            self.draw_drones()
            pygame.display.flip()

            # Log drone statistics
            if csv_writer:
                self.log_drone_stats(csv_writer)
                csv_file.flush()  # Ensure data is written to file

            clock.tick(FPS)

        # After loop ends, close the CSV file
        if csv_writer:
            csv_file.close()
        pygame.quit()

if __name__ == "__main__":
    simulation = GridSimulation(environment_file='environment.npy', cell_size=10, num_drones=5)
    simulation.run()