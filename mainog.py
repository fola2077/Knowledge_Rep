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
SNOW_COLOR = (255, 250, 250)
FOG_COLOR = (180, 180, 180, 80)
STORM_COLOR = (50, 50, 50)
LIGHTNING_COLOR = (255, 255, 255)
HARMATTAN_COLOR = (210, 180, 140, 100)
DRONE_COLOR = (255, 0, 0)  # Red color for drones
DRONE_RADIUS = 3 # Radius of drone representation

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

# STATE VARIABLES

class Drone:
    def __init__(self, id, position, battery=100, rotor_speed=1.0):
        """
        Initialize the drone with an ID, initial position, battery level, and rotor speed.
        """
        self.id = id
        self.position = position  # (x, y) tuple
        self.battery = battery  # Battery percentage
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.neighbors = []  # List of nearby drones for information sharing

    def move(self, dx, dy):
        """
        Move the drone by the given offsets in the x and y directions.
        """
        if self.battery <= 0:
            print(f"Drone {self.id} cannot move. Battery depleted!")
            return
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Ensure drone stays within grid bounds
        new_x = max(0, min(new_x, WIDTH))
        new_y = max(0, min(new_y, HEIGHT))

        # Update position
        self.position = (new_x, new_y)
        self.battery -= self.calculate_battery_usage(dx, dy)
        print(f"Drone {self.id} moved to {self.position}. Battery: {self.battery:.1f}%")

    def calculate_battery_usage(self, dx, dy):
        """
        Calculate battery consumption based on distance moved and rotor speed.
        """
        distance = math.sqrt(dx**2 + dy**2)
        usage = distance * self.rotor_speed * 0.1  # Adjusted battery usage rate
        return min(usage, self.battery)

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
            "battery": self.battery,
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast
        }


class GridSimulation:
    def __init__(self, width=1200, height=800, cell_size=10, num_islands=8, max_radius=10, num_buildings=50, num_drones=5):
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
        self.num_drones = num_drones
        self.water_level = 0.3  # Default water level
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Grid Simulation with Drones")

        # Initialize grid and features
        self.grid = self.generate_grid()
        self.land_mask = self.grid > self.water_level
        self.buildings = self.add_buildings()

        # Initialize drones
        self.drones = self.initialize_drones()

        # Define recharge stations (Optional Enhancement)
        self.recharge_stations = self.define_recharge_stations()

    def generate_grid(self):
        """
        Generate a terrain grid with islands using Gaussian smoothing.
        """
        grid = np.zeros((self.grid_width, self.grid_height))

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

    def initialize_drones(self):
        """
        Initialize drones with random positions on land.
        """
        drones = []
        land_indices = np.argwhere(self.land_mask)

        for i in range(1, self.num_drones + 1):
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

    def define_recharge_stations(self):
        """
        Define recharge stations at fixed locations on land.
        """
        recharge_stations = []
        land_indices = np.argwhere(self.land_mask)

        num_stations = max(1, self.num_drones // 2)  # Define at least one station

        for _ in range(num_stations):
            if len(land_indices) == 0:
                break
            x_idx, y_idx = random.choice(land_indices)
            x = x_idx * self.cell_size + self.cell_size // 2
            y = y_idx * self.cell_size + self.cell_size // 2

            # Ensure no building is present
            if self.buildings[x_idx, y_idx] > 0:
                continue

            # Ensure no drone is already at the location
            if any(math.isclose(x, drone.position[0], abs_tol=self.cell_size) and
                   math.isclose(y, drone.position[1], abs_tol=self.cell_size) for drone in self.drones):
                continue

            recharge_stations.append((x, y))
            print(f"Recharge station defined at ({x}, {y}).")

        return recharge_stations

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
        Draw the terrain grid, including water, land, buildings, and recharge stations.
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

        # Draw recharge stations
        for station in self.recharge_stations:
            pygame.draw.circle(self.screen, (0, 255, 0), (int(station[0]), int(station[1])), DRONE_RADIUS + 3)
            # Optionally, label the recharge stations
            station_text = FONT.render("R", True, (255, 255, 255))
            self.screen.blit(station_text, (station[0] - DRONE_RADIUS, station[1] - DRONE_RADIUS))

    def draw_grid_lines(self):
        """
        Draw grid lines for visual clarity.
        """
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (self.width, y))

    def update_drones(self):
        """
        Update all drones in the simulation.
        """
        for drone in self.drones:
            # Example Movement: Random walk
            if drone.battery > 0:
                dx = random.choice([-self.cell_size, 0, self.cell_size])
                dy = random.choice([-self.cell_size, 0, self.cell_size])

                # Predict new position
                new_x = drone.position[0] + dx
                new_y = drone.position[1] + dy

                # Ensure new position is within bounds
                new_x = max(0, min(new_x, self.width))
                new_y = max(0, min(new_y, self.height))

                # Check for collisions with buildings or water
                grid_x = int(new_x // self.cell_size)
                grid_y = int(new_y // self.cell_size)

                if self.land_mask[grid_x, grid_y] and self.buildings[grid_x, grid_y] == 0:
                    drone.move(dx, dy)
                else:
                    print(f"Drone {drone.id} cannot move to ({new_x}, {new_y}) - Obstacle detected.")
            else:
                # If battery is low, attempt to return to nearest recharge station
                self.return_to_recharge(drone)

    def return_to_recharge(self, drone):
        """
        Direct the drone to return to the nearest recharge station.
        """
        if not self.recharge_stations:
            print("No recharge stations available.")
            return

        # Find the nearest recharge station
        nearest_station = min(self.recharge_stations, key=lambda station: math.hypot(station[0] - drone.position[0], station[1] - drone.position[1]))
        station_x, station_y = nearest_station

        # Simple directional movement towards the recharge station
        dx = 0
        dy = 0
        if drone.position[0] < station_x:
            dx = self.cell_size
        elif drone.position[0] > station_x:
            dx = -self.cell_size

        if drone.position[1] < station_y:
            dy = self.cell_size
        elif drone.position[1] > station_y:
            dy = -self.cell_size

        # Move drone towards the recharge station
        drone.move(dx, dy)

        # Check if drone has arrived at the recharge station
        if math.isclose(drone.position[0], station_x, abs_tol=self.cell_size) and math.isclose(drone.position[1], station_y, abs_tol=self.cell_size):
            drone.battery = 100  # Recharge battery
            print(f"Drone {drone.id} has recharged at ({station_x}, {station_y}).")

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
                round(drone.battery, 2)
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
    simulation = GridSimulation()
    simulation.run()
