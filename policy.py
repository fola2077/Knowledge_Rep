import pygame
import random
import math
import time
import numpy as np

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Colors
LAND_COLOR = (34, 139, 34)
WATER_COLOR = (70, 130, 180)
RAIN_COLOR = (0, 0, 255)
FOG_COLOR = (180, 180, 180, 100)
DRONE_COLOR = (255, 0, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Integrated Simulation")

# Clock
clock = pygame.time.Clock()

class Drone:
    def __init__(self, id, position, battery=100, rotor_speed=1.0):
        self.id = id
        self.position = position  # (x, y)
        self.battery = battery
        self.rotor_speed = rotor_speed
        self.weather_affected = False

    def move(self, dx, dy, grid, weather):
        """Move the drone, taking terrain and weather into account."""
        if self.battery <= 0:
            print(f"Drone {self.id}: Battery depleted!")
            return

        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # Check terrain
        if 0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT:
            elevation = grid[new_x, new_y]
            if elevation < weather.water_level:  # Avoid water
                print(f"Drone {self.id}: Cannot move to water tile.")
                return

        # Adjust for weather
        if weather.wind_speed > 3.0:  # Arbitrary threshold
            self.rotor_speed *= 0.9  # Reduce efficiency
            self.weather_affected = True
        else:
            self.weather_affected = False

        # Move and reduce battery
        self.position = (new_x, new_y)
        self.battery -= 1
        print(f"Drone {self.id} moved to {self.position}. Battery: {self.battery}%.")

    def report_status(self):
        """Report current drone status."""
        return {
            "ID": self.id,
            "Position": self.position,
            "Battery": self.battery,
            "Rotor Speed": self.rotor_speed,
            "Weather Affected": self.weather_affected,
        }

class WeatherSystem:
    def __init__(self, season="rainy"):
        self.season = season
        self.weather_condition = "sunny"
        self.wind_speed = random.uniform(1, 5)
        self.water_level = 0.3  # Flood level

    def update_weather(self):
        """Randomly update weather conditions."""
        probabilities = {"sunny": 0.5, "rainy": 0.3, "foggy": 0.2}
        self.weather_condition = random.choices(list(probabilities.keys()), weights=probabilities.values())[0]

        # Adjust wind and water levels
        if self.weather_condition == "rainy":
            self.wind_speed = random.uniform(3, 7)
            self.water_level += 0.05  # Simulate flooding
        elif self.weather_condition == "foggy":
            self.wind_speed = random.uniform(1, 3)
        else:
            self.wind_speed = random.uniform(0, 2)
            self.water_level = max(0.3, self.water_level - 0.02)  # Dry up slowly

    def draw_weather(self, screen):
        """Visualize weather effects."""
        if self.weather_condition == "foggy":
            fog_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fog_surface.fill(FOG_COLOR)
            screen.blit(fog_surface, (0, 0))
        elif self.weather_condition == "rainy":
            for _ in range(100):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                pygame.draw.line(screen, RAIN_COLOR, (x, y), (x, y + 5), 1)

class GridSimulation:
    def __init__(self, width, height):
        self.grid = np.zeros((width, height))
        self.generate_terrain()

    def generate_terrain(self):
        """Generate random terrain for the grid."""
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                self.grid[x, y] = random.uniform(0, 1)  # Elevation (0 = water, 1 = land)

    def draw_grid(self, screen, weather):
        """Visualize the terrain."""
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                elevation = self.grid[x, y]
                if elevation < weather.water_level:
                    pygame.draw.rect(screen, WATER_COLOR, rect)
                else:
                    pygame.draw.rect(screen, LAND_COLOR, rect)

# Initialize systems
grid_sim = GridSimulation(GRID_WIDTH, GRID_HEIGHT)
weather = WeatherSystem(season="rainy")
drones = [Drone(id=i, position=(random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))) for i in range(3)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update systems
    weather.update_weather()
    for drone in drones:
        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        drone.move(dx, dy, grid_sim.grid, weather)

    # Draw everything
    screen.fill((0, 0, 0))
    grid_sim.draw_grid(screen, weather)
    weather.draw_weather(screen)
    for drone in drones:
        pygame.draw.circle(screen, DRONE_COLOR, (drone.position[0] * CELL_SIZE + CELL_SIZE // 2, drone.position[1] * CELL_SIZE + CELL_SIZE // 2), 5)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
