import pygame
import numpy as np
import random
import time
from scipy.ndimage import gaussian_filter

# Constants
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
FPS = 60

# Colors
WATER_COLOR = (70, 130, 180)
LAND_COLOR = (34, 139, 34)
BUILDING_COLOR = (139, 69, 19)
OIL_SPILL_COLOR = (0, 0, 0)
DETECTED_OIL_COLOR = (255, 0, 0)
DRONE_COLOR = (0, 255, 255)
DAY_COLOR = (135, 206, 235)
NIGHT_COLOR = (25, 25, 112)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Simulation with Weather and Oil Spill Detection")
clock = pygame.time.Clock()


### Classes

class TimeManager:
    """Manages the simulation's time system."""
    def __init__(self):
        self.hour = 6
        self.minute = 0

    def update(self, dt):
        self.minute += dt * 10
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
        if self.hour >= 24:
            self.hour = 0

    def is_daytime(self):
        return 6 <= self.hour < 18


class WeatherSystem:
    """Handles probabilistic weather conditions."""
    def __init__(self, season="rainy"):
        self.season = season
        self.time_of_day = "day"
        self.weather_condition = "sunny"
        self.temperature = 27
        self.humidity = 85
        self.wind_speed = random.uniform(1, 5)
        self.wind_direction_angle = random.uniform(0, 360)
        self.weather_change_interval = 20
        self.last_weather_change = time.time()

    def update(self):
        current_time = time.time()
        if current_time - self.last_weather_change >= self.weather_change_interval:
            self.last_weather_change = current_time
            self.weather_condition = random.choice(["sunny", "rainy", "foggy", "windy"])

    def draw_weather(self):
        screen.fill(DAY_COLOR if time_manager.is_daytime() else NIGHT_COLOR)
        if self.weather_condition == "rainy":
            for _ in range(50):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                pygame.draw.line(screen, (0, 0, 255), (x, y), (x, y + 10), 2)


class OilSpillage:
    """Represents an oil spillage on the grid."""
    def __init__(self, position):
        self.position = position
        self.detected = False

    def detect(self, drones, detection_radius):
        if not self.detected:
            for drone in drones:
                if np.linalg.norm(np.array(drone) - np.array(self.position)) <= detection_radius:
                    self.detected = True
                    return True
        return False


class DroneEnvironment:
    """Handles the grid, drones, and oil spillages."""
    def __init__(self, num_drones, num_spillages):
        self.grid = self.generate_grid()
        self.drones = [np.random.randint(0, GRID_WIDTH, 2) for _ in range(num_drones)]
        self.oil_spillages = [OilSpillage((random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)))
                              for _ in range(num_spillages)]
        self.detection_radius = 2

    def generate_grid(self):
        grid = np.zeros((GRID_WIDTH, GRID_HEIGHT))
        for _ in range(8):  # Islands
            cx, cy = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
            radius = random.randint(3, 10)
            for x in range(GRID_WIDTH):
                for y in range(GRID_HEIGHT):
                    if np.sqrt((x - cx)**2 + (y - cy)**2) < radius:
                        grid[x, y] += random.uniform(0.5, 1.0)
        return gaussian_filter(grid, sigma=1)

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            if action == 1: self.drones[i][1] -= 1  # Up
            elif action == 2: self.drones[i][1] += 1  # Down
            elif action == 3: self.drones[i][0] -= 1  # Left
            elif action == 4: self.drones[i][0] += 1  # Right
            self.drones[i] = np.clip(self.drones[i], 0, GRID_WIDTH - 1)

            reward = 0
            for spillage in self.oil_spillages:
                if spillage.detect(self.drones, self.detection_radius):
                    reward += 10
            rewards.append(reward)
        return rewards

    def render(self):
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[x, y] > 0.3:
                    pygame.draw.rect(screen, LAND_COLOR, rect)
                else:
                    pygame.draw.rect(screen, WATER_COLOR, rect)

        for spillage in self.oil_spillages:
            color = DETECTED_OIL_COLOR if spillage.detected else OIL_SPILL_COLOR
            rect = pygame.Rect(spillage.position[0] * CELL_SIZE, spillage.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)

        for drone in self.drones:
            center = (drone[0] * CELL_SIZE + CELL_SIZE // 2, drone[1] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(screen, DRONE_COLOR, center, CELL_SIZE // 3)


### Main Simulation

time_manager = TimeManager()
weather = WeatherSystem(season="rainy")
environment = DroneEnvironment(num_drones=3, num_spillages=10)

running = True
last_update_time = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    dt = time.time() - last_update_time
    last_update_time = time.time()

    # Update systems
    time_manager.update(dt)
    weather.update()
    actions = [random.randint(0, 4) for _ in range(len(environment.drones))]
    environment.step(actions)

    # Render
    weather.draw_weather()
    environment.render()

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
