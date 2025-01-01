# main.py

import pygame
import sys
import random
import numpy as np
import logging

from drone import Drone
from environment import Environment, WATER_LEVEL
from weather import WeatherSystem, TimeManager
from oilspillage import OilSpillage  # The new block-based class

from config import WIDTH, HEIGHT

# Constants
CELL_SIZE = 10
FPS = 60

# Colors
DRONE_COLOR = (255, 165, 0)  # Orange for drones
INFO_COLOR = (255, 255, 255) # White text

blue_shades = [
    (0, 0, 139),  # Dark Blue
    (0, 0, 205),  # Medium Blue
    (0, 0, 255)   # Blue
]

DRONE_SPEED_FACTOR = 0.95

# Initialize Pygame fonts
pygame.font.init()
FONT = pygame.font.Font(None, 24)

# Configure logging
logging.basicConfig(
    filename='main.log',
    filemode='w',  # Overwrite each run
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def render_static_environment(environment_surface, environment):
    """
    Render the static (land, water, buildings) environment to environment_surface.
    """
    environment_surface.fill((0, 0, 0))  # Black background

    for gx in range(environment.grid_width):
        for gy in range(environment.grid_height):
            rect = pygame.Rect(
                gx * CELL_SIZE,
                gy * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            elevation = environment.grid[gx, gy]

            if elevation > WATER_LEVEL:
                # Land
                pygame.draw.rect(environment_surface, (34, 139, 34), rect)
                if environment.buildings[gx, gy] > 0:
                    building_rect = pygame.Rect(
                        gx * CELL_SIZE + CELL_SIZE // 4,
                        gy * CELL_SIZE + CELL_SIZE // 4,
                        CELL_SIZE // 2,
                        CELL_SIZE // 2
                    )
                    pygame.draw.rect(environment_surface, (139, 69, 19), building_rect)
            else:
                # Water
                shade_index = min(
                    int(elevation / WATER_LEVEL * (len(blue_shades) - 1)),
                    len(blue_shades) - 1
                )
                shade = blue_shades[shade_index]
                pygame.draw.rect(environment_surface, shade, rect)

    # Grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(environment_surface, (200, 200, 200), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(environment_surface, (200, 200, 200), (0, y), (WIDTH, y))

    # Highlight buildings
    for gx in range(environment.grid_width):
        for gy in range(environment.grid_height):
            if environment.buildings[gx, gy] > 0:
                rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(environment_surface, (255, 255, 255), rect, 1)

def render_oil_spills(environment_surface, environment, oil_spillage_manager):
    """
    Render the oil spills based on oil concentration levels.
    """
    concentration_grid = oil_spillage_manager.combined_oil_concentration()
    for gx in range(environment.grid_width):
        for gy in range(environment.grid_height):
            concentration = concentration_grid[gx, gy]
            if concentration > 0:
                color = oil_spillage_manager.get_cell_color(concentration)
                rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(environment_surface, color, rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Simulation (Block-based Oil Spillage)")
    clock = pygame.time.Clock()

    # Setup environment, weather, etc.
    environment = Environment(load_from_file=True)
    time_manager = TimeManager()
    weather_system = WeatherSystem(time_manager)

    logging.info("Simulation started (Block-based OilSpillage).")

    # Create some drones
    num_drones = 5
    drones = []
    land_indices = list(zip(*np.where(environment.land_mask)))
    random.shuffle(land_indices)

    for i in range(1, num_drones + 1):
        if not land_indices:
            logging.warning("No land available to place more drones.")
            break
        gx, gy = land_indices.pop()
        x = gx * CELL_SIZE + CELL_SIZE // 2
        y = gy * CELL_SIZE + CELL_SIZE // 2

        # If building, skip
        if environment.buildings[gx, gy] > 0:
            logging.warning(f"Drone {i} avoided building at ({x}, {y}).")
            continue

        drone = Drone(
            id=i,
            position=(x, y),
            weather_system=weather_system,
            color=DRONE_COLOR
        )
        drone.load_environment(environment)
        drones.append(drone)
        logging.info(f"Drone {i} at position ({x}, {y}).")

    environment.drones = drones

    # # Create the block-based OilSpillage manager
    oil_spillage_manager = OilSpillage(environment, time_manager)

    environment_surface = pygame.Surface((WIDTH, HEIGHT))
    render_static_environment(environment_surface, environment)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # seconds

        # Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update time + weather
        time_manager.update(dt)
        weather_system.update(dt)

        # 1) Update oil spillage
        oil_spillage_manager.update(weather_system)

        # 2) Update drones
        for drone in drones:
            drone.find_neighbors(drones)
        for drone in drones:
            drone.share_information()
        for drone in drones:
            drone.update_behavior()

            movement_per_second = CELL_SIZE * DRONE_SPEED_FACTOR
            dx = int(random.randint(-CELL_SIZE, CELL_SIZE) * movement_per_second * dt)
            dy = int(random.randint(-CELL_SIZE, CELL_SIZE) * movement_per_second * dt)
            drone.move(dx, dy, drones)
            drone.detect_oil()

        # Render environment
        render_static_environment(environment_surface, environment)

        # Render block-based oil
        render_oil_spills(environment_surface, environment, oil_spillage_manager)

        # Render drones
        for drone in drones:
            pygame.draw.circle(
                environment_surface,
                drone.color,
                (int(drone.position.x), int(drone.position.y)),
                5
            )
            id_text = FONT.render(str(drone.id), True, INFO_COLOR)
            environment_surface.blit(id_text, (drone.position.x + 5, drone.position.y - 10))

        # Blit
        screen.blit(environment_surface, (0, 0))

        # Show stats
        stats = [
            f"Day: {time_manager.day_count + 1}",
            f"Time: {time_manager.hour:02}:{int(time_manager.minute):02}",
            f"Season: {time_manager.season}",
            f"Weather: {weather_system.get_current_weather().name}",
            f"Intensity: {weather_system.get_current_weather().intensity:.2f}",
            f"Temperature: {weather_system.get_current_weather().temperature:.1f} °C",
            f"Humidity: {weather_system.get_current_weather().humidity:.1f}%",
            f"Wind Speed: {weather_system.get_current_weather().wind_speed:.1f} m/s",
            f"Wind Dir: {weather_system.get_current_weather().wind_direction:.1f}°",
            f"Precipitation: {weather_system.get_current_weather().precipitation_type}",
            f"Visibility: {weather_system.get_current_weather().visibility:.2f}",
            f"Cloud Density: {weather_system.get_current_weather().cloud_density:.2f}",
            f"Air Pressure: {weather_system.get_current_weather().air_pressure:.1f} hPa",
            f"Drones: {len(drones)}"
        ]

        y_offset = 10
        for stat in stats:
            stat_surf = FONT.render(stat, True, INFO_COLOR)
            screen.blit(stat_surf, (10, y_offset))
            y_offset += 20

        pygame.display.flip()

    # On exit, save environment + drones
    environment.save_environment(drones)
    logging.info("Environment and drones saved on exit.")

    if hasattr(weather_system, 'close_csv'):
        weather_system.close_csv()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        pygame.quit()
        sys.exit()
