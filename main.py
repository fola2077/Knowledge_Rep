# main.py
import pygame
import sys
import random
import numpy as np
import logging

from drone import Drone
from environment import Environment, WATER_LEVEL
from weather import WeatherSystem, TimeManager
from config import WIDTH, HEIGHT

# Constants
CELL_SIZE = 10
FPS = 60

# Colors
DRONE_COLOR = (255, 165, 0)     # Red color for drones
INFO_COLOR = (255, 255, 255)  # White color for info text

blue_shades = [
    (0, 0, 139),      # Dark Blue
    (0, 0, 205),      # Medium Blue
    (0, 0, 255)       # Blue
]

DRONE_SPEED_FACTOR = 0.95 # Speed factor for drones

# Initialize Pygame fonts
pygame.font.init()
FONT = pygame.font.Font(None, 24)

# Configure logging for main.py
logging.basicConfig(
    filename='main.log',
    filemode='w',  # Overwrite log file each run
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add console handler for real-time logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def render_static_environment(environment_surface, environment):
    """
    Render the static elements of the environment onto the environment_surface.
    """
    environment_surface.fill((0, 0, 0))  # Black background

    for x in range(environment.grid_width):
        for y in range(environment.grid_height):
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            elevation = environment.grid[x, y]

            if elevation > WATER_LEVEL:
                pygame.draw.rect(environment_surface, (34, 139, 34), rect)  # LAND_COLOR

                if environment.buildings[x, y] > 0:
                    building_rect = pygame.Rect(
                        x * CELL_SIZE + CELL_SIZE // 4,
                        y * CELL_SIZE + CELL_SIZE // 4,
                        CELL_SIZE // 2,
                        CELL_SIZE // 2
                    )
                    pygame.draw.rect(environment_surface, (139, 69, 19), building_rect)  # BUILDING_COLOR
            else:
                # Draw water based on elevation
                shade_index = min(
                    int(elevation / WATER_LEVEL * (len(blue_shades) - 1)),
                    len(blue_shades) - 1
                )
                shade = blue_shades[shade_index]
                pygame.draw.rect(environment_surface, shade, rect)

    # Draw grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(environment_surface, (200, 200, 200), (x, 0), (x, HEIGHT))  # GRID_LINES_COLOR
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(environment_surface, (200, 200, 200), (0, y), (WIDTH, y))  # GRID_LINES_COLOR

    # Highlight buildings with white borders
    for x in range(environment.grid_width):
        for y in range(environment.grid_height):
            if environment.buildings[x, y] > 0:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(environment_surface, (255, 255, 255), rect, 1)  # HIGHLIGHT_COLOR

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Simulation with Environment and Weather")
    clock = pygame.time.Clock()

    # Initialize simulation components
    environment = Environment(load_from_file=True)
    time_manager = TimeManager()
    weather_system = WeatherSystem(time_manager)

    logging.info("Simulation started.")

    # Initialize drones based on the loaded environment
    num_drones = 5
    drones = []
    land_indices = list(zip(*np.where(environment.land_mask)))
    random.shuffle(land_indices)  # Shuffle to randomize drone placement

    for i in range(1, num_drones + 1):
        if not land_indices:
            logging.warning("No land available to place more drones.")
            break
        pos_idx = land_indices.pop()
        grid_x, grid_y = pos_idx
        x = grid_x * CELL_SIZE + CELL_SIZE // 2
        y = grid_y * CELL_SIZE + CELL_SIZE // 2

        # Ensure no building at this position
        if environment.buildings[grid_x, grid_y] > 0:
            logging.warning(f"Drone {i} placement on building at ({x}, {y}) avoided.")
            continue

        drone = Drone(id=i, position=(x, y), weather_system=weather_system, color=DRONE_COLOR)
        drone.load_environment(environment)
        drones.append(drone)
        logging.info(f"Drone {i} initialized at position ({x}, {y}).")

    environment.drones = drones  # If Environment class utilizes this

    # Create environment surface
    environment_surface = pygame.Surface((WIDTH, HEIGHT))
    render_static_environment(environment_surface, environment)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Delta time in seconds

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update simulation time and weather
        time_manager.update(dt)
        weather_system.update(dt)

        # Update drones: find neighbors, share information, and adjust behavior based on weather
        for drone in drones:
            drone.find_neighbors(drones)

        for drone in drones:
            drone.share_information()

        for drone in drones:
            drone.update_behavior()

            # Define base movement per second
            movement_per_second = CELL_SIZE * DRONE_SPEED_FACTOR

            # Example movement: random movement within cell size
            dx = int(random.randint(-CELL_SIZE, CELL_SIZE) * movement_per_second * dt) # movement is proportional to time elapsed
            dy = int(random.randint(-CELL_SIZE, CELL_SIZE) * movement_per_second * dt) # movement is proportional to time elapsed
            drone.move(dx, dy, drones)

        # Rendering
        # Clear environment_surface by re-rendering static environment
        render_static_environment(environment_surface, environment)

        # Draw drones on environment_surface
        for drone in drones:
            pygame.draw.circle(
                environment_surface, 
                drone.color, 
                (int(drone.position.x), int(drone.position.y)), 
                5
            )
            # Optionally, draw drone ID
            id_text = FONT.render(str(drone.id), True, INFO_COLOR)
            environment_surface.blit(id_text, (drone.position.x + 5, drone.position.y - 10))

        # Blit environment_surface onto the main screen
        screen.blit(environment_surface, (0, 0))

        # Render statistics
        stats = [
            f"Day: {time_manager.day_count + 1}",
            f"Time: {time_manager.hour:02}:{int(time_manager.minute):02}",
            f"Season: {time_manager.season}",
            f"Weather: {weather_system.get_current_weather().name}",
            f"Intensity: {weather_system.get_current_weather().intensity:.2f}",
            f"Temperature: {weather_system.get_current_weather().temperature:.1f} °C",
            f"Humidity: {weather_system.get_current_weather().humidity:.1f}%",
            f"Wind Speed: {weather_system.get_current_weather().wind_speed:.1f} m/s",
            f"Wind Direction: {weather_system.get_current_weather().wind_direction:.1f}°",
            f"Precipitation: {weather_system.get_current_weather().precipitation_type}",
            f"Visibility: {weather_system.get_current_weather().visibility:.2f}",
            f"Cloud Density: {weather_system.get_current_weather().cloud_density:.2f}",
            f"Air Pressure: {weather_system.get_current_weather().air_pressure:.1f} hPa",
            f"Drones: {len(drones)}"
        ]

        y_offset = 10
        for stat in stats:
            stat_surface = FONT.render(stat, True, INFO_COLOR)
            screen.blit(stat_surface, (10, y_offset))
            y_offset += 20

        # Update the display
        pygame.display.flip()

    # Save environment and drones upon exit
    environment.save_environment(drones)
    logging.info("Environment and drones saved on exit.")

    # Close the weather CSV file if necessary
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
