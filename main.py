# main.py

import pygame
import sys
import random
import numpy as np
import logging
from pygame.math import Vector2
from drone import Drone
from environment import Environment, WATER_LEVEL
from weather import WeatherSystem, TimeManager
from oilspillage import OilSpillage  # The new block-based class

from stable_baselines3 import PPO  # <-- Import the trained RL model
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

DRONE_SPEED_FACTOR = 2.0

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
            rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
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
    Render the oil spills with different colors for detected and undetected oil.
    """
    total_concentration = oil_spillage_manager.combined_oil_concentration()
    detected_concentration = oil_spillage_manager.combined_detected_concentration()

    for gx in range(environment.grid_width):
        for gy in range(environment.grid_height):
            concentration = total_concentration[gx, gy]
            detected = detected_concentration[gx, gy]
            
            if concentration > 0:
                rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if detected > 0:
                    # Bright green for detected oil
                    color = (0, 255, 0)  # Full green
                    pygame.draw.rect(environment_surface, color, rect)
                    # Add a border to make it more visible
                    pygame.draw.rect(environment_surface, (255, 255, 255), rect, 1)
                else:
                    # Purple for undetected oil
                    color = oil_spillage_manager.get_cell_color(concentration)
                    pygame.draw.rect(environment_surface, color, rect)


def render_oil_legend(screen):
    """
    Render a legend showing what the colors mean.
    """
    font = pygame.font.Font(None, 24)
    legend_items = [
        ("Undetected Oil", (128, 0, 128)),  # Purple
        ("Detected Oil", (0, 255, 0))       # Green
    ]
    
    y_offset = 10
    for text, color in legend_items:
        pygame.draw.rect(screen, color, (10, y_offset, 20, 20))
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (35, y_offset))
        y_offset += 30


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Simulation (Block-based Oil Spillage)")
    clock = pygame.time.Clock()

    # 1) Setup environment, time, weather, oil spillage
    environment = Environment(load_from_file=True)
    time_manager = TimeManager()
    weather_system = WeatherSystem(time_manager)
    oil_spillage_manager = OilSpillage(environment, time_manager)
    environment.set_oil_spillage_manager(oil_spillage_manager)
    logging.info("Visualization started (Block-based OilSpillage).")

    # 2) Create drones in environment
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

        if environment.buildings[gx, gy] > 0:
            logging.warning(f"Drone {i} avoided building at ({x}, {y}).")
            continue

        drone = Drone(
            id=i,
            position=(x, y),
            environment=environment,
            weather_system=weather_system,
            time_manager=time_manager,
            color=DRONE_COLOR
        )
        drone.load_environment(environment, oil_spillage_manager)
        drones.append(drone)
        logging.info(f"Drone {i} at position ({x}, {y}).")

    environment.drones = drones

    # 3) Load the trained RL model
    try:
        model = PPO.load("ppo_drone_final_multi")
        logging.info("Successfully loaded PPO model.")
    except FileNotFoundError:
        logging.error("Trained model file not found! Provide correct path to the model.")
        print("Model file 'ppo_drone_final_multi.zip' not found. Exiting.")
        pygame.quit()
        sys.exit()

    # 4) Function to gather observation (matching training env's logic)
    def get_observation(drones, environment, weather_system, oil_spillage_manager):
        max_drones = 5
        features_per_drone = 6
        obs_array = np.zeros(features_per_drone * max_drones + 1, dtype=np.float32)

        for idx, drone in enumerate(drones):
            if idx >= max_drones:
                break
            x = float(drone.position.x)  # Ensure these are scalar values
            y = float(drone.position.y)
            current_weather = weather_system.get_current_weather()
            wind_speed = float(current_weather.wind_speed)
            wind_direction = float(current_weather.wind_direction)
            visibility = float(current_weather.visibility)
            total_detected_oil = float(oil_spillage_manager.get_total_detected_oil())

            start = idx * features_per_drone
            obs_array[start : start+6] = [
                x, y, wind_speed, wind_direction, visibility, total_detected_oil
            ]

        # Add global feature: total_oil_remaining
        total_oil_remaining = float(oil_spillage_manager.get_total_oil())
        obs_array[-1] = total_oil_remaining

        return obs_array.reshape((1, -1))  # Ensure shape is (1, 31)

    # 5) Our partial environment "step" logic, if desired
    def step_environment(actions):
        """
        Applies the actions to the drones, then updates time, weather, spillage, etc.
        """
        # Convert actions to movements or scans, replicate training env logic
        for i, drone in enumerate(drones):
            act = actions[i]
            dx, dy = 0.0, 0.0

            if act == 0:  # Up
                dy = -drone.movement_speed
            elif act == 1:  # Down
                dy = drone.movement_speed
            elif act == 2:  # Left
                dx = -drone.movement_speed
            elif act == 3:  # Right
                dx = drone.movement_speed
            elif act == 4:
                # "Scan" action
                drone.scan_for_oil(frames=4)

            # Move only if act is in [0..3]
            if act in [0,1,2,3]:
                drone.move(dx, dy, drones)

        # Update time, weather, spillage for one "tick"
        dt = 1.0  # or clock.get_time() if you want it in ms
        time_manager.update(dt)
        weather_system.update(dt)
        current_total_minutes = time_manager.get_current_total_minutes()
        oil_spillage_manager.update(weather_system, dt, current_total_minutes)

    # Legend
    render_oil_legend(screen)

    environment_surface = pygame.Surface((WIDTH, HEIGHT))
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # A) Build observation
        try:
            obs = get_observation(drones, environment, weather_system, oil_spillage_manager)

            # B) Model predicts action
            actions, _states = model.predict(obs, deterministic=True)
            if isinstance(actions, np.ndarray):
                actions = actions.squeeze()  # Remove single-dimensional entries
        except ValueError as e:
            logging.error(f"Prediction error: {e}")
            actions = np.zeros(len(drones))  # Default actions
            
        # C) Step environment with these actions
        step_environment(actions)

        # D) Render environment
        render_static_environment(environment_surface, environment)
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

        # Blit to screen
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
