# main.py

import pygame
import sys
import random
import numpy as np
from drone import Drone
from environment import Environment, WATER_LEVEL
from weather import WeatherSystem, TimeManager

# Constants
WIDTH, HEIGHT = 1200, 800
CELL_SIZE = 10
FPS = 60

# Colors
LAND_COLOR = (34, 139, 34)            # Land
BUILDING_COLOR = (139, 69, 19)        # Buildings
GRID_LINES_COLOR = (200, 200, 200)    # Grid lines
HIGHLIGHT_COLOR = (255, 255, 255)     # Highlight color for buildings
DRONE_COLOR = (255, 0, 0)             # Red color for drones
INFO_COLOR = (255, 255, 255)          # White color for info text
blue_shades = [
    (0, 0, 139),      # Dark Blue
    (0, 0, 205),      # Medium Blue
    (0, 0, 255)       # Blue
]


# Initialize Pygame fonts
pygame.font.init()
FONT = pygame.font.Font(None, 24)

def main():
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Simulation with Environment and Weather")
    clock = pygame.time.Clock()

    # Initialize simulation components
    environment = Environment()
    time_manager = TimeManager()
    weather_system = WeatherSystem(time_manager)

    # Initialize drones
    num_drones = 5
    drones = []
    land_indices = list(zip(*np.where(environment.land_mask)))
    for i in range(1, num_drones + 1):
        if not land_indices:
            print("No land available to place more drones.")
            break
        pos_idx = random.choice(land_indices)
        x = pos_idx[0] * CELL_SIZE + CELL_SIZE // 2
        y = pos_idx[1] * CELL_SIZE + CELL_SIZE // 2

        # Ensure no building at this position
        if environment.buildings[pos_idx] > 0:
            print(f"Drone {i} placement on building at ({x}, {y}) avoided.")
            continue

        drone = Drone(id=i, position=(x, y), weather_system=weather_system, color=DRONE_COLOR)
        drone.load_environment(environment)
        drones.append(drone)
        print(f"Drone {i} initialized at position {drone.position}.")

    # Create separate surfaces
    environment_surface = pygame.Surface((WIDTH, HEIGHT))
    # weather_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)  # Allows transparency

    # Simulation loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000  # Delta time in seconds

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle key presses
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_s:
                    # Save environment and drones
                    environment.save_environment(drones)
                elif event.key == pygame.K_l:
                    # Load environment and drones
                    environment.load_environment(drones)
            # Handle mouse clicks
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click to add building
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    if 0 <= grid_x < environment.grid_width and 0 <= grid_y < environment.grid_height:
                        if environment.grid[grid_x, grid_y] > WATER_LEVEL:
                            environment.buildings[grid_x, grid_y] = random.uniform(1.0, 3.0)
                            print(f"Added building at ({x}, {y}).")
                            environment.log_action("Building Added", f"Added building at ({x}, {y})")
                elif event.button == 3:  # Right click to remove building
                    pos = pygame.mouse.get_pos()
                    x, y = pos
                    grid_x = x // CELL_SIZE
                    grid_y = y // CELL_SIZE
                    if 0 <= grid_x < environment.grid_width and 0 <= grid_y < environment.grid_height:
                        if environment.buildings[grid_x, grid_y] > 0:
                            environment.buildings[grid_x, grid_y] = 0.0
                            print(f"Removed building at ({x}, {y}).")
                            environment.log_action("Building Removed", f"Removed building at ({x}, {y})")

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
            # Example movement: random movement within cell size
            dx = random.randint(-CELL_SIZE, CELL_SIZE)
            dy = random.randint(-CELL_SIZE, CELL_SIZE)
            drone.move(dx, dy, drones)

        # Rendering
        # Clear environment_surface
        environment_surface.fill((0, 0, 0))  # Black background

        # Draw environment grid and buildings on environment_surface
        for x in range(environment.grid_width):
            for y in range(environment.grid_height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                elevation = environment.grid[x, y]

                if elevation > WATER_LEVEL:
                    pygame.draw.rect(environment_surface, LAND_COLOR, rect)

                    if environment.buildings[x, y] > 0:
                        building_rect = pygame.Rect(
                            x * CELL_SIZE + CELL_SIZE // 4,
                            y * CELL_SIZE + CELL_SIZE // 4,
                            CELL_SIZE // 2,
                            CELL_SIZE // 2
                        )
                        pygame.draw.rect(environment_surface, BUILDING_COLOR, building_rect)
                else:
                    # Draw water based on elevation
                    shade = blue_shades[min(int(elevation / WATER_LEVEL * (len(blue_shades) - 1)), len(blue_shades)-1)]
                    pygame.draw.rect(environment_surface, shade, rect)
                    # pygame.draw.rect(environment_surface, (0, 0, 255), rect)
                # elif elevation > WATER_LEVEL - 0.1:
                #     # Shallow water can be represented if needed
                #     pass
                # else:
                #     # Deep water can be represented if needed
                #     pass

        # Draw grid lines on environment_surface
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(environment_surface, GRID_LINES_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(environment_surface, GRID_LINES_COLOR, (0, y), (WIDTH, y))

        # Highlight buildings with white borders on environment_surface
        for x in range(environment.grid_width):
            for y in range(environment.grid_height):
                if environment.buildings[x, y] > 0:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(environment_surface, HIGHLIGHT_COLOR, rect, 1)  # White border

        # Draw drones on environment_surface
        for drone in drones:
            pygame.draw.circle(environment_surface, drone.color, (int(drone.position[0]), int(drone.position[1])), 5)
            # Optionally, draw drone ID
            id_text = FONT.render(str(drone.id), True, INFO_COLOR)
            environment_surface.blit(id_text, (drone.position[0] + 5, drone.position[1] - 10))

        # # Render weather effects on weather_surface
        # weather_surface.fill((0, 0, 0, 0))  # Clear previous weather
        # weather_system.render_weather_effects(weather_surface)

        # # Blit environment_surface and then weather_surface onto the main screen
        screen.blit(environment_surface, (0, 0))
        # screen.blit(weather_surface, (0, 0))  # Weather overlays on top

        # Draw UI elements directly on the main screen
        instructions = [
            "Press 's' to Save Environment",
            "Press 'l' to Load Environment",
            "Left Click: Add Building",
            "Right Click: Remove Building",
            "Press 'ESC' to Quit"
        ]

        stats = [
            f"Day: {time_manager.day_count + 1}",
            f"Time: {time_manager.hour:02}:{int(time_manager.minute):02}",
            f"Season: {time_manager.season}",
            f"Weather: {weather_system.current_state.name}",
            f"Intensity: {weather_system.current_state.intensity:.2f}",
            f"Temperature: {weather_system.current_state.temperature:.1f} °C",
            f"Humidity: {weather_system.current_state.humidity:.1f}%",
            f"Wind Speed: {weather_system.current_state.wind_speed:.1f} m/s",
            f"Wind Direction: {weather_system.current_state.wind_direction:.1f}°",
            f"Precipitation: {weather_system.current_state.precipitation_type}",
            f"Visibility: {weather_system.current_state.visibility:.2f}",
            f"Cloud Density: {weather_system.current_state.cloud_density:.2f}",
            f"Air Pressure: {weather_system.current_state.air_pressure:.1f} hPa",
            f"Drones: {len(drones)}"
        ]

        # Render instructions
        for idx, text in enumerate(instructions):
            text_surface = FONT.render(text, True, INFO_COLOR)
            screen.blit(text_surface, (10, HEIGHT - 20 * (len(instructions) - idx) - 140))

        # Render statistics
        y_offset = 10
        for stat in stats:
            stat_surface = FONT.render(stat, True, INFO_COLOR)
            screen.blit(stat_surface, (10, y_offset))
            y_offset += 20

        # Update the display
        pygame.display.flip()

    # Close the weather CSV file
    weather_system.close_csv()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        pygame.quit()
        sys.exit()
