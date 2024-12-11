import pygame
import numpy as np
import random
from scipy.ndimage import gaussian_filter

# Screen dimensions
WIDTH, HEIGHT = 800, 800
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Colors
WATER_COLOR = (70, 130, 180)  # Deep water
SHALLOW_WATER_COLOR = (100, 149, 237)  # Shallow water
LAND_COLOR = (34, 139, 34)  # Land
BUILDING_COLOR = (139, 69, 19)  # Buildings
GRID_LINES_COLOR = (200, 200, 200)  # Grid lines
OIL_SPILL_COLOR = (0, 0, 0)  # Oil spill (black)
DETECTED_OIL_COLOR = (255, 0, 0)  # Detected oil spill (red)
DRONE_COLOR = (0, 255, 255)  # Drone (cyan)

# Initialize screen
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Futuristic Grid Simulation with Oil Spillage")

# OilSpillage Class
class OilSpillage:
    def __init__(self, position):
        """
        Represents an oil spillage on the grid.
        :param position: Tuple (x, y) for grid cell coordinates.
        """
        self.position = position
        self.detected = False

    def detect(self, drone_positions, detection_radius):
        """
        Check if a drone has detected this spillage.
        :param drone_positions: List of drone positions as tuples.
        :param detection_radius: Detection radius in cells.
        :return: True if detected, False otherwise.
        """
        if not self.detected:
            for drone_pos in drone_positions:
                distance = np.linalg.norm(np.array(drone_pos) - np.array(self.position))
                if distance <= detection_radius:
                    self.detected = True
                    return True
        return False

# Generate the grid
def generate_grid(grid_width, grid_height, num_islands, max_radius):
    grid = np.zeros((grid_width, grid_height))

    # Create islands
    for _ in range(num_islands):
        cx, cy = random.randint(0, grid_width - 1), random.randint(0, grid_height - 1)
        radius = random.randint(3, max_radius)

        for x in range(grid_width):
            for y in range(grid_height):
                if np.sqrt((x - cx)**2 + (y - cy)**2) < radius:
                    grid[x, y] += random.uniform(0.5, 1.0)

    # Smooth terrain
    grid = gaussian_blur(grid, sigma=1)
    return grid

def gaussian_blur(grid, sigma):
    return gaussian_filter(grid, sigma=sigma)

# Add buildings
def add_buildings(grid, land_mask, num_buildings):
    buildings = np.zeros_like(grid)
    land_indices = np.argwhere(land_mask)

    for _ in range(num_buildings):
        if len(land_indices) == 0:
            break
        x, y = random.choice(land_indices)
        buildings[x, y] = random.uniform(1.0, 3.0)

    return buildings

# Add oil spillages
def add_oil_spillages(grid, water_mask, num_spillages):
    spillages = []
    water_indices = np.argwhere(water_mask)

    for _ in range(num_spillages):
        if len(water_indices) == 0:
            break
        x, y = random.choice(water_indices)
        spillages.append(OilSpillage((x, y)))

    return spillages

# Draw the grid
def draw_grid(screen, grid, buildings, oil_spillages, water_level):
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            elevation = grid[x, y]

            if elevation > water_level:
                pygame.draw.rect(screen, LAND_COLOR, rect)

                if buildings[x, y] > 0:
                    building_rect = pygame.Rect(
                        x * CELL_SIZE + CELL_SIZE // 4,
                        y * CELL_SIZE + CELL_SIZE // 4,
                        CELL_SIZE // 2,
                        CELL_SIZE // 2
                    )
                    pygame.draw.rect(screen, BUILDING_COLOR, building_rect)
            elif elevation > water_level - 0.1:
                pygame.draw.rect(screen, SHALLOW_WATER_COLOR, rect)
            else:
                pygame.draw.rect(screen, WATER_COLOR, rect)

    # Draw oil spillages
    for spillage in oil_spillages:
        color = DETECTED_OIL_COLOR if spillage.detected else OIL_SPILL_COLOR
        rect = pygame.Rect(
            spillage.position[0] * CELL_SIZE, spillage.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE
        )
        pygame.draw.rect(screen, color, rect)

# Draw drones
def draw_drones(screen, drone_positions):
    for drone_pos in drone_positions:
        center = (
            drone_pos[0] * CELL_SIZE + CELL_SIZE // 2,
            drone_pos[1] * CELL_SIZE + CELL_SIZE // 2,
        )
        pygame.draw.circle(screen, DRONE_COLOR, center, CELL_SIZE // 3)

# Draw grid lines
def draw_grid_lines(screen):
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_LINES_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_LINES_COLOR, (0, y), (WIDTH, y))

# Main simulation
def main():
    clock = pygame.time.Clock()
    running = True

    # Generate terrain and features
    grid = generate_grid(GRID_WIDTH, GRID_HEIGHT, num_islands=8, max_radius=10)
    water_level = 0.3
    water_mask = grid <= water_level
    land_mask = grid > water_level
    buildings = add_buildings(grid, land_mask, num_buildings=50)
    oil_spillages = add_oil_spillages(grid, water_mask, num_spillages=20)

    # Drones
    drone_positions = [np.random.randint(0, GRID_WIDTH, 2) for _ in range(3)]
    detection_radius = 3

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # Random drone movements
        for i in range(len(drone_positions)):
            drone_positions[i] += np.random.randint(-1, 2, size=2)
            drone_positions[i] = np.clip(drone_positions[i], 0, GRID_WIDTH - 1)

        # Detect oil spillages
        for spillage in oil_spillages:
            spillage.detect(drone_positions, detection_radius)

        # Draw the terrain, spillages, and drones
        draw_grid(screen, grid, buildings, oil_spillages, water_level)
        draw_drones(screen, drone_positions)
        draw_grid_lines(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
