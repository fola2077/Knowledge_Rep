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
WATER_COLOR = (70, 130, 180)
SHALLOW_WATER_COLOR = (100, 149, 237)
LAND_COLOR = (34, 139, 34)
BUILDING_COLOR = (139, 69, 19)
GRID_LINES_COLOR = (200, 200, 200)
OIL_SPILL_COLOR = (0, 0, 0)
DETECTED_OIL_COLOR = (255, 0, 0)
DRONE_COLOR = (0, 255, 255)
IDENTIFIED_OIL_COLOR = (0, 255, 0)  # Green for correctly identified spills
VIEW_RADIUS_COLOR = (255, 255, 0, 100)  # Transparent yellow for drone view

# Initialize screen
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sequential Decision Problem: Oil Spill Detection")

# OilSpillage Class
class OilSpillage:
    def __init__(self, position, lifespan=500):
        self.position = position
        self.detected = False
        self.identified = False
        self.lifespan = lifespan  # Time (frames) before the spill disappears

    def detect(self, drone_positions, detection_radius):
        if not self.detected:
            for drone_pos in drone_positions:
                distance = np.linalg.norm(np.array(drone_pos) - np.array(self.position))
                if distance <= detection_radius:
                    self.detected = True
                    return True
        return False

    def identify(self):
        if self.detected:
            self.identified = True

    def update_lifespan(self):
        if self.identified:
            self.lifespan -= 1
        return self.lifespan > 0

# Grid generation and feature addition
def generate_probability_grid(grid_width, grid_height, water_mask):
    probability_grid = np.zeros((grid_width, grid_height))
    probability_grid[water_mask] = np.random.uniform(0.2, 0.4, np.sum(water_mask))
    return probability_grid

def update_probabilities(probability_grid, checked_positions, water_mask, decay_factor=0.7, increase_factor=0.002):
    for pos in checked_positions:
        probability_grid[pos[0], pos[1]] *= decay_factor
    unchecked = np.logical_and(probability_grid > 0, water_mask)
    probability_grid[unchecked] += increase_factor * (1 - probability_grid[unchecked])
    probability_grid = np.clip(probability_grid, 0, 1)
    return probability_grid

def generate_grid(grid_width, grid_height, num_islands, max_radius):
    grid = np.zeros((grid_width, grid_height))
    for _ in range(num_islands):
        cx, cy = random.randint(0, grid_width - 1), random.randint(0, grid_height - 1)
        radius = random.randint(3, max_radius)
        for x in range(grid_width):
            for y in range(grid_height):
                if np.sqrt((x - cx)**2 + (y - cy)**2) < radius:
                    grid[x, y] += random.uniform(0.5, 1.0)
    return gaussian_filter(grid, sigma=1)

def add_buildings(grid, land_mask, num_buildings):
    buildings = np.zeros_like(grid)
    land_indices = np.argwhere(land_mask)
    for _ in range(num_buildings):
        if len(land_indices) == 0:
            break
        x, y = random.choice(land_indices)
        buildings[x, y] = random.uniform(1.0, 3.0)
    return buildings

def add_oil_spillages(grid, water_mask, num_spillages):
    spillages = []
    water_indices = np.argwhere(water_mask)
    for _ in range(num_spillages):
        if len(water_indices) == 0:
            break
        x, y = random.choice(water_indices)
        spillages.append(OilSpillage((x, y)))
    return spillages

# Agent class for rational AI drones
class DroneAgent:
    def __init__(self, num_drones, detection_radius, altitudes):
        self.positions = [np.random.uniform(0, GRID_WIDTH, 2) for _ in range(num_drones)]
        self.velocities = [np.zeros(2, dtype=float) for _ in range(num_drones)]
        self.detection_radius = detection_radius
        self.altitudes = altitudes  # Altitude for each drone
        self.checked_positions = []
        self.shared_knowledge = np.zeros((GRID_WIDTH, GRID_HEIGHT))  # Shared knowledge grid
        self.rewards = np.zeros(num_drones)  # Track rewards for each drone

    def choose_actions(self, probability_grid, land_mask):
        actions = []
        for i, pos in enumerate(self.positions):
            # Priority-based movement: high probabilities and unexplored areas
            x, y = int(pos[0]), int(pos[1])

            # Ensure drones recognize land and move over it without interpreting it as water/oil
            if land_mask[x, y]:
                gradient = np.array([0.0, 0.0])  # Ignore gradient on land, keep moving
            else:
                gradient = np.array([
                    probability_grid[min(x + 1, GRID_WIDTH - 1), y] - probability_grid[max(x - 1, 0), y],
                    probability_grid[x, min(y + 1, GRID_HEIGHT - 1)] - probability_grid[x, max(y - 1, 0)]
                ])

            # Normalize gradient and add noise for exploration
            if np.linalg.norm(gradient) > 0:
                gradient = gradient / np.linalg.norm(gradient)
            noise = np.random.uniform(-0.1, 0.1, size=2)
            action = gradient + noise

            # Update velocity to simulate smoother movement
            self.velocities[i] += 0.1 * action  # Smooth acceleration
            self.velocities[i] *= 0.9  # Damping for smooth deceleration
            actions.append(self.velocities[i])
        return actions

    def update_positions(self, actions):
        for i, action in enumerate(actions):
            self.positions[i] += action
            self.positions[i] = np.clip(self.positions[i], 0, GRID_WIDTH - 1)
            self.checked_positions.append(tuple(map(int, self.positions[i])))

    def update_shared_knowledge(self):
        # Share knowledge between drones
        for pos in self.checked_positions:
            self.shared_knowledge[pos[0], pos[1]] = 1  # Mark as checked

    def update_rewards(self, oil_spillages):
        for i, pos in enumerate(self.positions):
            x, y = int(pos[0]), int(pos[1])
            for spillage in oil_spillages:
                if spillage.detected and not spillage.identified and spillage.position == (x, y):
                    self.rewards[i] += 10  # Reward for identifying a spill
                elif self.shared_knowledge[x, y] == 1:
                    self.rewards[i] -= 1  # Penalty for redundant checks

    def get_view_radius(self, index):
        # The higher the altitude, the larger the view radius
        base_radius = self.detection_radius
        return base_radius + self.altitudes[index]

# Drawing functions
def draw_grid(screen, grid, buildings, probability_grid, water_level, oil_spillages):
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
    for spillage in oil_spillages:
        rect = pygame.Rect(
            spillage.position[0] * CELL_SIZE, spillage.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE
        )
        if spillage.identified:
            pygame.draw.rect(screen, IDENTIFIED_OIL_COLOR, rect)
        elif spillage.detected:
            pygame.draw.rect(screen, DETECTED_OIL_COLOR, rect)
        else:
            pygame.draw.rect(screen, OIL_SPILL_COLOR, rect)

def draw_drones(screen, drone_positions, drone_altitudes, agent):
    for i, drone_pos in enumerate(drone_positions):
        center = (
            int(drone_pos[0] * CELL_SIZE + CELL_SIZE // 2),
            int(drone_pos[1] * CELL_SIZE + CELL_SIZE // 2),
        )
        pygame.draw.circle(screen, DRONE_COLOR, center, CELL_SIZE // 3)
        view_radius = agent.get_view_radius(i) * CELL_SIZE
        pygame.draw.circle(screen, VIEW_RADIUS_COLOR[:-1], center, int(view_radius), 1)  # Visualize view radius

def draw_grid_lines(screen):
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRID_LINES_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRID_LINES_COLOR, (0, y), (WIDTH, y))

# Main simulation
def main():
    clock = pygame.time.Clock()
    running = True

    grid = generate_grid(GRID_WIDTH, GRID_HEIGHT, num_islands=8, max_radius=10)
    water_level = 0.3
    water_mask = grid <= water_level
    land_mask = grid > water_level
    buildings = add_buildings(grid, land_mask, num_buildings=50)
    probability_grid = generate_probability_grid(GRID_WIDTH, GRID_HEIGHT, water_mask)
    oil_spillages = add_oil_spillages(grid, water_mask, num_spillages=20)

    agent = DroneAgent(num_drones=3, detection_radius=3, altitudes=[2, 3, 4])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        actions = agent.choose_actions(probability_grid, land_mask)
        agent.update_positions(actions)
        agent.update_shared_knowledge()
        agent.update_rewards(oil_spillages)
        probability_grid = update_probabilities(probability_grid, agent.checked_positions, water_mask)

        # Update spills and remove expired ones
        oil_spillages = [spill for spill in oil_spillages if spill.update_lifespan()]

        # Add new random spills
        if random.random() < 0.01:  # Low probability of new spill each frame
            oil_spillages += add_oil_spillages(grid, water_mask, num_spillages=1)

        for spillage in oil_spillages:
            if spillage.detect(agent.positions, agent.detection_radius):
                spillage.identify()

        draw_grid(screen, grid, buildings, probability_grid, water_level, oil_spillages)
        draw_drones(screen, agent.positions, agent.altitudes, agent)
        draw_grid_lines(screen)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
