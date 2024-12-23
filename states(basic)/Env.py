import pygame
import numpy as np
import random
import math
from scipy.ndimage import gaussian_filter
import sys
import csv
import time

# Constants
WIDTH, HEIGHT = 1200, 800
CELL_SIZE = 10
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE
WATER_LEVEL = 0.3
FPS = 60

# Colors
WATER_COLOR = (70, 130, 180)          # Deep water
SHALLOW_WATER_COLOR = (100, 149, 237) # Shallow water
LAND_COLOR = (34, 139, 34)            # Land
BUILDING_COLOR = (139, 69, 19)        # Buildings
GRID_LINES_COLOR = (200, 200, 200)    # Grid lines
HIGHLIGHT_COLOR = (255, 255, 255)     # Highlight color for buildings
INFO_COLOR = (255, 255, 255)          # White color for info text

# Initialize Pygame fonts
pygame.font.init()
FONT = pygame.font.Font(None, 24)

class Grid:
    def __init__(self, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT, num_islands=8, max_radius=10, num_buildings=50):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_islands = num_islands
        self.max_radius = max_radius
        self.num_buildings = num_buildings
        self.grid = self.generate_grid()
        self.land_mask = self.grid > WATER_LEVEL
        self.buildings = self.add_buildings()
        self.print_grid_stats()
    
    def generate_grid(self):
        grid = np.zeros((self.grid_width, self.grid_height))
    
        # Create islands
        for _ in range(self.num_islands):
            cx, cy = random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1)
            radius = random.randint(3, self.max_radius)
            for x in range(cx - radius, cx + radius + 1):
                for y in range(cy - radius, cy + radius + 1):
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if distance < radius:
                            grid[x, y] += random.uniform(0.5, 1.0)
        
        # Force some land for testing
        for i in range(10):
            if i < self.grid_width and i < self.grid_height:
                grid[i, i] = WATER_LEVEL + 0.5  # Ensure these cells are land
    
        # Smooth terrain
        grid = gaussian_filter(grid, sigma=1)
        return grid
    
    def add_buildings(self):
        buildings = np.zeros_like(self.grid)
        land_indices = np.argwhere(self.land_mask)
    
        for _ in range(self.num_buildings):
            if len(land_indices) == 0:
                break
            x_idx, y_idx = random.choice(land_indices)
            buildings[x_idx, y_idx] = random.uniform(1.0, 3.0)
    
        return buildings
    
    def draw(self, screen):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                elevation = self.grid[x, y]
    
                if elevation > WATER_LEVEL:
                    pygame.draw.rect(screen, LAND_COLOR, rect)
    
                    if self.buildings[x, y] > 0:
                        building_rect = pygame.Rect(
                            x * CELL_SIZE + CELL_SIZE // 4,
                            y * CELL_SIZE + CELL_SIZE // 4,
                            CELL_SIZE // 2,
                            CELL_SIZE // 2
                        )
                        pygame.draw.rect(screen, BUILDING_COLOR, building_rect)
                elif elevation > WATER_LEVEL - 0.1:
                    pygame.draw.rect(screen, SHALLOW_WATER_COLOR, rect)
                else:
                    pygame.draw.rect(screen, WATER_COLOR, rect)
    
    def draw_grid_lines(self, screen):
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(screen, GRID_LINES_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, GRID_LINES_COLOR, (0, y), (WIDTH, y))
    
    def save_environment(self, filename='environment.npy'):
        environment_data = {
            'grid': self.grid,
            'buildings': self.buildings
        }
        np.save(filename, environment_data)
        print(f"Environment saved to '{filename}'.")
    
    def load_environment(self, filename='environment.npy'):
        try:
            environment_data = np.load(filename, allow_pickle=True).item()
            self.grid = environment_data['grid']
            self.buildings = environment_data['buildings']
            self.land_mask = self.grid > WATER_LEVEL
            print(f"Environment loaded from '{filename}'.")
        except Exception as e:
            print(f"Failed to load environment from '{filename}': {e}")
            # If loading fails, generate a new grid
            self.grid = self.generate_grid()
            self.land_mask = self.grid > WATER_LEVEL
            self.buildings = self.add_buildings()
            self.print_grid_stats()
    
    def print_grid_stats(self):
        print("=== Grid Statistics ===")
        print(f"Min elevation: {self.grid.min():.2f}")
        print(f"Max elevation: {self.grid.max():.2f}")
        print(f"Mean elevation: {self.grid.mean():.2f}")
        print(f"Total land cells: {np.sum(self.land_mask)}")
        print(f"Total buildings: {np.sum(self.buildings > 0)}")
        print("========================")

class Simulation:
    def __init__(self, num_islands=8, max_radius=10, num_buildings=50):
        """
        Initialize the simulation with grid parameters.
        """
        self.grid = Grid(num_islands=num_islands, max_radius=max_radius, num_buildings=num_buildings)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Grid Visualization")
        self.clock = pygame.time.Clock()
        self.font = FONT
        self.running = True
    
    def run(self):
        """
        Run the main simulation loop.
        """
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # Handle key presses
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_s:
                        self.grid.save_environment()
                    elif event.key == pygame.K_l:
                        self.grid.load_environment()
                # Handle mouse clicks for interactivity
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click to add building
                        self.add_building(event.pos)
                    elif event.button == 3:  # Right click to remove building
                        self.remove_building(event.pos)
            
            # Clear screen
            self.screen.fill((0, 0, 0))  # Black background

            # Draw terrain and buildings
            self.grid.draw(self.screen)

            # Draw grid lines
            self.grid.draw_grid_lines(self.screen)

            # Highlight buildings
            self.highlight_buildings()

            # Draw UI
            self.draw_ui()

            # Update the display
            pygame.display.flip()

            # Control frame rate
            self.clock.tick(FPS)
    
        pygame.quit()
        sys.exit()
    
    def add_building(self, pos):
        """
        Add a building at the clicked position if it's on land.
        """
        x, y = pos
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE
        if 0 <= grid_x < self.grid.grid_width and 0 <= grid_y < self.grid.grid_height:
            if self.grid.grid[grid_x, grid_y] > WATER_LEVEL:
                self.grid.buildings[grid_x, grid_y] = random.uniform(1.0, 3.0)
                print(f"Added building at ({x}, {y}).")
            else:
                print(f"Cannot add building at ({x}, {y}) - Not on land.")
    
    def remove_building(self, pos):
        """
        Remove a building at the clicked position.
        """
        x, y = pos
        grid_x = x // CELL_SIZE
        grid_y = y // CELL_SIZE
        if 0 <= grid_x < self.grid.grid_width and 0 <= grid_y < self.grid.grid_height:
            if self.grid.buildings[grid_x, grid_y] > 0:
                self.grid.buildings[grid_x, grid_y] = 0.0
                print(f"Removed building at ({x}, {y}).")
            else:
                print(f"No building to remove at ({x}, {y}).")
    
    def highlight_buildings(self):
        """
        Highlight buildings with a white border.
        """
        for x in range(self.grid.grid_width):
            for y in range(self.grid.grid_height):
                if self.grid.buildings[x, y] > 0:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 1)  # White border
    
    def draw_ui(self):
        """
        Draw user instructions and grid statistics on the screen.
        """
        instructions = [
            "Press 's' to Save Environment",
            "Press 'l' to Load Environment",
            "Left Click: Add Building",
            "Right Click: Remove Building",
            "Press 'ESC' to Quit"
        ]
        
        stats = [
            f"Min Elevation: {self.grid.grid.min():.2f}",
            f"Max Elevation: {self.grid.grid.max():.2f}",
            f"Mean Elevation: {self.grid.grid.mean():.2f}",
            f"Land Cells: {np.sum(self.grid.land_mask)}",
            f"Buildings: {np.sum(self.grid.buildings > 0)}"
        ]
        
        # Draw instructions
        for idx, text in enumerate(instructions):
            text_surface = self.font.render(text, True, INFO_COLOR)
            self.screen.blit(text_surface, (10, HEIGHT - 20 * (len(instructions) - idx) - 100))
        
        # Draw statistics
        for idx, stat in enumerate(stats):
            stat_surface = self.font.render(stat, True, INFO_COLOR)
            self.screen.blit(stat_surface, (10, 10 + 20 * idx))

def main():
    try:
        simulation = Simulation()
        simulation.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
