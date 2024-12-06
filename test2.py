import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 10

# Colors
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
GREEN = (0, 255, 0)

# Initialize screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Agent Oil Spill Detection")

# Drone class
class Drone:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.speed = 1
        self.path = []
        self.id = id
        self.color = GREEN if id % 2 == 0 else RED

    def move(self, target_x, target_y):
        if self.x < target_x:
            self.x += self.speed
        elif self.x > target_x:
            self.x -= self.speed
        if self.y < target_y:
            self.y += self.speed
        elif self.y > target_y:
            self.y -= self.speed

    def detect_spill(self, oil_spills):
        for spill in oil_spills:
            if self.x == spill[0] and self.y == spill[1]:
                return spill
        return None

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Function to generate random path
def generate_random_path(cols, rows):
    return [(random.randint(0, cols - 1), random.randint(0, rows - 1)) for _ in range(5)]

# Main simulation function
def main():
    clock = pygame.time.Clock()
    running = True

    # Create grid
    cols, rows = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

    # Generate oil spills
    oil_spills = [(random.randint(0, cols-1), random.randint(0, rows-1)) for _ in range(10)]

    # Initialize drones
    num_drones = 10
    drones = [Drone(random.randint(0, cols-1), random.randint(0, rows-1), id=i) for i in range(num_drones)]

    while running:
        screen.fill(WHITE)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw grid
        for x in range(0, WIDTH, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

        # Draw oil spills
        for spill in oil_spills:
            pygame.draw.rect(screen, BLACK, (spill[0] * GRID_SIZE, spill[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Move drones and detect oil spills
        for drone in drones:
            if not drone.path:
                drone.path = generate_random_path(cols, rows)
            target_x, target_y = drone.path.pop(0)
            drone.move(target_x, target_y)

            detected_spill = drone.detect_spill(oil_spills)
            if detected_spill:
                print(f"Drone {drone.id} detected oil spill at {detected_spill}")
                oil_spills.remove(detected_spill)

            drone.draw()

        # Update display
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()

if __name__ == "__main__":
    main()
