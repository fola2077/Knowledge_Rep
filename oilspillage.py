import pygame
import numpy as np

# Constants for visualization
WINDOW_SIZE = 500
DRONE_RADIUS = 10
SPILL_RADIUS = 15
DETECTION_RADIUS = 50
FPS = 30

class DroneEnvironment:
    def __init__(self, num_drones=3, num_spillages=5):
        self.num_drones = num_drones
        self.num_spillages = num_spillages
        self.drones = []
        self.spillages = []
        self.done = False

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("Drone Oil Spillage Detection")
        self.clock = pygame.time.Clock()

        # Initialize drones and spillages
        self.initialize_drones()
        self.initialize_spillages()

    def initialize_drones(self):
        """Initialize drones at random positions."""
        self.drones = [np.random.uniform(0, WINDOW_SIZE, 2) for _ in range(self.num_drones)]

    def initialize_spillages(self):
        """Initialize oil spillages at random positions."""
        self.spillages = [{"position": np.random.uniform(0, WINDOW_SIZE, 2), "detected": False} for _ in range(self.num_spillages)]

    def reset(self):
        """Reset the environment."""
        self.initialize_drones()
        for spillage in self.spillages:
            spillage["detected"] = False

    def step(self, actions):
        """Move drones and check for oil spillage detection."""
        rewards = []
        for i, action in enumerate(actions):
            if action == 1: self.drones[i][1] -= 5  # Move up
            elif action == 2: self.drones[i][1] += 5  # Move down
            elif action == 3: self.drones[i][0] -= 5  # Move left
            elif action == 4: self.drones[i][0] += 5  # Move right

            # Keep drones within bounds
            self.drones[i] = np.clip(self.drones[i], 0, WINDOW_SIZE)

            # Check for spillage detection
            reward = 0
            for spillage in self.spillages:
                if not spillage["detected"]:
                    distance = np.linalg.norm(self.drones[i] - spillage["position"])
                    if distance <= DETECTION_RADIUS:
                        spillage["detected"] = True
                        reward += 10
            rewards.append(reward)

        return self.get_state(), rewards, self.done, {}

    def get_state(self):
        """Return the current positions of the drones."""
        return np.array(self.drones)

    def render(self):
        """Render the environment using Pygame."""
        self.screen.fill((255, 255, 255))  # Clear screen with white background

        # Draw oil spillages
        for spillage in self.spillages:
            color = (255, 0, 0) if not spillage["detected"] else (0, 255, 0)  # Red for undetected, Green for detected
            pygame.draw.circle(self.screen, color, spillage["position"].astype(int), SPILL_RADIUS)

        # Draw drones
        for drone in self.drones:
            pygame.draw.circle(self.screen, (0, 0, 255), drone.astype(int), DRONE_RADIUS)

        # Draw detection radius
        for drone in self.drones:
            pygame.draw.circle(self.screen, (0, 0, 255, 50), drone.astype(int), DETECTION_RADIUS, 1)

        pygame.display.flip()
        self.clock.tick(FPS)

    def close(self):
        """Close the Pygame window."""
        pygame.quit()


# Main block to run the simulation
if __name__ == "__main__":
    env = DroneEnvironment()

    # Example random actions
    actions = [0] * env.num_drones  # Replace with actual action logic (0 for no action)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Take a step in the environment
        state, rewards, done, _ = env.step(actions)

        # Render the environment
        env.render()

    env.close()
