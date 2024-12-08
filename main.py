import pygame
import random 
import math
import time
import noise # For Perlin noise generation

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Weather Simulation with Day/Night Cycles")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors
DAY_COLOR = (135, 206, 235)  # Sky Blue
NIGHT_COLOR = (25, 25, 112)  # Dark Blue
RAIN_COLOR = (0, 0, 255)  # Blue for raindrops
CLOUD_COLOR = (200, 200, 200)  # Light Grey for clouds
SUN_COLOR = (255, 223, 0)  # Yellow for sun
MOON_COLOR = (255, 255, 224)  # Light yellow for moon


class MultiAgentDrone:
    None

class Environment:

    None

# Weather Class
class Weather:
    def __init__(self):
        # Time attributes
        self.time_of_day = "day"  # Options: "day", "night"
        self.cycle_duration = 15  # Duration of day/night in seconds
        self.current_time = 0

        # Weather attributes
        self.weather_condition = "sunny"  # Options: "sunny", "rainy", "cloudy"
        self.background_color = DAY_COLOR

        # Effects
        self.raindrops = []
        self.clouds = [pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)) for _ in range(8)]

        # Transition
        self.transition_color = list(DAY_COLOR)  # Used for smooth transitions

    def toggle_time_of_day(self):
        """Switch between day and night."""
        if self.time_of_day == "day":
            self.time_of_day = "night"
            self.background_color = NIGHT_COLOR
        else:
            self.time_of_day = "day"
            self.background_color = DAY_COLOR

    def update_cycle(self, dt):
        """Update time of day and smooth transition."""
        self.current_time += dt
        if self.current_time >= self.cycle_duration:
            self.current_time = 0
            self.toggle_time_of_day()

        # Smooth color transition
        target_color = NIGHT_COLOR if self.time_of_day == "night" else DAY_COLOR
        for i in range(3):
            if self.transition_color[i] < target_color[i]:
                self.transition_color[i] += 1
            elif self.transition_color[i] > target_color[i]:
                self.transition_color[i] -= 1
        self.background_color = tuple(self.transition_color)

    def change_weather(self):
        """Randomly change the weather condition."""
        self.weather_condition = random.choice(["sunny", "rainy", "cloudy"])
        if self.weather_condition == "rainy":
            self.raindrops = [pygame.Vector2(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(100)]
        elif self.weather_condition == "cloudy":
            self.clouds = [pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)) for _ in range(5)]
        else:
            self.raindrops = []
            self.clouds = []

    def draw(self, screen):
        """Render the current weather."""
        # Draw the sun or moon
        if self.time_of_day == "day":
            pygame.draw.circle(screen, SUN_COLOR, (WIDTH - 100, 100), 50)
        else:
            pygame.draw.circle(screen, MOON_COLOR, (WIDTH - 100, 100), 50)

        # Draw clouds
        if self.weather_condition == "cloudy":
            for cloud in self.clouds:
                pygame.draw.ellipse(screen, CLOUD_COLOR, (cloud.x, cloud.y, 120, 60))

        # Draw raindrops
        if self.weather_condition == "rainy":
            for drop in self.raindrops:
                pygame.draw.line(screen, RAIN_COLOR, (drop.x, drop.y), (drop.x, drop.y + 10), 2)

    def update_weather_effects(self):
        """Update raindrop positions."""
        if self.weather_condition == "rainy":
            for drop in self.raindrops:
                drop.y += 5  # Raindrop speed
                if drop.y > HEIGHT:
                    drop.y = random.randint(-20, -1)
                    drop.x = random.randint(0, WIDTH)


        # Update cloud movement
        if self.weather_condition in ["cloudy", "rainy"]:
            for cloud in self.clouds:
                cloud.x += 0.5
                if cloud.x > WIDTH + 100:
                    cloud.x = -200

# Initialize the Weather object
weather = Weather()

# Main Game Loop
running = True
last_update_time = time.time()
weather.change_weather()  # Set initial weather

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get delta time for smooth updates
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time

    # Update weather cycle and effects
    weather.update_cycle(dt)
    weather.update_weather_effects()

    # Periodically change weather
    if random.randint(0, 300) == 0:  # Random chance to change weather
        weather.change_weather()

    # Draw everything
    screen.fill(weather.background_color)
    weather.draw(screen)

    # Refresh screen
    pygame.display.flip()
    clock.tick(60)

pygame.quit()

