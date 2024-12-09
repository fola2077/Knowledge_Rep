import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Main Window Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Weather Simulation")

# Stats Window Dimensions
STATS_WIDTH, STATS_HEIGHT = 300, 200
stats_screen = pygame.display.set_mode((STATS_WIDTH, STATS_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Weather Stats")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors
DAY_COLOR = (135, 206, 235)  # Sky Blue
NIGHT_COLOR = (25, 25, 112)  # Dark Blue
RAIN_COLOR = (0, 0, 255)  # Blue for raindrops
CLOUD_COLOR = (200, 200, 200)  # Light Grey for clouds

# Font
font = pygame.font.Font(None, 36)


# Helper Function: Draw Text
def draw_text(surface, text, font, color, x, y):
    """Draw text on the screen."""
    rendered_text = font.render(text, True, color)
    surface.blit(rendered_text, (x, y))


class Weather:
    def __init__(self):
        self.time_of_day = "day"
        self.weather_condition = "sunny"
        self.wind_speed = random.uniform(1, 5)
        self.wind_direction = random.choice([-1, 1])
        self.raindrops = []
        self.clouds = []
        self.weather_change_interval = 20  # Seconds between weather changes
        self.last_weather_change = time.time()
        self.clouds = [pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)) for _ in range(10)]

    def toggle_time_of_day(self):
        """Switch between day and night."""
        self.time_of_day = "night" if self.time_of_day == "day" else "day"

    def maybe_change_weather(self):
        """Change weather condition at intervals."""
        current_time = time.time()
        if current_time - self.last_weather_change >= self.weather_change_interval:
            self.weather_condition = random.choice(["sunny", "rainy", "cloudy", "foggy", "stormy", "windy", "heatwave"])
            self.last_weather_change = current_time
            print(f"Weather changed to: {self.weather_condition}")

            # Adjust raindrops or clouds
            if self.weather_condition in ["rainy", "stormy"]:
                self.raindrops = [pygame.Vector2(random.randint(0, WIDTH), random.randint(-50, HEIGHT)) for _ in range(100)]
            elif self.weather_condition in ["cloudy", "foggy"]:
                self.clouds = [pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)) for _ in range(10)]
            else:
                self.raindrops = []
                self.clouds = []

    def update_effects(self):
        """Update weather effects based on conditions."""
        # Update raindrops
        for drop in self.raindrops:
            drop.y += 5  # Fall speed
            drop.x += self.wind_speed * self.wind_direction
            if drop.y > HEIGHT or drop.x < 0 or drop.x > WIDTH:
                drop.y = random.randint(-20, -1)
                drop.x = random.randint(0, WIDTH)

        # Update clouds
        for cloud in self.clouds:
            cloud.x += self.wind_speed * self.wind_direction
            if cloud.x > WIDTH + 100:
                cloud.x = -200
            elif cloud.x < -200:
                cloud.x = WIDTH + 100

    def draw_effects(self, screen):
        """Draw weather effects."""
        for cloud in self.clouds:
            pygame.draw.ellipse(screen, CLOUD_COLOR, (cloud.x, cloud.y, 120, 60))
        for drop in self.raindrops:
            pygame.draw.line(screen, RAIN_COLOR, (drop.x, drop.y), (drop.x, drop.y + 10), 2)

    def get_stats(self):
        """Return current weather stats."""
        wind_dir = "Right" if self.wind_direction > 0 else "Left"
        return {
            "Time of Day": self.time_of_day.capitalize(),
            "Weather": self.weather_condition.capitalize(),
            "Wind Speed": f"{self.wind_speed:.1f} m/s",
            "Wind Direction": wind_dir,
        }


# Initialize Weather
weather = Weather()

# Main Game Loop
running = True
last_update_time = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update time
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time

    # Update weather
    weather.maybe_change_weather()
    weather.update_effects()

    # Draw the environment in the main window
    screen.fill(DAY_COLOR if weather.time_of_day == "day" else NIGHT_COLOR)
    weather.draw_effects(screen)

    # Draw stats in the stats window
    stats_screen.fill((50, 50, 50))  # Dark background for stats window
    stats = weather.get_stats()
    y_offset = 10
    for key, value in stats.items():
        draw_text(stats_screen, f"{key}: {value}", font, (255, 255, 255), 10, y_offset)
        y_offset += 30

    # Update both windows
    pygame.display.update()
    clock.tick(60)

pygame.quit()
