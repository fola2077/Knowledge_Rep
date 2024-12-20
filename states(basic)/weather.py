import pygame
import random
import math
import time

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Probabilistic Weather System - Niger Delta")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors
DAY_COLOR = (135, 206, 235)
NIGHT_COLOR = (25, 25, 112)
CLOUD_COLOR = (200, 200, 200)
RAIN_COLOR = (0, 0, 255)
FOG_COLOR = (180, 180, 180, 100)
HAZE_COLOR = (210, 180, 140, 70)
FPS = 60 #frames per second

class TimeManager:
    """Manages the simulation's time system."""
    def __init__(self):
        self.hour = 6
        self.minute = 0
        self.day_duration = 24  # Hours in a day

    def update(self, dt):
        """Update the time of day."""
        self.minute += dt * 10  # Accelerated time progression
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
        if self.hour >= self.day_duration:
            self.hour = 0

    def is_daytime(self):
        """Determine if it is daytime."""
        return 6 <= self.hour < 18


# Helper Function: Draw Text
def draw_text(surface, text, font, color, x, y):
    """Draw text on a surface."""
    rendered_text = font.render(text, True, color)
    surface.blit(rendered_text, (x, y))


class WeatherSystem:
    def __init__(self, season="rainy"):
        # Time and Season
        self.time_manager = TimeManager() # Unified time management
        self.season = season
        self.hour = 6  # 6 AM start
        self.minute = 0
        self.time_of_day = "day"  # 'day', 'night'
        self.cycle_duration = 24  # 24-hour simulation
        self.last_update_time = time.time()


        # Weather Attributes
        self.weather_condition = "sunny"
        self.dynamic_condition = None
        self.temperature = 27  # Default to a warm average
        self.humidity = 85  # High humidity typical of the Niger Delta
        self.wind_speed = random.uniform(1, 5)  # m/s
        self.wind_direction_angle = random.uniform(0, 360)
        self.precipitation = 0  # mm/hour
        self.weather_intensity = 0.1  # Scales effects, 0.1 (mild) to 1.0 (severe)

        # Weather Transition Timers
        self.weather_change_interval = 20  # Seconds
        self.last_weather_change = time.time()

        # Effects
        self.clouds = [self.create_cloud() for _ in range(random.randint(5, 15))]
        self.raindrops = []

    def create_cloud(self):
        """
        Create a cloud with random properties.
        """
        return {
            "position": pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)),
            "size": random.uniform(80, 150),
            "speed": random.uniform(0.5, 2.0),
        }

    def update_time(self, dt):
        """
        Update time using TimeManager.
        """
        self.time_manager.update(dt)
        self.update_time_of_day()

    def update_time_of_day(self):
        """
        Update the time of day based on TimeManager's hour.
        """
        self.time_of_day = "day" if self.time_manager.is_daytime() else "night"
        
    def change_season(self):
        """Change the season based on time or a defined rule."""
        if self.season == "rainy":
            self.season = "dry"
            print("Season changed to Dry Season.")
        else:
            self.season = "rainy"
            print("Season changed to Rainy Season.")
        self.adjust_for_season()

    def adjust_for_season(self):
        """Adjust weather attributes based on the current season."""
        if self.season == "rainy":
            self.temperature = random.uniform(25, 32)
            self.humidity = random.randint(80, 95)
            self.precipitation_probability = random.uniform(0.7, 0.9)
        elif self.season == "dry":
            self.temperature = random.uniform(21, 30)
            self.humidity = random.randint(40, 60)
            self.precipitation_probability = random.uniform(0.1, 0.2)

    def update_weather(self):
        """Update weather condition probabilistically."""
        current_time = time.time()
        if current_time - self.last_weather_change >= self.weather_change_interval:
            self.last_weather_change = current_time

            # Determine the weather condition based on probabilities
            if self.season == "rainy":
                probabilities = {
                    "sunny": 0.2,
                    "rainy": 0.5,
                    "foggy": 0.1,
                    "windy": 0.2,
                }
            else:  # Dry season
                probabilities = {
                    "sunny": 0.5,
                    "rainy": 0.1,
                    "foggy": 0.1,
                    "windy": 0.3,
                }

            self.weather_condition = self.weighted_choice(probabilities)
            self.dynamic_condition = self.determine_dynamic_condition()

            # Adjust intensity and temperature
            self.weather_intensity = random.uniform(0.1, 1.0)
            self.adjust_temperature_and_humidity()

    def weighted_choice(self, probabilities):
        """Select a weather condition based on probabilities."""
        total = sum(probabilities.values())
        rand = random.uniform(0, total)
        cumulative = 0
        for condition, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return condition

    def determine_dynamic_condition(self):
        """Determine secondary dynamic weather conditions."""
        if self.weather_condition == "rainy" and random.random() < 0.3:
            return "heavystorms"
        elif self.weather_condition == "rainy" and random.random() < 0.5:
            return "rain + foggy"
        elif self.weather_condition == "sunny" and random.random() < 0.2:
            return "sunny + windy"
        elif self.season == "dry" and random.random() < 0.3:
            return "harmattan"
        return None

    def adjust_temperature_and_humidity(self):
        """Adjust temperature and humidity based on conditions."""
        if self.weather_condition == "rainy":
            self.temperature -= random.uniform(1, 3)
            self.humidity = random.randint(80, 95)
        elif self.weather_condition == "sunny":
            self.temperature += random.uniform(1, 4)
            self.humidity = random.randint(40, 60)
        elif self.weather_condition == "foggy":
            self.temperature -= random.uniform(0.5, 1.5)
            self.humidity = random.randint(90, 100)

    def draw_weather(self, screen):
        """Visualize weather effects."""
        # Background
        bg_color = DAY_COLOR if self.time_of_day == "day" else NIGHT_COLOR
        screen.fill(bg_color)

        # Clouds
        for cloud in self.clouds:
            pygame.draw.ellipse(
                screen, CLOUD_COLOR, (cloud["position"].x, cloud["position"].y, cloud["size"], cloud["size"] // 2)
            )
            cloud["position"].x += cloud["speed"]
            if cloud["position"].x > WIDTH:
                cloud["position"].x = -cloud["size"]


    def draw_rain(self, screen):
        for drop in self.raindrops:
            pygame.draw.line(screen, RAIN_COLOR, drop["start"], drop["end"], 2)
            drop["start"].y += drop["speed"]
            drop["end"].y += drop["speed"]
            if drop["start"].y > HEIGHT:
                drop["start"].y = random.randint(-10, 0)
                drop["end"].y = drop["start"].y + 10


    def draw_fog(self, screen):
        """Render fog overlay."""
        fog_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fog_surface.fill((180, 180, 180, 80))
        screen.blit(fog_surface, (0, 0))

    def get_stats(self):
        """Return current weather stats."""
        return {
            "Time": f"{time_manager.hour:02}:{time_manager.minute:02}",
            "Time of Day": self.time_of_day.capitalize(),
            "Weather": self.weather_condition.capitalize(),
            "Dynamic Condition": self.dynamic_condition or "None",
            "Temperature": f"{self.temperature:.1f} °C",
            "Humidity": f"{self.humidity}%",
            "Wind Speed": f"{self.wind_speed:.1f} m/s",
            "Wind Direction": f"{self.wind_direction_angle:.1f}°",
        }


# Initialize the weather system
weather = WeatherSystem(season="rainy")
font = pygame.font.Font(None, 24)

 # Initialize components
time_manager = TimeManager()
weather = WeatherSystem(season="rainy")

# Main Loop
running = True
last_update_time = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Time and delta updates
    current_time = time.time()
    dt = current_time - last_update_time
    last_update_time = current_time

    # Update system
    weather.update_time(dt)
    weather.update_weather()

    # Draw weather and stats
    weather.draw_weather(screen)
    stats = weather.get_stats()
    y_offset = 10
    for key, value in stats.items():
        draw_text(screen, f"{key}: {value}", font, (255, 255, 255), 10, y_offset)
        y_offset += 30

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
