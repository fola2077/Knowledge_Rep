import pygame
import random
import math
import time
import csv
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Weather System Simulation")

# Clock for controlling the frame rate
clock = pygame.time.Clock()

# Colors for Different Weather Conditions
DAY_COLOR = (135, 206, 235)
NIGHT_COLOR = (25, 25, 112)
CLOUD_COLOR = (220, 220, 220)
RAIN_COLOR = (0, 0, 255)
SNOW_COLOR = (255, 250, 250)
FOG_COLOR = (180, 180, 180, 80)
HAZE_COLOR = (210, 180, 140, 70)
STORM_COLOR = (50, 50, 50)
LIGHTNING_COLOR = (255, 255, 255)
HARMATTAN_COLOR = (210, 180, 140, 100)

SKY_COLORS = {
    "sunrise": (255, 223, 186),
    "day": DAY_COLOR,
    "sunset": (255, 140, 0),
    "night": NIGHT_COLOR
}


# Fonts
FONT = pygame.font.Font(None, 24)

# Frame Rate
FPS = 60

# Seasons
SEASONS = ["Rainy", "Dry"]

# Time Management
class TimeManager:
    """Manages the simulation's time system."""
    def __init__(self):
        self.hour = 6  # Start at 6 AM
        self.minute = 0
        self.day_duration = 24  # Hours in a day
        self.season = "Rainy"  # Start with Rainy season
        self.season_duration = 30  # Days per season
        self.day_count = 0

    def update(self, dt):
        """Update the time of day."""
        self.minute += dt * 10  # Accelerated time progression
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
            self.check_hourly_events()

        if self.hour >= self.day_duration:
            self.hour = 0
            self.day_count += 1
            self.check_season_change()

    def update(self, wind_speed, wind_direction):
        angle_rad = math.radians(wind_direction)
        wind_dx = math.cos(angle_rad) * wind_speed
        wind_dy = math.sin(angle_rad) * wind_speed
        self.x += wind_dx * 0.1
        self.y += (self.speed + wind_dy) * 0.1
        if self.y > HEIGHT:
            self.reset()


    def is_daytime(self):
        """Determine if it is daytime."""
        return 6 <= self.hour < 18

    def check_season_change(self):
        """Change season after a defined duration."""
        if self.day_count >= self.season_duration:
            self.day_count = 0
            self.change_season()

    def change_season(self):
        """Toggle between Rainy and Dry seasons."""
        self.season = "Dry" if self.season == "Rainy" else "Rainy"
        print(f"Season changed to {self.season} Season.")

    def check_hourly_events(self):
        """Trigger events that occur every hour."""
        # Placeholder for any hourly events (e.g., logging)
        pass

# Weather State Definition
class WeatherState:
    """Represents a weather state with associated attributes."""
    def __init__(self, name, temperature_range, humidity_range, wind_speed_range,
                 precipitation_type, visibility, cloud_density):
        self.name = name
        self.temperature = random.uniform(*temperature_range)
        self.humidity = random.uniform(*humidity_range)
        self.wind_speed = random.uniform(*wind_speed_range)
        self.wind_direction = random.uniform(*wind_direction_range)  # Degrees
        self.precipitation_type = precipitation_type  # e.g., 'None', 'Rain'
        self.visibility = visibility  # Affects sensor accuracy
        self.cloud_density = cloud_density  # Affects visibility
        self.intensity = 0.5  # Initial intensity (0 to 1)

    def increase_intensity(self, amount=0.01):
        """Gradually increase the intensity of the weather state."""
        self.intensity = min(1.0, self.intensity + amount)

    def decrease_intensity(self, amount=0.01):
        """Gradually decrease the intensity of the weather state."""
        self.intensity = max(0.0, self.intensity - amount)

# Weather System with Gradual Transitions and Intensity-Dependent Probabilities
class WeatherSystem:
    def __init__(self, time_manager):
        self.time_manager = time_manager

        # Define possible weather states
        self.states = {
            "Sunny": WeatherState(
                "Sunny",
                temperature_range=(25, 35),
                humidity_range=(30, 50),
                wind_speed_range=(5, 15),
                precipitation_type="None",
                visibility=1.0,
                cloud_density=0.2
            ),
            "Rainy": WeatherState(
                "Rainy",
                temperature_range=(20, 30),
                humidity_range=(70, 100),
                wind_speed_range=(10, 25),
                precipitation_type="Rain",
                visibility=0.6,
                cloud_density=0.7
            ),
            "Foggy": WeatherState(
                "Foggy",
                temperature_range=(15, 25),
                humidity_range=(80, 100),
                wind_speed_range=(0, 5),
                precipitation_type="None",
                visibility=0.3,
                cloud_density=0.5
            ),
            "Stormy": WeatherState(
                "Stormy",
                temperature_range=(18, 28),
                humidity_range=(75, 100),
                wind_speed_range=(20, 40),
                precipitation_type="Rain",
                visibility=0.5,
                cloud_density=0.8
            ),
            "Harmattan": WeatherState(
                "Harmattan",
                temperature_range=(15, 25),
                humidity_range=(20, 40),
                wind_speed_range=(10, 20),
                precipitation_type="None",
                visibility=0.4,
                cloud_density=0.3
            )
        }

        # Initialize current weather state based on the current season
        self.current_state = self.initialize_state()

        # Define transition probabilities based on current state and intensity
        self.transition_probabilities = {
            "Rainy": {
                "Sunny": 0.2,
                "Rainy": 0.5,
                "Stormy": 0.2,
                "Foggy": 0.1
            },
            "Sunny": {
                "Sunny": 0.6,
                "Rainy": 0.2,
                "Harmattan": 0.1,
                "Foggy": 0.1
            },
            "Stormy": {
                "Stormy": 0.5,
                "Rainy": 0.3,
                "Sunny": 0.2
            },
            "Foggy": {
                "Foggy": 0.6,
                "Rainy": 0.2,
                "Sunny": 0.2
            },
            "Harmattan": {
                "Harmattan": 0.7,
                "Sunny": 0.2,
                "Foggy": 0.1
            }
        }

        # Transition settings
        self.transition_timer = 0
        self.transition_interval = 60  # Seconds between potential transitions

        # CSV Logging setup
        self.csv_file = open('weather_stats.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'Day', 'Hour', 'Season', 'Weather',
            'Intensity', 'Temperature (°C)', 'Humidity (%)',
            'Wind Speed (m/s)', 'Precipitation', 'Visibility', 'Cloud Density'
        ])

    def initialize_state(self):
        """Initialize the weather state based on the current season."""
        if self.time_manager.season == "Rainy":
            return self.states["Rainy"]
        else:
            return self.states["Sunny"]

    def update(self, dt):
        """Update the weather state based on gradual changes and transition rules."""
        self.transition_timer += dt

        # Gradual intensity changes
        self.handle_intensity_changes()

        self.update_wind_direction()

        # Attempt weather transition at defined intervals
        if self.transition_timer >= self.transition_interval:
            self.transition_timer = 0
            self.attempt_transition()
        self.update_weather_effects(dt)

    def handle_intensity_changes(self):
        """Gradually increase or decrease the intensity of the current weather state."""
        # Example logic: If intensity is high, start decreasing; if low, start increasing
        if self.current_state.intensity >= 0.8:
            self.current_state.decrease_intensity(amount=0.005)
        elif self.current_state.intensity <= 0.3:
            self.current_state.increase_intensity(amount=0.005)
        else:
            # Randomly decide to increase or decrease
            if random.random() < 0.5:
                self.current_state.increase_intensity(amount=0.002)
            else:
                self.current_state.decrease_intensity(amount=0.002)
    
    def update_wind_direction(self):
        """Gradually change the wind direction."""
        change = random.uniform(-1, 1)  # Degrees per update
        self.current_state.wind_direction = (self.current_state.wind_direction + change) % 360


    def get_sky_color(self):
        """Calculate the current sky color based on the time of day."""
        if 5 <= self.time_manager.hour < 7:
            # Sunrise transition
            factor = (self.time_manager.hour + self.time_manager.minute / 60 - 5) / 2
            return self.interpolate_color(SKY_COLORS["sunrise"], SKY_COLORS["day"], factor)
        elif 7 <= self.time_manager.hour < 18:
            # Daytime
            return SKY_COLORS["day"]
        elif 18 <= self.time_manager.hour < 20:
            # Sunset transition
            factor = (self.time_manager.hour + self.time_manager.minute / 60 - 18) / 2
            return self.interpolate_color(SKY_COLORS["sunset"], SKY_COLORS["night"], factor)
        else:
            # Nighttime
            return SKY_COLORS["night"]

    def interpolate_color(self, color_start, color_end, factor):
        """Interpolate between two colors."""
        return tuple([
            int(color_start[i] + (color_end[i] - color_start[i]) * factor)
            for i in range(3)
        ])



    def attempt_transition(self):
        """Attempt to transition to a new weather state based on probabilities and intensity."""
        current = self.current_state.name
        intensity = self.current_state.intensity

        # Adjust transition probabilities based on intensity
        adjusted_probabilities = {}
        for state, prob in self.transition_probabilities.get(current, {}).items():
            if state == current:
                # Higher probability to stay in the current state if intensity is high
                adjusted_probabilities[state] = prob * intensity
            else:
                # Lower probability to transition if current intensity is high
                adjusted_probabilities[state] = prob * (1 - intensity)

        # Normalize the probabilities
        total = sum(adjusted_probabilities.values())
        if total == 0:
            # Prevent division by zero; default to staying in current state
            return
        for state in adjusted_probabilities:
            adjusted_probabilities[state] /= total

        # Weighted random choice
        new_state_name = self.weighted_choice(adjusted_probabilities)

        if new_state_name != current:
            self.current_state = self.states[new_state_name]
            print(f"Weather changed to {new_state_name} with intensity {self.current_state.intensity:.2f}")

    def weighted_choice(self, probabilities):
        """Select a new state based on adjusted probabilities."""
        total = sum(probabilities.values())
        rand = random.uniform(0, total)
        cumulative = 0
        for state, prob in probabilities.items():
            cumulative += prob
            if rand <= cumulative:
                return state
        return "Sunny"  # Fallback

    def get_current_weather(self):
        """Return the current weather state."""
        return self.current_state

    def log_weather_stats(self):
        """Log the current weather statistics to the CSV file."""
        self.csv_writer.writerow([
            self.time_manager.day_count + 1,
            self.time_manager.hour,
            self.time_manager.season,
            self.current_state.name,
            f"{self.current_state.intensity:.2f}",
            f"{self.current_state.temperature:.1f}",
            f"{self.current_state.humidity:.1f}",
            f"{self.current_state.wind_speed:.1f}",
            self.current_state.precipitation_type,
            f"{self.current_state.visibility:.2f}",
            f"{self.current_state.cloud_density:.2f}"
        ])

    def close_csv(self):
        """Close the CSV file."""
        self.csv_file.close()

    def render_weather_effects(self, surface):
        """Draw visual representations of current weather."""
        weather = self.current_state

        # Adjust background color based on time of day
        bg_color = self.get_sky_color()
        surface.fill(bg_color)

        # Overlay based on weather type and intensity
        if weather.name == "Sunny":
            # Increase brightness based on intensity
            brightness = min(255, int(135 + (weather.intensity * 120)))  # From 135 to 255
            surface.fill((brightness, 206, 235), special_flags=pygame.BLEND_ADD)
        elif weather.name == "Rainy":
            # Draw rain intensity
            for _ in range(int(100 * weather.intensity)):
                x = random.randint(0, WIDTH)
                y = random.randint(0, HEIGHT)
                length = random.randint(5, 15)
                pygame.draw.line(surface, RAIN_COLOR, (x, y), (x, y + length), 1)
        elif weather.name == "Foggy":
            # Draw fog overlay
            fog_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fog_alpha = int(80 * weather.intensity)
            fog_surface.fill((180, 180, 180, fog_alpha))
            surface.blit(fog_surface, (0, 0))
        elif weather.name == "Stormy":
            # Draw storm clouds and lightning
            for _ in range(int(50 * weather.intensity)):
                cloud_x = random.randint(0, WIDTH)
                cloud_y = random.randint(0, HEIGHT // 2)
                cloud_width = random.randint(50, 150)
                cloud_height = cloud_width // 2
                pygame.draw.ellipse(surface, STORM_COLOR, (cloud_x, cloud_y, cloud_width, cloud_height))
            # Occasionally draw lightning
            if random.random() < 0.05 * weather.intensity:
                start_x = random.randint(0, WIDTH)
                start_y = random.randint(0, HEIGHT // 2)
                end_x = start_x + random.randint(-20, 20)
                end_y = start_y + random.randint(50, 150)
                pygame.draw.line(surface, LIGHTNING_COLOR, (start_x, start_y), (end_x, end_y), 2)
        elif weather.name == "Harmattan":
            # Draw hazy overlay
            haze_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            haze_alpha = int(70 * weather.intensity)
            haze_surface.fill((210, 180, 140, haze_alpha))
            surface.blit(haze_surface, (0, 0))

# Simulation Orchestration
class Simulation:
    def __init__(self):
        self.time_manager = TimeManager()
        self.weather_system = WeatherSystem(self.time_manager)
        self.running = True

    def run(self):
        last_update_time = time.time()
        while self.running:
            dt = clock.tick(FPS) / 1000  # Delta time in seconds

            for event in pygame.event.get():
                if event.type == QUIT:
                    self.running = False

            # Update simulation components
            self.time_manager.update(dt)
            self.weather_system.update(dt)

            # Render everything
            self.render()

            # Log stats at the end of each hour
            if self.time_manager.minute == 0 and self.time_manager.hour != 0:
                self.weather_system.log_weather_stats()

        # After loop ends, close the CSV file
        self.weather_system.close_csv()
        pygame.quit()

    def render(self):
        """Render all simulation components."""
        # Render weather effects on the screen
        self.weather_system.render_weather_effects(screen)

        # Draw UI Stats
        self.draw_stats()

        pygame.display.flip()

    def draw_stats(self):
        """Display simulation statistics."""
        y = 10
        weather = self.weather_system.get_current_weather()
        stats = [
            f"Day: {self.time_manager.day_count + 1}",
            f"Time: {self.time_manager.hour:02}:{int(self.time_manager.minute):02}",
            f"Season: {self.time_manager.season}",
            f"Weather: {weather.name}",
            f"Intensity: {weather.intensity:.2f}",
            f"Temperature: {weather.temperature:.1f} °C",
            f"Humidity: {weather.humidity:.1f}%",
            f"Wind Speed: {weather.wind_speed:.1f} m/s",
            f"Wind Direction: {weather.wind_direction:.1f}°",
            f"Precipitation: {weather.precipitation_type}",
            f"Visibility: {weather.visibility:.2f}",
            f"Cloud Density: {weather.cloud_density:.2f}"
        ]

        for stat in stats:
            text = FONT.render(stat, True, (255, 255, 255))
            screen.blit(text, (10, y))
            y += 20

if __name__ == "__main__":
    simulation = Simulation()
    simulation.run()
