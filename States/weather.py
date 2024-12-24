# Weather Class

import pygame
import random
import math
import time
import csv
from pygame.locals import *
import json
from config import WIDTH, HEIGHT


# Frame Rate
FPS = 60


# Colors for Different Weather Conditions
DAY_COLOR = (135, 206, 235)
NIGHT_COLOR = (25, 25, 112)
CLOUD_COLOR = (220, 220, 220)
RAIN_COLOR = (0, 0, 255)
FOG_COLOR = (180, 180, 180, 80)
STORM_COLOR = (50, 50, 50)
LIGHTNING_COLOR = (255, 255, 255)
HARMATTAN_COLOR = (210, 180, 140, 100)

# Sky Colors for Day-Night Cycle
SKY_COLORS = {
    "sunrise": (255, 223, 186),
    "day": DAY_COLOR,
    "sunset": (255, 140, 0),
    "night": NIGHT_COLOR
}


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
                 precipitation_type, visibility_range, cloud_density, air_pressure_range=(980, 1050)):
        self.name = name
        self.temperature = random.uniform(*temperature_range)
        self.humidity = random.uniform(*humidity_range)
        self.wind_speed = random.uniform(*wind_speed_range)
        self.wind_direction = random.uniform(0, 360)  # Degrees
        self.precipitation_type = precipitation_type  # e.g., 'None', 'Rain', 'Thunderstorm'
        self.visibility_range = visibility_range  # (max visibility, min visibility)
        self.cloud_density = cloud_density  # Affects visibility
        self.air_pressure = random.uniform(*air_pressure_range)  # hPa
        self.intensity = 0.5  # Initial intensity (0 to 1)
        self.visibility = self.calculate_visibility()

    def calculate_visibility(self):
        """Calculate visibility based on intensity and visibility range."""
        max_vis, min_vis = self.visibility_range
        # Base visibility reduction based on intensity
        visibility = max_vis - self.intensity * (max_vis - min_vis)

        # Additional reduction based on cloud density
        visibility -= self.cloud_density * 0.1  # Adjustable scaling factor

        # Additional reduction based on humidity (for Foggy weather)
        if self.name == "Foggy":
            visibility -= (self.humidity / 100) * 0.1  # Adjust scaling as needed

        # Seasonal adjustment for Harmattan
        if self.name == "Harmattan":
            visibility -= 0.2  # Reduction due to dust

        # Ensure visibility is within the defined range
        visibility = max(min_vis, visibility)

        # Debugging Statement
        # print(f"[DEBUG] {self.name} Weather: Intensity={self.intensity:.2f}, Visibility={visibility:.2f}")

        return visibility

    def update_visibility(self):
        """Update visibility based on intensity changes."""
        self.visibility = self.calculate_visibility()

    def increase_intensity(self, amount=0.01):
        """Gradually increase the intensity of the weather state."""
        self.intensity = min(1.0, self.intensity + amount)
        self.update_visibility()  # Ensure visibility is updated
        # print(f"[DEBUG] {self.name} Weather: Intensity increased to {self.intensity:.2f}")

    def decrease_intensity(self, amount=0.01):
        """Gradually decrease the intensity of the weather state."""
        self.intensity = max(0.0, self.intensity - amount)
        self.update_visibility()  # Ensure visibility is updated
        # print(f"[DEBUG] {self.name} Weather: Intensity decreased to {self.intensity:.2f}")

# Cloud Class for Visual Representation
class Cloud:
    """Represents a single cloud in the simulation."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize or reset the cloud's position and properties."""
        self.x = random.randint(-WIDTH, WIDTH)
        self.y = random.randint(50, HEIGHT // 2)
        self.speed = random.uniform(0.2, 0.5)  # Slower movement
        self.size = random.randint(100, 200)  # Width of the cloud

    def update(self):
        """Update the cloud's position."""
        self.x += self.speed
        if self.x - self.size > WIDTH:
            self.reset()

    def draw(self, surface):
        """Draw the cloud on the given surface."""
        # Draw multiple overlapping ellipses to form a fluffy cloud
        ellipse_width = self.size
        ellipse_height = self.size // 2
        offsets = [(-ellipse_width * 0.3, 0), (0, -ellipse_height * 0.5), (ellipse_width * 0.3, 0)]
        for dx, dy in offsets:
            pygame.draw.ellipse(surface, CLOUD_COLOR, (self.x + dx, self.y + dy, ellipse_width, ellipse_height))

# Lightning Strike Class
class LightningStrike:
    """Handles lightning strikes in stormy weather."""
    def __init__(self):
        self.active = False
        self.duration = 0
        self.start_time = 0
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

    def trigger(self):
        """Initiate a lightning strike."""
        if not self.active:
            self.active = True
            self.duration = random.uniform(0.1, 0.3)
            self.start_time = time.time()
            self.start_x = random.randint(0, WIDTH)
            self.start_y = random.randint(0, HEIGHT // 2)
            self.end_x = self.start_x + random.randint(-20, 20)
            self.end_y = self.start_y + random.randint(50, 150)
            print(f"[DEBUG] Lightning Strike initiated at ({self.start_x}, {self.start_y}) to ({self.end_x}, {self.end_y})")

    def update(self):
        """Update the lightning strike status."""
        if self.active and (time.time() - self.start_time) > self.duration:
            self.active = False
            print(f"[DEBUG] Lightning Strike ended.")

    def draw(self, surface):
        """Draw the lightning strike."""
        if self.active:
            pygame.draw.line(surface, LIGHTNING_COLOR, (self.start_x, self.start_y), (self.end_x, self.end_y), 2)

# Raindrop Class for Visual Representation
class Raindrop:
    """Represents a single raindrop in the simulation."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize or reset the raindrop's position and properties."""
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(-HEIGHT, 0)
        self.length = random.randint(10, 20)
        self.speed = random.uniform(4, 7)
        self.vx = 0
        self.vy = self.speed
        self.color = (173, 216, 230, 100)  # Light blue with alpha 100

    def update(self, wind_speed, wind_direction):
        """Update the raindrop's position based on wind."""
        angle_rad = math.radians(wind_direction)
        wind_dx = math.cos(angle_rad) * wind_speed
        wind_dy = math.sin(angle_rad) * wind_speed

        # Apply wind to raindrop velocity
        self.vx += wind_dx * 0.1
        self.vy += wind_dy * 0.1

        # Update position
        self.x += self.vx * 0.1
        self.y += self.vy * 0.1

        # Reset if out of bounds
        if self.y > HEIGHT:
            self.reset()

    def draw(self, surface):
        """Draw the raindrop on the given surface."""
        pygame.draw.line(surface, RAIN_COLOR, (self.x, self.y), (self.x, self.y + self.length), 1)

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
                visibility_range=(1.0, 0.8), # minimal impact on visibility
                cloud_density=0.2
            ),
            "Rainy": WeatherState(
                "Rainy",
                temperature_range=(20, 30),
                humidity_range=(70, 100),
                wind_speed_range=(10, 25),
                precipitation_type="Rain",
                visibility_range=(0.6, 0.3), # reduced visibility with higher intensity
                cloud_density=0.7
            ),
            "Foggy": WeatherState(
                "Foggy",
                temperature_range=(15, 25),
                humidity_range=(80, 100),
                wind_speed_range=(0, 5),
                precipitation_type="None",
                visibility_range = (0.4, 0.1), # very low visibility
                cloud_density=0.5
            ),
            "Stormy": WeatherState(
                "Stormy",
                temperature_range=(18, 28),
                humidity_range=(75, 100),
                wind_speed_range=(20, 40),
                precipitation_type="Rain",
                visibility_range=(0.5, 0.2), # reduced visibility with higher intensity
                cloud_density=0.8
            ),
            "Harmattan": WeatherState(
                "Harmattan",
                temperature_range=(15, 25),
                humidity_range=(20, 40),
                wind_speed_range=(10, 20),
                precipitation_type="None",
                visibility_range=(0.7, 0.3), # reduced visibility due to dust
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

        # Initialize weather effects
        self.raindrops = [Raindrop() for _ in range(200)]
        self.lightning = LightningStrike()

        # Initialize clouds
        self.clouds = [Cloud() for _ in range(10)]  # Adjust number as needed

        # CSV Logging setup
        try:
            self.csv_file = open('weather_stats.csv', mode='w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Day', 'Hour', 'Season', 'Weather',
                'Intensity', 'Temperature (°C)', 'Humidity (%)',
                'Wind Speed (m/s)', 'Wind Direction (°)', 'Precipitation',
                'Visibility', 'Cloud Density', 'Air Pressure (hPa)'
            ])
        except IOError as e:
            print(f"Failed to open CSV file: {e}")
            self.csv_writer = None

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

        # Update wind direction gradually
        self.update_wind_direction()

        # Attempt weather transition at defined intervals
        if self.transition_timer >= self.transition_interval:
            self.transition_timer = 0
            self.attempt_transition()

        # Update weather effects
        self.update_weather_effects(dt)

        # Log current weather statistics
        self.log_weather_stats()

    def handle_intensity_changes(self):
        """Gradually increase or decrease the intensity of the current weather state."""
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
    
    def log_weather_change(self, new_state):
        """Log the weather state change."""
        self.csv_writer.writerow([
            self.time_manager.day_count + 1,
            self.time_manager.hour,
            self.time_manager.season,
            new_state,
            f"{self.current_state.intensity:.2f}",
            f"{self.current_state.temperature:.1f}",
            f"{self.current_state.humidity:.1f}",
            f"{self.current_state.wind_speed:.1f}",
            f"{self.current_state.wind_direction:.1f}",
            self.current_state.precipitation_type,
            f"{self.current_state.visibility:.2f}",
            f"{self.current_state.cloud_density:.2f}",
            f"{self.current_state.air_pressure:.1f}"
        ])
        self.csv_file.flush()

    def log_weather_stats(self):
        """Log the current weather statistics to the CSV file."""
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    self.time_manager.day_count + 1,
                    self.time_manager.hour,
                    self.time_manager.season,
                    self.current_state.name,
                    f"{self.current_state.intensity:.2f}",
                    f"{self.current_state.temperature:.1f}",
                    f"{self.current_state.humidity:.1f}",
                    f"{self.current_state.wind_speed:.1f}",
                    f"{self.current_state.wind_direction:.1f}",
                    self.current_state.precipitation_type,
                    f"{self.current_state.visibility:.2f}",
                    f"{self.current_state.cloud_density:.2f}",
                    f"{self.current_state.air_pressure:.1f}"
                ])
            except IOError as e:
                print(f"Failed to write to CSV file: {e}")

    def close_csv(self):
        """Close the CSV file."""
        if self.csv_file:
            try:
                self.csv_file.close()
            except IOError as e:
                print(f"Failed to close CSV file: {e}")

    def update_weather_effects(self, dt):
        """Update dynamic weather effects like rain and lightning."""
        weather = self.current_state

        # Update raindrops based on wind speed and direction
        if weather.precipitation_type == "Rain":
            for drop in self.raindrops:
                drop.update(wind_speed=weather.wind_speed, wind_direction=weather.wind_direction)
        else:
            # Reset raindrops if not raining
            for drop in self.raindrops:
                drop.reset()

        # Handle lightning in Stormy weather
        if weather.name == "Stormy" and weather.intensity > 0.5:
            if random.random() < 0.02 * weather.intensity:
                self.lightning.trigger()
        else:
            self.lightning.active = False

        self.lightning.update()

        # Update clouds
        for cloud in self.clouds:
            cloud.update()

    def render_weather_effects(self, surface):
        """Draw visual representations of current weather."""
        weather = self.current_state

        # semi-transparent surface for weather effects
        weather_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

        # Fill with transparent color
        weather_surface.fill((0, 0, 0, 0))

        # Get dynamic sky color
        bg_color = self.get_sky_color()
        surface.fill(bg_color)

        # Overlay based on weather type and intensity
        if weather.name == "Sunny":
            # Calculate brightness factor based on intensity
            brightness_alpha = int(weather.intensity * 100)  # Alpha ranges from 0 (transparent) to 100 (semi-transparent)
            brightness_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            brightness_surface.fill((255, 255, 200, brightness_alpha))  # Slightly yellowish white with variable alpha
            weather_surface.blit(brightness_surface, (0, 0))

        elif weather.name == "Rainy":
            # Draw raindrops
            for drop in self.raindrops:
                drop.draw(weather_surface)

        elif weather.name == "Foggy":
            # Draw fog overlay
            fog_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fog_alpha = int(80 * weather.intensity)  # Adjust scaling as needed
            fog_surface.fill((180, 180, 180, fog_alpha))
            weather_surface.blit(fog_surface, (0, 0))

        elif weather.name == "Stormy":
            # Draw storm clouds
            for cloud in self.clouds:
                cloud.draw(weather_surface)
            # Draw lightning
            self.lightning.draw(weather_surface)
        elif weather.name == "Harmattan":
            # Draw hazy overlay
            haze_alpha = int(70 * weather.intensity)
            haze_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            haze_surface.fill((210, 180, 140, haze_alpha))
            weather_surface.blit(haze_surface, (0, 0))

        # overlay based on visibility to simulate reduced visibility
        if weather.visibility < 1.0:
            visibility_factor = 1.0 - weather.visibility  # Higher factor means lower visibility
            visibility_alpha = int(visibility_factor * 150)  # Adjust scaling as needed
            visibility_overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            visibility_overlay.fill((0, 0, 0, visibility_alpha))  # Dark overlay
            weather_surface.blit(visibility_overlay, (0, 0))

        # Draw clouds moving across the screen
        if weather.name in ["Sunny", "Rainy", "Foggy", "Stormy", "Harmattan"]:
            for cloud in self.clouds:
                cloud.draw(weather_surface)

        # Blit the semi-transparent weather_surface onto the main surface
        surface.blit(weather_surface, (0, 0))


