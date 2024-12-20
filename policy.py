import pygame
import random
import math
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Probabilistic Weather System - Niger Delta")

# Colors
COLORS = {
    'day': (135, 206, 235),
    'night': (25, 25, 112),
    'cloud': (200, 200, 200),
    'dark_cloud': (50, 50, 50),
    'rain': (0, 0, 255),
    'fog': (180, 180, 180, 100),
    'haze': (210, 180, 140, 70),
    'lightning': (255, 255, 200)
}

@dataclass
class WeatherParams:
    """Weather parameters configuration"""
    MAX_RAINDROPS: int = 1000
    MAX_CLOUDS: int = 20
    WIND_INFLUENCE: float = 0.5
    LIGHTNING_PROBABILITY: float = 0.02
    WEATHER_CHANGE_INTERVAL: float = 20.0

class TimeSystem:
    """Enhanced time management system"""
    def __init__(self, time_scale: float = 10.0):
        self.hour = 6
        self.minute = 0.0 # store as float for smooth transitions
        self.time_scale = time_scale
        self.sunrise = 6
        self.sunset = 18

    def update(self, dt: float) -> None:
        """Update time with delta time"""
        self.minute += dt * self.time_scale
        if self.minute >= 60:
            self.minute = 0
            self.hour += 1
            if self.hour >= 24:
                self.hour = 0

    
    # Add method to get integer minute for display
    def get_display_minute(self) -> int:
        """Get the current minute as an integer for display"""
        return int(self.minute)

    def get_display_hour(self) -> int:
        """Get the current hour as an integer for display"""
        return int(self.hour)

    def is_daytime(self) -> bool:
        """Check if current time is daytime"""
        return self.sunrise <= self.hour < self.sunset

    def get_time_factor(self) -> float:
        """Get interpolation factor for day/night transition"""
        if self.is_daytime():
            return 1.0
        return max(0.0, math.sin(math.pi * (self.hour - self.sunset) / (24 - (self.sunset - self.sunrise))))

class WeatherEffect:
    """Base class for weather effects"""
    def __init__(self):
        self.active = False
        self.intensity = 0.0

    def update(self, dt: float, weather_system: 'WeatherSystem') -> None:
        pass

    def draw(self, surface: pygame.Surface) -> None:
        pass

class RainEffect(WeatherEffect):
    """Enhanced rain effect with proper physics"""
    def __init__(self):
        super().__init__()
        self.raindrops: List[Dict] = []
        self.max_drops = WeatherParams.MAX_RAINDROPS

    def update(self, dt: float, weather_system: 'WeatherSystem') -> None:
        if not self.active:
            self.raindrops.clear()
            return

        # Add new raindrops based on intensity
        while len(self.raindrops) < self.max_drops * self.intensity:
            self.raindrops.append(self._create_raindrop())

        # Update existing raindrops
        wind_vector = pygame.Vector2(
            math.cos(math.radians(weather_system.wind_direction)) * weather_system.wind_speed,
            5.0 + weather_system.wind_speed * 0.5
        )

        for drop in self.raindrops:
            drop['position'] += wind_vector * dt * 50
            if drop['position'].y > HEIGHT:
                drop['position'] = pygame.Vector2(random.randint(0, WIDTH), -10)

    def draw(self, surface: pygame.Surface) -> None:
        if not self.active:
            return

        for drop in self.raindrops:
            end_pos = drop['position'] + pygame.Vector2(0, 10)
            pygame.draw.line(surface, COLORS['rain'], drop['position'], end_pos, 2)

    def _create_raindrop(self) -> Dict:
        return {
            'position': pygame.Vector2(random.randint(0, WIDTH), random.randint(-100, 0))
        }

class CloudSystem:
    """Enhanced cloud management system"""
    def __init__(self):
        self.clouds: List[Dict] = []
        self.max_clouds = WeatherParams.MAX_CLOUDS

    def update(self, dt: float, weather_system: 'WeatherSystem') -> None:
        # Maintain cloud count based on weather condition
        target_clouds = int(self.max_clouds * weather_system.cloud_coverage)
        
        while len(self.clouds) < target_clouds:
            self.clouds.append(self._create_cloud())
        
        while len(self.clouds) > target_clouds:
            self.clouds.pop()

        # Update cloud positions
        wind_vector = pygame.Vector2(
            math.cos(math.radians(weather_system.wind_direction)),
            math.sin(math.radians(weather_system.wind_direction))
        )
        
        for cloud in self.clouds:
            cloud['position'] += wind_vector * weather_system.wind_speed * dt * 30
            self._wrap_cloud_position(cloud)

    def draw(self, surface: pygame.Surface, is_stormy: bool) -> None:
        color = COLORS['dark_cloud'] if is_stormy else COLORS['cloud']
        for cloud in self.clouds:
            pos = cloud['position']
            size = cloud['size']
            pygame.draw.ellipse(surface, color, (pos.x, pos.y, size, size * 0.6))

    def _create_cloud(self) -> Dict:
        return {
            'position': pygame.Vector2(random.randint(0, WIDTH), random.randint(50, 150)),
            'size': random.uniform(80, 150),
            'speed': random.uniform(0.5, 2.0)
        }

    def _wrap_cloud_position(self, cloud: Dict) -> None:
        if cloud['position'].x > WIDTH:
            cloud['position'].x = -cloud['size']
        elif cloud['position'].x < -cloud['size']:
            cloud['position'].x = WIDTH
        if cloud['position'].y > HEIGHT:
            cloud['position'].y = random.randint(50, 150)
        elif cloud['position'].y < 0:
            cloud['position'].y = HEIGHT

class WeatherSystem:
    """Enhanced weather system with improved state management"""
    def __init__(self, season: str = "rainy"):
        self.time_system = TimeSystem()
        self.season = season
        self.weather_condition = "sunny"
        self.dynamic_condition = None
        
        # Weather parameters
        self.temperature = 27.0
        self.humidity = 85.0
        self.wind_speed = random.uniform(1, 5)
        self.wind_direction = random.uniform(0, 360)
        self.cloud_coverage = 0.0
        
        # Weather effects
        self.rain_effect = RainEffect()
        self.cloud_system = CloudSystem()
        
        # Transition parameters
        self.target_values = self._get_default_targets()
        self.transition_speed = 0.1
        self.last_weather_change = time.time()

    def update(self, dt: float) -> None:
        """Update weather system state"""
        self.time_system.update(dt)
        self._update_weather_state(dt)
        self._update_effects(dt)

    def draw(self, surface: pygame.Surface) -> None:
        """Draw weather effects"""
        # Set background color based on time of day
        bg_color = self._interpolate_colors(
            COLORS['day'],
            COLORS['night'],
            self.time_system.get_time_factor()
        )
        surface.fill(bg_color)

        # Draw weather effects
        self.cloud_system.draw(surface, self.weather_condition == "stormy")
        self.rain_effect.draw(surface)
        
        if self.dynamic_condition == "harmattan":
            self._draw_harmattan(surface)

    def _update_weather_state(self, dt: float) -> None:
        """Update weather conditions and handle transitions"""
        current_time = time.time()
        if current_time - self.last_weather_change >= WeatherParams.WEATHER_CHANGE_INTERVAL:
            self.last_weather_change = current_time
            self._change_weather_condition()

        # Smooth parameter transitions
        for param in ['temperature', 'humidity', 'wind_speed', 'cloud_coverage']:
            current = getattr(self, param)
            target = self.target_values[param]
            setattr(self, param, self._lerp(current, target, self.transition_speed * dt))

    def _update_effects(self, dt: float) -> None:
        """Update all weather effects"""
        self.cloud_system.update(dt, self)
        self.rain_effect.update(dt, self)

    def _change_weather_condition(self) -> None:
        """Change weather condition based on probabilities"""
        probabilities = self._get_season_probabilities()
        self.weather_condition = self._weighted_choice(probabilities)
        self.dynamic_condition = self._determine_dynamic_condition()
        self.target_values = self._get_condition_targets()
        
        # Update effect states
        self.rain_effect.active = self.weather_condition in ["rainy", "stormy"]
        self.rain_effect.intensity = 1.0 if self.weather_condition == "stormy" else 0.5

    def _get_season_probabilities(self) -> Dict[str, float]:
        """Get weather probabilities based on season"""
        if self.season == "rainy":
            return {
                "sunny": 0.2,
                "rainy": 0.4,
                "stormy": 0.2,
                "cloudy": 0.2
            }
        return {
            "sunny": 0.5,
            "cloudy": 0.3,
            "rainy": 0.1,
            "stormy": 0.1
        }

    def _get_condition_targets(self) -> Dict[str, float]:
        """Get target values for current weather condition"""
        base_targets = self._get_default_targets()
        
        if self.weather_condition == "rainy":
            base_targets.update({
                'temperature': random.uniform(23, 28),
                'humidity': random.uniform(80, 95),
                'cloud_coverage': random.uniform(0.7, 0.9)
            })
        elif self.weather_condition == "stormy":
            base_targets.update({
                'temperature': random.uniform(21, 25),
                'humidity': random.uniform(85, 98),
                'cloud_coverage': random.uniform(0.8, 1.0)
            })
        
        return base_targets

    def _get_default_targets(self) -> Dict[str, float]:
        """Get default target values"""
        return {
            'temperature': 27.0,
            'humidity': 85.0,
            'wind_speed': random.uniform(1, 5),
            'cloud_coverage': 0.2
        }

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between two values"""
        return a + (b - a) * t

    @staticmethod
    def _interpolate_colors(color1: Tuple, color2: Tuple, factor: float) -> Tuple:
        """Interpolate between two colors"""
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

    @staticmethod
    def _weighted_choice(probabilities: Dict[str, float]) -> str:
        """Select a weather condition based on probabilities"""
        choices = list(probabilities.keys())
        weights = list(probabilities.values())
        return random.choices(choices, weights=weights, k=1)[0]

def main():
    """Main program loop"""
    clock = pygame.time.Clock()
    weather_system = WeatherSystem()
    font = pygame.font.Font(None, 24)
    
    running = True
    last_time = time.time()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Calculate delta time
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Update and draw
        weather_system.update(dt)
        weather_system.draw(screen)
        
        # Draw stats
        stats = [
            f"Time: {weather_system.time_system.get_display_hour():02d}:{weather_system.time_system.get_display_minute():02d}",
            f"Weather: {weather_system.weather_condition}",
            f"Temperature: {weather_system.temperature:.1f}°C",
            f"Humidity: {weather_system.humidity:.1f}%",
            f"Wind Speed: {weather_system.wind_speed:.1f} m/s"
        ]

        for i, stat in enumerate(stats):
            text = font.render(stat, True, (255, 255, 255))
            screen.blit(text, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()