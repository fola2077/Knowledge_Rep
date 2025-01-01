# oil_spillage.py

import numpy as np
import random
import math
import logging
from scipy import signal

from environment import Environment, WATER_LEVEL
from config import CELL_SIZE

class OilSpill:
    """
    Represents an individual oil spill.
    """
    def __init__(self, environment, initial_cells, initial_concentration=1.0):
        self.environment = environment
        self.grid_width = environment.grid_width
        self.grid_height = environment.grid_height
        self.concentration = np.zeros((self.grid_width, self.grid_height), dtype=float)
        for (gx, gy) in initial_cells:
            self.concentration[gx, gy] = initial_concentration

    def spread(self, wind_dx, wind_dy):
        """
        Spread the oil spill using convolution to simulate diffusion.
        """
        kernel = self.get_spread_kernel(wind_dx, wind_dy)
        new_concentration = signal.convolve2d(
            self.concentration,
            kernel,
            mode='same',
            boundary='fill',
            fillvalue=0
        )

        # Apply environmental constraints
        water_mask = self.environment.grid <= WATER_LEVEL
        self.concentration = np.where(water_mask, new_concentration, 0)

        # Normalize concentrations
        self.concentration = np.clip(self.concentration, 0, 1)

    def get_spread_kernel(self, wind_dx, wind_dy):
        """
        Get a convolution kernel adjusted for wind influence.
        """
        base_kernel = np.array([
            [0.05, 0.1, 0.05],
            [0.1,  0.4, 0.1],
            [0.05, 0.1, 0.05]
        ])

        # Adjust the kernel based on wind direction
        # Shift the kernel in the direction of the wind
        shift_x = int(round(wind_dx))
        shift_y = int(round(wind_dy))
        kernel = np.roll(base_kernel, shift=shift_x, axis=1)
        kernel = np.roll(kernel, shift=shift_y, axis=0)

        # Normalize the kernel to ensure total sum is 1
        kernel /= kernel.sum()

        return kernel

    def evaporate_and_dissolve(self, temperature):
        """
        Reduce the oil concentration based on evaporation and dissolution.
        """
        # Simple evaporation model: evaporation rate increases with temperature
        evaporation_rate = 0.005 + 0.0005 * (temperature - 15)  # Adjust as needed
        evaporation_rate = np.clip(evaporation_rate, 0, 0.05)
        self.concentration *= (1 - evaporation_rate)

    def is_active(self):
        """
        Check if the spill still has significant oil concentration.
        """
        return np.any(self.concentration > 0.01)

class OilSpillage:
    """
    Manages all oil spills in the environment.
    """
    def __init__(self, environment, time_manager):
        self.environment = environment
        self.time_manager = time_manager
        self.logger = logging.getLogger('OilSpillage')
        self.spills = []
        self.next_spawn_day = 1  # First spawn at Day 1
        self.last_expansion_time = None  # Track last expansion time

    def update(self, weather_system):
        """
        Update oil spills based on time and environmental factors.
        """
        # Check if we should spawn new spills
        if self.time_manager.day_count >= self.next_spawn_day:
            self.spawn_new_spills()
            # Choose next spawn day: current day + random(1..3)
            self.next_spawn_day = self.time_manager.day_count + random.randint(1, 3)

        # Check if it's time to expand spills
        current_total_minutes = (
            self.time_manager.day_count * 24 * 60 +
            self.time_manager.hour * 60 +
            self.time_manager.minute
        )
        if self.last_expansion_time is None:
            self.last_expansion_time = current_total_minutes

        # Every hour, update spills
        if current_total_minutes - self.last_expansion_time >= 180:
            self.last_expansion_time = current_total_minutes
            wind_dir = weather_system.current_state.wind_direction
            wind_speed = weather_system.current_state.wind_speed
            temperature = weather_system.current_state.temperature
            wind_dx, wind_dy = self.get_wind_vector(wind_dir, wind_speed)
            self.update_spills(wind_dx, wind_dy, temperature)

    def spawn_new_spills(self):
        """
        Spawn new spills at random locations on water.
        """
        num_spills = random.randint(1, 4)
        for _ in range(num_spills):
            block_cells = self.find_random_4cell_block()
            if not block_cells:
                self.logger.warning("Failed to find a suitable location for new oil spill.")
                continue

            spill = OilSpill(self.environment, block_cells)
            self.spills.append(spill)
            self.logger.info(f"Spawned oil spill at {block_cells}, day={self.time_manager.day_count}")
            print(f"Spawned oil spill at {block_cells}, day={self.time_manager.day_count}")

    def find_random_4cell_block(self):
        """
        Find a 2x2 block on water to place a new spill.
        """
        for _ in range(100):
            gx = random.randint(0, self.environment.grid_width - 2)
            gy = random.randint(0, self.environment.grid_height - 2)
            coords = [(gx, gy), (gx+1, gy), (gx, gy+1), (gx+1, gy+1)]
            if all(self.environment.grid[cx, cy] <= WATER_LEVEL for (cx, cy) in coords):
                return coords
        return None

    def update_spills(self, wind_dx, wind_dy, temperature):
        """
        Update each spill's spread and evaporation.
        """
        active_spills = []
        for spill in self.spills:
            spill.spread(wind_dx, wind_dy)
            spill.evaporate_and_dissolve(temperature)
            if spill.is_active():
                active_spills.append(spill)
            else:
                self.logger.info("An oil spill has dissipated completely.")
        self.spills = active_spills  # Remove inactive spills

    def get_wind_vector(self, wind_degs, wind_speed):
        """
        Calculate wind vector components.
        """
        scaling_factor = 0.1  # Adjust as needed
        wind_radians = math.radians(wind_degs)
        dx = math.cos(wind_radians) * wind_speed * scaling_factor
        dy = math.sin(wind_radians) * wind_speed * scaling_factor
        return dx, dy

    def combined_oil_concentration(self):
        """
        Combine oil concentrations from all active spills.
        """
        w, h = self.environment.grid_width, self.environment.grid_height
        total_concentration = np.zeros((w, h), dtype=float)
        for spill in self.spills:
            total_concentration += spill.concentration
        # Ensure concentration values are between 0 and 1
        total_concentration = np.clip(total_concentration, 0, 1)
        return total_concentration

    def get_cell_color(self, concentration):
        """
        Map oil concentration to a color for rendering.
        The color starts as dark purple at high concentration and
        becomes lighter as the concentration decreases.
        """
        # Define the color for maximum concentration (dark purple)
        max_concentration_color = np.array([39, 4, 51])  # RGB values for dark purple

        # Define the color for minimum concentration (lightest acceptable color)
        min_concentration_color = np.array([169, 17, 222])  # Lighter purple or near transparent

        # Normalize concentration to range [0,1]
        norm_concentration = np.clip(concentration, 0, 1)

        # Invert concentration to have dark color at high concentrations
        inverted_concentration = norm_concentration

        # Interpolate between min and max colors
        color = inverted_concentration * max_concentration_color + (1 - inverted_concentration) * min_concentration_color

        # Ensure values are within valid RGB range
        color = np.clip(color, 0, 255).astype(int)

        return tuple(color)