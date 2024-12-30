# oil_spillage.py

import numpy as np
import random
import math
import logging

from environment import Environment, WATER_LEVEL
from config import WIDTH, HEIGHT, CELL_SIZE

class OilSpillage:
    def __init__(self, environment, volume_range=(50, 200), oil_type='Light Crude'):
        """
        Initialize an OilSpillage manager that can create multiple spills per day.

        Parameters:
            environment: The Environment instance (with .grid, .grid_width, .grid_height).
            volume_range: A (min, max) tuple for random spill volume generation.
            oil_type: Type of oil. Affects spread/evap rates.
        """
        self.environment = environment
        self.oil_type = oil_type
        self.min_volume, self.max_volume = volume_range
        self.logger = logging.getLogger('OilSpillage')

        # In this approach, we don't store just one 'self.grid';
        # we might have multiple discrete spills. Let's store them in a list of grids
        self.active_spills = []  # Each element is a separate np array, same shape as environment.grid

        # Constants for oil behavior
        self.spread_fraction = 0.10  # e.g., 10% spreads out each "hourly" step
        self.max_oil_per_cell = 10
        self.spill_logs = []

        # Precompute rates from oil type
        self.spread_rate = self.calculate_spread_rate(oil_type)
        self.evaporation_rate = self.calculate_evaporation_rate(oil_type)

        # We track time to know when an hour/day passes
        self.accumulated_time = 0.0
        self.day_index = 0  # To track when we roll over to a new day and create new spills

    def calculate_spread_rate(self, oil_type):
        # You can still keep or remove if you want more advanced logic
        if oil_type == 'Light Crude':
            return 0.05
        elif oil_type == 'Heavy Crude':
            return 0.02
        else:
            return 0.04

    def calculate_evaporation_rate(self, oil_type):
        if oil_type == 'Light Crude':
            return 0.02
        elif oil_type == 'Heavy Crude':
            return 0.005
        else:
            return 0.01

    def create_daily_spills(self, num_spills=4):
        """
        Create 'num_spills' new random oil spills on water cells.
        """
        for _ in range(num_spills):
            spill_volume = random.uniform(self.min_volume, self.max_volume)
            # find a random water cell
            while True:
                x = random.randint(0, self.environment.grid_width - 1)
                y = random.randint(0, self.environment.grid_height - 1)
                if self.environment.grid[x, y] <= WATER_LEVEL:
                    break
            # Make a new grid for this spill
            new_spill_grid = np.zeros_like(self.environment.grid)
            new_spill_grid[x, y] = spill_volume

            msg = (f"New oil spill created on Day {self.day_index} at cell ({x},{y}) "
                   f"with volume={spill_volume:.2f}")
            self.logger.info(msg)
            print(msg)

            self.active_spills.append(new_spill_grid)

    def update(self, dt, weather_system, day_count):
        """
        Called every frame/timestep with dt (in seconds) and the current day_count from the TimeManager.
        If day_count changes, we create new spills (ensuring at least 4 per day).
        """
        # Check if we rolled over to a new day
        if day_count != self.day_index:
            self.day_index = day_count
            # Create new spills for the new day
            self.create_daily_spills(num_spills=4)

        self.accumulated_time += dt

        # We'll do the "big spread step" only once per hour
        # i.e. if accumulated_time crosses a multiple of 3600
        while self.accumulated_time >= 3600.0:
            self.accumulated_time -= 3600.0
            # Spread + advect each spill
            for i in range(len(self.active_spills)):
                self.active_spills[i] = self.spread_oil(self.active_spills[i], weather_system)
                self.active_spills[i] = self.advect(self.active_spills[i], weather_system)
                # clamp after advection
                self.active_spills[i] = np.minimum(np.maximum(self.active_spills[i], 0),
                                                   self.max_oil_per_cell)

        # Evaporation occurs every frame (small increments each dt)
        for i in range(len(self.active_spills)):
            self.active_spills[i] = self.weather_oil(self.active_spills[i], dt, weather_system)

    def spread_oil(self, spill_grid, weather_system):
        """
        Spread some fraction of oil to neighbors in the same grid, one cell at a time.
        """
        new_grid = np.copy(spill_grid)
        for x in range(self.environment.grid_width):
            for y in range(self.environment.grid_height):
                amt = spill_grid[x, y]
                if amt > 0:
                    neighbors = self.get_neighbors(x, y)
                    portion = amt * self.spread_fraction
                    if neighbors and portion > 0:
                        share = portion / len(neighbors)
                        for (nx, ny) in neighbors:
                            # Only spread onto water
                            if self.environment.grid[nx, ny] <= WATER_LEVEL:
                                new_grid[nx, ny] += share
                            else:
                                # If we find land, log it and skip
                                if (nx, ny) not in self.spill_logs:
                                    self.spill_logs.append((nx, ny))
                                    logmsg = (f"Oil tried to spread onto land at ({nx},{ny}). "
                                              f"Stopping spread there.")
                                    self.logger.info(logmsg)
                                    print(logmsg)
                        new_grid[x, y] -= portion
        return new_grid

    def advect(self, spill_grid, weather_system):
        """
        Move the entire oil distribution ONE cell in the nearest cardinal direction of wind.
        """
        wind_dir = weather_system.current_state.wind_direction
        (dx, dy) = self.cardinal_direction(wind_dir)

        # Move entire spill by 1 cell in that direction
        new_grid = np.zeros_like(spill_grid)
        for x in range(self.environment.grid_width):
            for y in range(self.environment.grid_height):
                amt = spill_grid[x, y]
                if amt > 0:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.environment.grid_width and 0 <= ny < self.environment.grid_height:
                        new_grid[nx, ny] += amt
                    else:
                        # oil leaves the map
                        pass
        return new_grid

    def cardinal_direction(self, wind_degs):
        """
        Convert wind_degs (0-359) to a nearest cardinal direction:
          0   -> East (1,0)
          90  -> North (0,-1)
          180 -> West (-1,0)
          270 -> South (0,1)
        We also handle angles in between by rounding to nearest cardinal.
        """
        wind_degs %= 360
        # We'll do a simple approach: each quadrant is 90 degrees
        if wind_degs < 45 or wind_degs >= 315:
            return (1, 0)   # East
        elif wind_degs < 135:
            return (0, -1)  # North (in many grids, y-1 is up)
        elif wind_degs < 225:
            return (-1, 0)  # West
        else:
            return (0, 1)   # South

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.environment.grid_width and 0 <= ny < self.environment.grid_height:
                    neighbors.append((nx, ny))
        return neighbors

    def weather_oil(self, spill_grid, dt, weather_system):
        """
        Evaporation each frame. The total volume goes down gradually.
        """
        temperature = weather_system.current_state.temperature
        evaporation_factor = self.evaporation_rate * (1 + (temperature - 15) / 100)
        # apply small evaporation each dt
        new_grid = spill_grid - spill_grid * evaporation_factor * dt
        new_grid = np.maximum(new_grid, 0)
        return new_grid

    def combined_oil_grid(self):
        """
        If you want to render all spills as one, sum them up.
        """
        if not self.active_spills:
            return np.zeros_like(self.environment.grid)
        total_grid = np.zeros_like(self.environment.grid)
        for g in self.active_spills:
            total_grid += g
        return total_grid
