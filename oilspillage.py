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
        Initialize an OilSpillage manager that can create multiple spills per day,
        while modeling more realistic spreading, drifting, evaporation, and partial dispersion.
        """
        self.environment = environment
        self.oil_type = oil_type
        self.min_volume, self.max_volume = volume_range
        self.logger = logging.getLogger('OilSpillage')

        # We'll hold multiple spills, each is an np array of shape environment.grid
        self.active_spills = []

        # Base parameters (you can tune these)
        self.spread_fraction = 0.10      # fraction of oil that spreads to neighbors each hour
        self.max_oil_per_cell = 10.0     # clamp maximum concentration in each cell
        self.spill_logs = []

        # Extended oil properties
        self.viscosity = self._assign_viscosity(oil_type)  # (cP) just a rough guess
        self.wave_factor_base = 0.05     # base wave effect on fragmentation and dispersion
        self.fragmentation_prob = 0.1    # chance that a slick fragments instead of distributing evenly
        self.dispersion_rate = 0.05      # fraction that can be dispersed into water column if wave_factor is high

        # Precompute or store rates from oil type
        self.spread_rate = self.calculate_spread_rate(oil_type)       # still used if you want
        self.evaporation_rate = self.calculate_evaporation_rate(oil_type)

        # Timers / day logic
        self.accumulated_time = 0.0
        self.day_index = 0

    def _assign_viscosity(self, oil_type):
        """
        Assign a rough viscosity (in cP) based on oil type.
        Lower = spreads more easily; higher = spreads less.
        """
        if oil_type == 'Light Crude':
            return 100.0    # very rough
        elif oil_type == 'Heavy Crude':
            return 10000.0
        else:
            return 500.0    # default “medium” viscosity

    def calculate_spread_rate(self, oil_type):
        # If you want a generic rate, keep it. If not, remove or adapt.
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
            # Make a new grid
            new_spill_grid = np.zeros_like(self.environment.grid)
            new_spill_grid[x, y] = spill_volume

            msg = (f"New oil spill created on Day {self.day_index} at cell ({x},{y}) "
                   f"with volume={spill_volume:.2f}")
            self.logger.info(msg)
            print(msg)

            self.active_spills.append(new_spill_grid)

    def update(self, dt, weather_system, day_count):
        """
        Called each frame. If the day changes, create new spills.
        Accumulate dt, and once we pass 3600s, do an hourly big step: spread, advect, clamp.
        Then also apply evaporation (small increments) each frame.
        """
        if day_count != self.day_index:
            self.day_index = day_count
            self.create_daily_spills(num_spills=4)

        self.accumulated_time += dt

        # Big step each hour
        while self.accumulated_time >= 3600.0:
            self.accumulated_time -= 3600.0
            for i in range(len(self.active_spills)):
                # spread
                self.active_spills[i] = self.spread_oil(self.active_spills[i], weather_system)
                # advect
                self.active_spills[i] = self.advect(self.active_spills[i], weather_system)
                # clamp
                self.active_spills[i] = np.minimum(np.maximum(self.active_spills[i], 0),
                                                   self.max_oil_per_cell)

        # Evaporation each frame
        for i in range(len(self.active_spills)):
            self.active_spills[i] = self.weather_oil(self.active_spills[i], dt, weather_system)

    def spread_oil(self, spill_grid, weather_system):
        """
        Enhanced spreading that accounts for viscosity, wave action, fragmentation, and partial dispersion.
        """
        new_grid = np.copy(spill_grid)
        wave_factor = self.compute_wave_factor(weather_system)  # depends on wind speed, etc.

        for x in range(self.environment.grid_width):
            for y in range(self.environment.grid_height):
                amt = spill_grid[x, y]
                if amt > 0:
                    # basic thickness estimate: volume (liters) / area. We'll treat 1 cell = 1m^2 for simplicity
                    # or you can use (CELL_SIZE**2) if each cell is 10x10 => 100m^2, etc.
                    # This is purely conceptual
                    thickness = amt / 1.0

                    # viscosity factor => thicker or more viscous => spreads less
                    # simple approach: fraction = spread_fraction * wave_factor * (base / (1 + log(viscosity)))
                    visc_factor = 1.0 / (1.0 + math.log10(self.viscosity + 1))
                    local_spread_frac = self.spread_fraction * wave_factor * visc_factor

                    portion = amt * local_spread_frac

                    if portion > 0:
                        neighbors = self.get_neighbors(x, y)
                        if neighbors:
                            # Check fragmentation
                            if random.random() < self.fragmentation_prob * wave_factor:
                                # Fragment: choose a random subset of neighbors
                                subset_count = max(1, len(neighbors)//2)
                                chosen_neighbors = random.sample(neighbors, subset_count)
                                share = portion / subset_count
                                for (nx, ny) in chosen_neighbors:
                                    if self.environment.grid[nx, ny] <= WATER_LEVEL:
                                        new_grid[nx, ny] += share
                                    else:
                                        self.log_land_spread(nx, ny)
                                new_grid[x, y] -= portion
                            else:
                                # Even distribution
                                share = portion / len(neighbors)
                                for (nx, ny) in neighbors:
                                    if self.environment.grid[nx, ny] <= WATER_LEVEL:
                                        new_grid[nx, ny] += share
                                    else:
                                        self.log_land_spread(nx, ny)
                                new_grid[x, y] -= portion

                    # partial dispersion if wave_factor is high
                    dispersion_amt = amt * self.dispersion_rate * wave_factor
                    new_grid[x, y] -= dispersion_amt
                    # dispersed oil => not tracked on surface. If you want to track in water column,
                    # store it in a separate array. For now, we treat it as "lost from surface."

        return new_grid

    def compute_wave_factor(self, weather_system):
        """
        A small function that returns a wave factor 0..1 based on wind speed or rapids.
        If you have rapids, you might add a 'self.environment.rapids_level'.
        """
        wind_speed = weather_system.current_state.wind_speed  # e.g. 0..20 m/s
        # scale it so 20 m/s => wave_factor ~ 1.0
        factor = min(1.0, wind_speed / 20.0)
        return self.wave_factor_base + factor  # e.g. base 0.05 + up to 1 => up to 1.05

    def log_land_spread(self, nx, ny):
        """Helper to log once if oil tries to spread onto land in cell (nx, ny)."""
        if (nx, ny) not in self.spill_logs:
            self.spill_logs.append((nx, ny))
            logmsg = (f"Oil tried to spread onto land at ({nx},{ny}). Stopping spread there.")
            self.logger.info(logmsg)
            print(logmsg)

    def advect(self, spill_grid, weather_system):
        """
        Move oil 1 cell based on cardinal direction. Could combine wind + currents here.
        """
        wind_dir = weather_system.current_state.wind_direction
        (dx, dy) = self.cardinal_direction(wind_dir)

        new_grid = np.zeros_like(spill_grid)
        for x in range(self.environment.grid_width):
            for y in range(self.environment.grid_height):
                amt = spill_grid[x, y]
                if amt > 0:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < self.environment.grid_width and 0 <= ny < self.environment.grid_height:
                        new_grid[nx, ny] += amt
                    # else: it goes off-grid
        return new_grid

    def cardinal_direction(self, wind_degs):
        wind_degs %= 360
        if wind_degs < 45 or wind_degs >= 315:
            return (1, 0)   # East
        elif wind_degs < 135:
            return (0, -1) # North (assuming y-1 is up in your coordinate system)
        elif wind_degs < 225:
            return (-1, 0) # West
        else:
            return (0, 1)  # South

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
        Evaporation each frame. For a bit more realism:
        - Possibly do a faster initial evaporation, then slow (two-phase).
        - We'll keep it simple, but you can see how to tweak it.
        """
        temperature = weather_system.current_state.temperature
        evaporation_factor = self.evaporation_rate * (1 + (temperature - 15) / 100)

        # For a bigger early-hour evaporation, you could do:
        # if your environment time < a certain threshold, multiply evaporation_factor by 2

        new_grid = spill_grid - (spill_grid * evaporation_factor * dt)
        new_grid = np.maximum(new_grid, 0)
        return new_grid

    def combined_oil_grid(self):
        """
        Summation of all active spill arrays => total surface oil distribution.
        """
        if not self.active_spills:
            return np.zeros_like(self.environment.grid)
        total_grid = np.zeros_like(self.environment.grid)
        for g in self.active_spills:
            total_grid += g
        return total_grid
