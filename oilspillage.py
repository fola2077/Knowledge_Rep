# oil_spillage.py

import numpy as np
import random
import math
import logging

from config import WIDTH, HEIGHT, CELL_SIZE

class OilSpillage:
    def __init__(self, environment, start_position, volume, oil_type='Light Crude'):
        """
        Initialize the oil spillage event.

        Parameters:
            environment: The Environment instance.
            start_position: Tuple (x, y) in pixels where the spill originates.
            volume: The initial volume of the oil spill (arbitrary units).
            oil_type: Type of oil spilled, affecting spread and weathering.
        """
        self.environment = environment
        self.start_position = start_position  # Position in pixels
        self.volume = volume
        self.oil_type = oil_type
        self.time_elapsed = 0  # Time since spill started
        self.grid = np.zeros_like(environment.grid)  # Oil concentration grid matching environment
        self.spread_rate = self.calculate_spread_rate(oil_type)
        self.evaporation_rate = self.calculate_evaporation_rate(oil_type)
        self.logger = logging.getLogger('OilSpillage')
        self.initialize_spill()

    def initialize_spill(self):
        """
        Place the initial oil concentration on the grid.
        """
        grid_x = int(self.start_position[0] // CELL_SIZE)
        grid_y = int(self.start_position[1] // CELL_SIZE)
        if 0 <= grid_x < self.environment.grid_width and 0 <= grid_y < self.environment.grid_height:
            self.grid[grid_x, grid_y] = self.volume
            self.logger.info(f"Oil spill initialized at grid cell ({grid_x}, {grid_y}) with volume {self.volume}.")
            print(f"Oil spill volume set at grid cell ({grid_x}, {grid_y})")
        else:
            self.logger.error("Oil spill start position is out of bounds.")
            print("Error: Oil spill start position is out of bounds.")

    def calculate_spread_rate(self, oil_type):
        """
        Calculate the spread rate based on the oil type.
        """
        # Simplified spread rate; you can adjust based on oil type
        if oil_type == 'Light Crude':
            return 0.1
        elif oil_type == 'Heavy Crude':
            return 0.05
        else:
            return 0.08  # Default spread rate

    def calculate_evaporation_rate(self, oil_type):
        """
        Calculate the evaporation rate based on the oil type.
        """
        if oil_type == 'Light Crude':
            return 0.02
        elif oil_type == 'Heavy Crude':
            return 0.005
        else:
            return 0.01  # Default evaporation rate      

    def update(self, dt, weather_system):
        """
        Update the oil spill over time.

        Parameters:
            dt: Time delta since the last update (seconds).
            weather_system: The WeatherSystem instance.
        """
        self.time_elapsed += dt
        self.spread_oil(dt, weather_system)
        self.weather_oil(dt, weather_system)
        max_conc = np.max(self.grid)
        print(f"Oil spill update - Time: {self.time_elapsed:.2f}s, Max concentration: {max_conc}")

    def spread_oil(self, dt, weather_system):
        """
        Simulate the spreading of oil on the grid.

        Parameters:
            dt: Time delta since the last update (seconds).
            weather_system: The WeatherSystem instance.
        """
        # Diffusion component
        diffusion_coefficient = self.spread_rate
        self.grid = self.diffuse(self.grid, diffusion_coefficient, dt)

        # Advection component (wind effect)
        wind_speed = weather_system.current_state.wind_speed
        wind_direction = weather_system.current_state.wind_direction
        self.grid = self.advect(self.grid, wind_speed, wind_direction, dt)

    def diffuse(self, grid, diffusion_coefficient, dt):
        """
        Apply diffusion to the oil concentration grid.

        Parameters:
            grid: Current oil concentration grid.
            diffusion_coefficient: Rate at which oil diffuses over the grid.
            dt: Time delta (seconds).
        """
        # Simple diffusion using convolution with a kernel
        kernel_size = 3
        scaling_factor = 10  # Adjust based on grid size
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        diffused_grid = grid + diffusion_coefficient * dt * scaling_factor * self.convolve(grid, kernel)
        return diffused_grid

    def convolve(self, grid, kernel):
        """
        Apply convolution to the grid with given kernel.

        Parameters:
            grid: Grid to convolve.
            kernel: Convolution kernel.
        """
        from scipy.ndimage import convolve
        return convolve(grid, kernel, mode='constant', cval=0.0)

    def advect(self, grid, wind_speed, wind_direction, dt):
        """
        Apply advection to the oil concentration grid based on wind.

        Parameters:
            grid: Current oil concentration grid.
            wind_speed: Speed of the wind (m/s).
            wind_direction: Direction of the wind (degrees).
            dt: Time delta (seconds).
        """
        # Calculate shift in x and y based on wind speed and direction
        angle_rad = math.radians(wind_direction)
        advection_scaling = 0.5 # Adjust based on grid size
        dx = np.cos(angle_rad) * wind_speed * dt * advection_scaling 
        dy = np.sin(angle_rad) * wind_speed * dt * advection_scaling

        dx_int = int(round(dx))
        dy_int = int(round(dy))

        shifted_grid = np.roll(grid, shift=dx_int, axis=0)
        shifted_grid = np.roll(shifted_grid, shift=dy_int, axis=1)

        # Apply boundary conditions
        if dx > 0:
            shifted_grid[:int(dx), :] = 0
        elif dx < 0:
            shifted_grid[int(dx):, :] = 0
        if dy > 0:
            shifted_grid[:, :int(dy)] = 0
        elif dy < 0:
            shifted_grid[:, int(dy):] = 0

        return shifted_grid
    
    def weather_oil(self, dt, weather_system):
        """
        Simulate weathering processes like evaporation.

        Parameters:
            dt: Time delta (seconds).
            weather_system: The WeatherSystem instance.
        """
        # Evaporation reduces oil volume
        temperature = weather_system.current_state.temperature  # °C
        evaporation_factor = self.evaporation_rate * (1 + (temperature - 15) / 100)  # Adjust based on temperature
        self.grid -= self.grid * evaporation_factor * dt
        self.grid = np.maximum(self.grid, 0)  # Ensure no negative concentrations

