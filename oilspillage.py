# oilspillage.py

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
    def __init__(self, environment, initial_cells, initial_concentration=1.0, creation_time_minutes=0):
        self.environment = environment
        self.grid_width = environment.grid_width
        self.grid_height = environment.grid_height
        self.concentration = np.zeros((self.grid_width, self.grid_height), dtype=float)
        for (gx, gy) in initial_cells:
            self.concentration[gx, gy] = initial_concentration
        self.detected_concentration = np.zeros_like(self.concentration)
        self.status = 'active' # 'active', 'stopped', 'pending_removal'
        self.time_since_all_detected = 0.0 # In sim hours
        self.detection_time = np.full((self.grid_width, self.grid_height), np.nan)
        self.pending_removal_start_time = None
        self.detected = False
        self.creation_time_minutes = creation_time_minutes

    def spread(self, wind_dx, wind_dy):
        """
        Spread the oil spill using convolution to simulate diffusion.
        """
        if self.status != 'active':
            return # Do not spread if spill is not active
        
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

        # Adjusts the kernel based on wind direction
        # Shifting the kernel in the direction of the wind
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
        evaporation_rate = 0.0005 + 0.00005 * (temperature - 15)  # Adjust as needed
        evaporation_rate = np.clip(evaporation_rate, 0, 0.02)
        self.concentration *= (1 - evaporation_rate)

    def mark_detected(self, gx, gy):
        """
        Mark a spepcific cell as detected.
        """
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            self.detected_concentration[gx, gy] = self.concentration[gx, gy]

    def get_total_oil(self):
        """
        Get the total amount of oil concentration in the spill.
        """
        return np.sum(self.concentration)
    
    def get_total_detected_oil(self):
        """
        Get the total amount of detected oil concentration.
        """
        return np.sum(self.detected_concentration)
    
    def update_status(self, current_total_minutes):
        """
        Update the status of the spill based on detection.
        """
        total_oil = self.get_total_oil()
        total_detected = self.get_total_detected_oil()
        detection_ratio = total_detected / total_oil if total_oil > 0 else 0.0

        detection_threshold = 0.90

        self.detected = detection_ratio >= detection_threshold

        if total_oil == 0:
            self.status = 'removed'
            return  # No oil left


        if self.status == 'active' and detection_ratio >= 0.5:
            self.status = 'stopped'
            print("Oil spill spread has been stopped as 50% has been detected.")
        
        if self.status == 'stopped' and detection_ratio >= detection_threshold:
            self.status = 'pending_removal'
            if self.pending_removal_start_time is None:
                self.pending_removal_start_time = current_total_minutes
                print("All oil spill has been detected. Will remove in 3 simulation hours.")
        
        if self.status == 'pending_removal':
            time_in_pending = current_total_minutes - self.pending_removal_start_time
            if time_in_pending >= 3 * 60:
                self.concentration = np.zeros_like(self.concentration)
                self.detected_concentration = np.zeros_like(self.detected_concentration)
                print("Oil spill has been removed after full detection.")
                self.status = 'removed'  # mark as removed

    def is_active(self):
        """
        Check if the spill still has significant oil concentration.
        """
        return np.any(self.concentration > 0.001) and self.status != 'removed'
    
    def expire_detections(self, current_time_minutes, expiry_duration_minutes=6 * 60):
        """
        Clears detections older than 'expiry_duration' hours.
        """
        expired = self.detection_time < (current_time_minutes - expiry_duration_minutes)
        self.detected_concentration[expired] = 0.0
        self.detection_time[expired] = np.nan
        num_expired = np.sum(expired)
        if num_expired > 0:
            print(f"{num_expired} detections expired.")

class OilSpillage:
    """
    Manages all oil spills in the environment.
    """
    def __init__(self, environment, time_manager):
        self.environment = environment
        self.time_manager = time_manager
        self.logger = logging.getLogger('OilSpillage')
        self.spills = []
        self.next_spawn_day = 0  # First spawn at Day 1
        self.last_expansion_time = None  # Track last expansion time
        self.next_spawn_time_minutes = 0
        self.min_active_spills = 3 # Min number
        self.max_active_spills = 10

    def reset(self):
        """
        Resets the oil spills to their initial state.
        """
        self.spills = []
        self.next_spawn_day = self.time_manager.day_count + 1  # Reset to spawn on next day
        self.next_spawn_time_minutes = self.time_manager.get_current_total_minutes() # Spawn immediately
        self.last_expansion_time = None
        self.logger.info("Oil spills have been reset.")

    def update(self, weather_system, dt, current_total_minutes):
        """
        Update oil spills based on time and environmental factors.
        """

        if self.last_expansion_time is None:
            self.last_expansion_time = current_total_minutes

        # Every 3 hours, update spills
        if (current_total_minutes - self.last_expansion_time) >= (3*60):
            self.last_expansion_time = current_total_minutes
            wind_dir = weather_system.current_state.wind_direction
            wind_speed = weather_system.current_state.wind_speed
            temperature = weather_system.current_state.temperature
            wind_dx, wind_dy = self.get_wind_vector(wind_dir, wind_speed)
            self.update_spills(wind_dx, wind_dy, temperature, dt, current_total_minutes)

        # Check if we should spawn new spills
        active_spills_count = len([spill for spill in self.spills if spill.is_active()])
        if active_spills_count < self.min_active_spills:
            spills_to_spawn = self.min_active_spills - active_spills_count
            total_spills_after_spawn = active_spills_count + spills_to_spawn
            if total_spills_after_spawn > self.max_active_spills:
                spills_to_spawn = self.max_active_spills - active_spills_count
            self.spawn_new_spills(spills_to_spawn)


    def spawn_new_spills(self, num_spills=None):
        """
        Spawn new spills at random locations on water.
        """
        if num_spills is None:
            num_spills = random.randint(5, 9)  # Default 

        current_time_minutes = self.time_manager.get_current_total_minutes()
        
        for _ in range(num_spills):
            block_cells = self.find_random_4cell_block()
            if not block_cells:
                self.logger.warning("Failed to find a suitable location for new oil spill.")
                continue

            # Pass creation_time_minutes to track when the spill was created
            spill = OilSpill(self.environment, block_cells, creation_time_minutes=current_time_minutes)
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

    def update_spills(self, wind_dx, wind_dy, temperature, dt, current_total_minutes):
        """
        Update each spill's spread and evaporation.
        """
        active_spills = []
        for spill in self.spills:
            spill.spread(wind_dx, wind_dy)
            spill.evaporate_and_dissolve(temperature)
            spill.update_status(current_total_minutes)
            current_time = self.time_manager.current_sim_time
            spill.expire_detections(current_time)
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

    def get_total_oil(self):
        """
        Returns the total oil concentration remaining in the environment.
        """
        total_oil = 0.0
        for spill in self.spills:
            total_oil += np.sum(spill.concentration)
        return total_oil

    def get_total_detected_oil(self):
        """
        Returns the total detected oil concentration.
        """
        total_detected = 0.0
        for spill in self.spills:
            total_detected += np.sum(spill.detected_concentration)
        return total_detected

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
    
    def combined_detected_concentration(self):
        """
        Combine detected oil concentrations from all active spills.
        """
        w, h = self.environment.grid_width, self.environment.grid_height
        total_detected = np.zeros((w, h), dtype=float)
        for spill in self.spills:
            total_detected += spill.detected_concentration
        # Ensure values are between 0 and 1
        total_detected = np.clip(total_detected, 0, 1)
        return total_detected

    def get_cell_color(self, concentration):
        """
        Map oil concentration to a color for rendering.
        Returns a more visible purple color for undetected oil.
        """
        # Define the color for maximum concentration (dark purple)
        max_concentration_color = np.array([128, 0, 128])  # More visible purple

        # Define the color for minimum concentration (lighter purple)
        min_concentration_color = np.array([200, 100, 200])  # Lighter purple

        # Normalize concentration to range [0,1]
        norm_concentration = np.clip(concentration, 0, 1)

        # Interpolate between min and max colors
        color = norm_concentration * max_concentration_color + (1 - norm_concentration) * min_concentration_color

        # Ensure values are within valid RGB range
        color = np.clip(color, 0, 255).astype(int)

        return tuple(color)
    

    def mark_cell_detected(self, gx, gy, detection_time_minutes):
        """
        Marks a specific cell as detected across all spills.
        """
        detection_found = False
        for spill in self.spills:
            if 0 <= gx < spill.grid_width and 0 <= gy < spill.grid_height:
                if spill.concentration[gx, gy] > 0:
                    spill.detected_concentration[gx, gy] = spill.concentration[gx, gy]
                    spill.detection_time[gx, gy] = detection_time_minutes
                    detection_found = True
                    print(f"Oil detected and marked at cell ({gx}, {gy}) with concentration {spill.concentration[gx, gy]:.5f}")
        
        if detection_found:
            self.logger.info(f"Cell ({gx}, {gy}) marked as detected.")

    def all_spills_detected(self):
        """Check if all active spills have been detected."""
        return False

    def get_cell_concentration(self, position):
        """
        Returns the combined oil concentration at the given world position.
        """
        # Convert world position to grid indices
        gx = int(position.x // CELL_SIZE)
        gy = int(position.y // CELL_SIZE)

        # Ensure indices are within bounds
        if 0 <= gx < self.environment.grid_width and 0 <= gy < self.environment.grid_height:
            total_concentration = 0.0
            for spill in self.spills:
                total_concentration += spill.concentration[gx, gy]
            
            # Clamp the total concentration to [0, 1]
            total_concentration = min(total_concentration, 1.0)
            return total_concentration
        else:
            # If position is out of bounds, returns zero concentration
            return 0.0
    
    def get_time_since_spill(self, position):
        """
        Returns the time elapsed since the spill at the given position appeared.
        """
        gx = int(position.x // CELL_SIZE)
        gy = int(position.y // CELL_SIZE)
        if not (0 <= gx < self.environment.grid_width and 0 <= gy < self.environment.grid_height):
            print(f"Position ({gx}, {gy}) out of bounds.")
            return None

        min_time_since_spill = float('inf')
        found_spill = False
        for spill in self.spills:
            if spill.concentration[gx, gy] > 0:
                found_spill = True
                time_since_spill = self.time_manager.get_current_total_minutes() - spill.creation_time_minutes
                if time_since_spill < min_time_since_spill:
                    min_time_since_spill = time_since_spill

        if not found_spill:
            print(f"No spill found at grid position ({gx}, {gy})")
            return None  # No spill at this position
        else:
            return min_time_since_spill