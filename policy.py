# policy.py

import pygame
import sys
import random
import numpy as np
import logging
from pygame.math import Vector2
from drone import Drone
from environment import Environment, WATER_LEVEL
from weather import WeatherSystem, TimeManager
from oilspillage import OilSpillage  
from config import WIDTH, HEIGHT, CELL_SIZE

class DroneEnv:
    def __init__(self, num_drones=5, max_steps=1000):

        # **Initialize environment parameters**

        self.num_drones = num_drones
        self.max_steps = max_steps
        self.min_steps = 100
        self.current_step = 0

        # **STATES: Initialize the environment and its components (environment, time, weather, oil spillage)**

        self.environment = Environment()
        self.time_manager = TimeManager()
        self.weather_system = WeatherSystem(self.time_manager)
        self.oil_spillage = OilSpillage(self.environment, self.time_manager)
        self.environment.set_oil_spillage_manager(self.oil_spillage)

        # **STATES: Initialize drones with their positions and load them into the environment**

        self.drones = [
            Drone(id=i, position=(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)),
                  environment=self.environment, weather_system=self.weather_system,
                  time_manager=self.time_manager)
            for i in range(num_drones)
        ]
        for drone in self.drones:
            drone.load_environment(self.environment, self.oil_spillage)

        # **ACTIONS: Define the action space (0: Up, 1: Down, 2: Left, 3: Right, 4: Scan for Oil)**
        self.action_space = 5  # 0: Up, 1: Down, 2: Left, 3: Right, 4: Scan for Oil,

        # **OBSERVATIONS: Define the observation space dimensions**
        self.observation_space = 6  # x, y, wind_speed, visibility, oil_detected, is_returning, oil_scanned

    def reset(self):
        """Reset the environment to an initial state."""
        # Reset environment state
        self.current_step = 0
        self.environment.reset()
        self.oil_spillage.reset()
        self.time_manager.reset()
        self.weather_system.reset()

        # Reset drones' states
        for drone in self.drones:
            drone.position = Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
            drone.reset()
            drone.detected_cells.clear()  
            drone.scan_mode = False      
        return self._get_obs()

    def step(self, actions):
        """
        Execute actions for all drones.

        """
        rewards = []    # **REWARDS: Initialize reward list for all drones**
        infos = []      # Information about each drone's action
        prev_total_oil = self.oil_spillage.get_total_oil()


        # For each drone, process their action
        for drone, action in zip(self.drones, actions):
            reward = 0.0  # Initialize reward for this drone
            detected = False
            detection_delay = None

            # **STATES: Update drone's time since last detection**
            drone.time_since_last_detection = min(drone.time_since_last_detection + 1, 100)  # Cap at 100 to limit penalty

            # **REWARDS: Apply non-detection penalty**
            non_detection_penalty = -0.01
            reward += non_detection_penalty

            # **ACTIONS & TRANSITION FUNCTION: Process movement or scanning actions**
            if action in [0, 1, 2, 3]:
                dx, dy = self._action_to_movement(action)
                drone.move(dx, dy, self.drones)
                
                # **EXOGENOUS FUNCTION & TRANSITION FUNCTION: Check for oil detection**
                detected, detection_delay = drone.detect_oil()

                if detected:
                    drone.time_since_last_detection = 0

                    if detection_delay is not None:
                        # ** REWARDS: Apply quick detection bonus **
                        quick_detection_bonus = max(0.0, 5.0 - 0.05 * detection_delay)  # Adjusted parameters
                        reward += quick_detection_bonus

                        # ** REWARDS: Apply base detection reward based on oil concentration**
                        concentration = self.oil_spillage.get_cell_concentration(drone.position)
                        detection_reward = 10.0 * concentration
                        reward += detection_reward

                        # ** REWARDS: First-Time Detection Bonus**
                        cell = (int(drone.position.x // CELL_SIZE), int(drone.position.y // CELL_SIZE))
                        if cell not in drone.detected_cells:
                            first_detection_bonus = 2.0
                            reward += first_detection_bonus
                            drone.detected_cells.add(cell)
                    else:
                        # Handle cases where detection_delay is None
                        # Assumes a default small bonus or skip
                        quick_detection_bonus = 0.0
                        print(f"Note: detection_delay is None for drone {drone.id} at position ({drone.position.x}, {drone.position.y})")

            elif action == 4:  # Scan action
                 # ** POLICY: Drone decides to scan for oil **
                detected = drone.scan_for_oil(frames=4)
                if detected:
                    drone.time_since_last_detection = 0
                    detection_delay = self.oil_spillage.get_time_since_spill(drone.position)

                    if detection_delay is not None:
                        # **REWARDS: Apply adjusted scan detection bonus**
                        quick_detection_bonus = max(0.0, 4.0 - 0.05 * detection_delay)
                        reward += quick_detection_bonus

                        # **REWARDS: Apply scan detection reward based on oil concentration**
                        concentration = self.oil_spillage.get_cell_concentration(drone.position)
                        scan_detection_reward = 7.0 * concentration
                        reward += scan_detection_reward
                    else:
                        quick_detection_bonus = 0.0
                        print(f"Note: detection_delay is None for drone {drone.id} at position ({drone.position.x}, {drone.position.y})")
                else:
                    # **REWARDS: Penalty for unsuccessful scan**
                    reward -= 0.02  # Reduced penalty for unsuccessful scan

            # **REWARDS: Exploration bonus for visiting new cells**
            cell = (int(drone.position.x // CELL_SIZE), int(drone.position.y // CELL_SIZE))
            if cell not in drone.visited_cells:
                exploration_bonus = 0.5
                reward += exploration_bonus
                drone.visited_cells.add(cell)

            rewards.append(reward)
            info = {"action": action, "detected": detected, "detection_delay": detection_delay}
            infos.append(info)

        # **TRANSITION FUNCTION: Update the environment (time, weather, oil spillage)**
        dt = 1  # Define your time step
        self.time_manager.update(dt)
        self.weather_system.update(dt)
        self.oil_spillage.update(self.weather_system, dt, self.time_manager.get_current_total_minutes())

        # **REWARDS: Apply reward based on oil reduction in the environment**

        current_total_oil = self.oil_spillage.get_total_oil()
        oil_reduction = max(0.0, prev_total_oil - current_total_oil)
        for i in range(self.num_drones):
            # **Adjusted Oil Reduction Reward**
            # Ensured that oil_reduction is non-negative
            rewards[i] += oil_reduction * 0.05  # Adjusted scaling factor

        # Increment step counter
        self.current_step += 1

        # **TERMINATION CONDITION: Check if the maximum number of steps has been reached**
        done = self.current_step >= self.max_steps

        # Gather observations
        obs = self._get_obs()

        return obs, rewards, done, infos

    def _action_to_movement(self, action):
        """Convert discrete action to movement vector."""
        # **ACTIONS: Map action indices to movement directions**
        movement_map = {
            0: (0, -1),  # Up
            1: (0, 1),   # Down
            2: (-1, 0),  # Left
            3: (1, 0)    # Right
        }
        return movement_map.get(action, (0, 0))

    def _get_obs(self):
        """Get observations for all drones."""
        observations = []
        for drone in self.drones:
            wind_speed = self.weather_system.current_state.wind_speed
            visibility = self.weather_system.current_state.visibility
            oil_detected = int(len(drone.detected_cells) > 0)
            oil_scanned = int(getattr(drone, 'scan_mode', False))
            obs = np.array([
                drone.position.x / WIDTH,
                drone.position.y / HEIGHT,
                wind_speed / 50.0,    # Normalize assuming max wind speed 50
                visibility,           # Already between 0 and 1
                oil_detected,
                oil_scanned
            ], dtype=np.float32)
            observations.append(obs)
        return observations

    def render(self):
        """Render the environment."""
        self.environment.draw(screen)  
        for drone in self.drones:
            # Draw drones on the screen
            pygame.draw.circle(screen, drone.color, (int(drone.position.x), int(drone.position.y)), 5)
        pygame.display.flip()
