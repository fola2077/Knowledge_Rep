import math
import random

class Drone:
    def __init__(self, id, position, battery=100, rotor_speed=1.0):
        """
        Initialize the drone with an ID, initial position, battery level, and rotor speed.
        """
        self.id = id
        self.position = position  # (x, y, z) tuple
        self.battery = battery  # Battery percentage
        self.rotor_speed = rotor_speed  # Speed of the rotor
        self.weather_forecast = None  # Current weather information
        self.depth = 0  # Depth for potential underwater operations (optional)
        self.neighbors = []  # List of nearby drones for information sharing

    def move(self, dx, dy, dz):
        """
        Move the drone by the given offsets in the x, y, and z directions.
        """
        if self.battery <= 0:
            print(f"Drone {self.id} cannot move. Battery depleted!")
            return
        self.position = (
            self.position[0] + dx,
            self.position[1] + dy,
            self.position[2] + dz
        )
        self.battery -= self.calculate_battery_usage(dx, dy, dz)
        print(f"Drone {self.id} moved to {self.position}. Battery: {self.battery}%")

    def calculate_battery_usage(self, dx, dy, dz):
        """
        Calculate battery consumption based on distance moved and rotor speed.
        """
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        usage = distance * self.rotor_speed * 0.5  # Arbitrary battery usage rate
        return min(usage, self.battery)

    def adjust_rotor_speed(self, new_speed):
        """
        Adjust the drone's rotor speed.
        """
        self.rotor_speed = max(0.1, min(new_speed, 5.0))  # Keep rotor speed between 0.1 and 5.0
        print(f"Drone {self.id} rotor speed adjusted to {self.rotor_speed}.")

    def set_weather_forecast(self, forecast):
        """
        Update the weather forecast affecting the drone's operations.
        """
        self.weather_forecast = forecast
        print(f"Drone {self.id} weather forecast updated to {forecast}.")

    def share_information(self):
        """
        Share information with the nearest neighbors. (Commented out for simplicity)
        """
        # if self.neighbors:
        #     for neighbor in self.neighbors:
        #         neighbor.receive_information(f"Drone {self.id}: Current position {self.position}.")
        print(f"Drone {self.id} would share information with neighbors: {self.neighbors}")

    def receive_information(self, info):
        """
        Receive shared information from another drone.
        """
        print(f"Drone {self.id} received info: {info}")

    def report_status(self):
        """
        Report the current status of the drone.
        """
        return {
            "id": self.id,
            "position": self.position,
            "battery": self.battery,
            "rotor_speed": self.rotor_speed,
            "weather_forecast": self.weather_forecast,
            "depth": self.depth
        }
