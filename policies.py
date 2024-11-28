from stable_baselines3 import PPO
import numpy as np

class DronePolicy:
    def __init__(self, model_path="drone_policy_model"):
        """Load the trained reinforcement learning model."""
        self.model = PPO.load(model_path)

    def decide_action(self, state):
        """Use the RL model to decide the next action based on the current state."""
        # Preprocess state if necessary (flatten or normalize)
        state_array = np.array([
            *state["drone_position"],
            state["battery_level"] / 100,
            state["weather_conditions"]["visibility"],
        ])
        action, _ = self.model.predict(state_array, deterministic=True)
        return action

# Example usage
if __name__ == "__main__":
    # Initialize policy from saved model
    policy = DronePolicy()

    # Example state
    example_state = {
        "drone_position": (10, 10, 5),
        "battery_level": 80,
        "weather_conditions": {"visibility": 0.9, "wind_speed": 0.3},
        "spill_probabilities": np.zeros((100, 100)),
    }

    # Get decision
    action = policy.decide_action(example_state)
    print(f"Action decided by policy: {action}")
