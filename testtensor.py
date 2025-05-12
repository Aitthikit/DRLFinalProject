import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON file
file_path = "obs/Env4/Env4"
with open(file_path, 'r') as f:
    obs_data = json.load(f)

# Convert to numpy array and squeeze singleton dimension
obs_array = np.array(obs_data).squeeze(axis=1)  # Shape: (3000, 6)

# Define descriptive names for each state
state_names = [
    "Cart Position (x)",
    "Cart Velocity (ẋ)",
    "Pole 1 Angle (θ₁)",
    "Pole 1 Angular Velocity (θ̇₁)",
    "Pole 2 Angle (θ₂)",
    "Pole 2 Angular Velocity (θ̇₂)"
]

# Plot each state separately
timesteps = np.arange(obs_array.shape[0])
for i in range(len(state_names)):
    plt.figure()
    plt.plot(timesteps, obs_array[:, i])
    plt.title(f'{state_names[i]} over Time')
    plt.xlabel('Timestep')
    plt.ylabel(state_names[i])
    plt.grid(True)
    plt.tight_layout()
    plt.show()
