"""
This file is used to create the cart pole environment for the agent to interact with.
"""

import gymnasium as gym
import numpy as np

"""
The environment has the following specifications:

- Observation space: Box(5,)
    - The observation space is a 5-dimensional vector that contains the following:
        1. Cart position
        2. Cart velocity
        3. Pole angle
        4. Pole angular velocity
        5. Time step
        
- Action space: Discrete(2)
    - The action space is a discrete set of two actions:
        1. Move left
        2. Move right
        
- Reward function:
    - The reward function is a +1 reward for each time step the pole remains upright.
    - Total reward is the time until the pole falls.
- Dynamics:
    - Frictionless cart and pole
    - F = 10N
    - Mass of the cart = 1kg
    - Mass of the pole = 0.1kg
    - Bound the cart position to the range [-3, 3] m
    - Bound the cart velocity to the range [-10, 10] m/s
    - Bound the pole angular velocity to the range [-pi, pi] rad/s
    - Time step = 0.02s and the episode terminates if the time step reaches 20 seconds.
    - The episode terminates if the pole angle is not in the range [-5*pi/12, 5*pi/12].
    - Discount factor = 1
"""


class CartPoleEnv(gym.envs.classic_control.CartPoleEnv):
    def __init__(self):
        self.gravity = 9.8  # Gravity constant
        self.mass_cart = 1.0  # Mass of the cart
        self.mass_pole = 0.1  # Mass of the pole
        self.cart_length = 0.5  # Length of the pole
        self.force_mag = 10.0  # Force magnitude
        self.dt = 0.02  # Time step
        self.max_cart_pos = 3.0  # Maximum cart position
        self.min_cart_pos = -3.0  # Minimum cart position
        self.max_cart_vel = 10.0  # Maximum cart velocity
        self.min_cart_vel = -10.0  # Minimum cart velocity
        self.max_pole_angle = (5.0 * np.pi) / 12.0  # Maximum pole angle
        self.min_pole_angle = (-5.0 * np.pi) / 12.0  # Minimum pole angle
        self.max_pole_angular_vel = np.pi  # Maximum pole angular velocity
        self.min_pole_angular_vel = -np.pi  # Minimum pole angular velocity
        self.max_time_steps = int(20 / self.dt)  # Maximum time steps
        self.reward_range = (-np.inf, np.inf)  # Reward range
        self.action_space = gym.spaces.Discrete(
            2
        )  # Action space: 1 -> Move right, 0 -> Move left
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )  # Observation space

    def reset(self):
        self.time_step = 0
        self.cart_pos = 0.0
        self.cart_vel = 0.0
        self.pole_angle = 0.0
        self.pole_angular_vel = 0.0
        return self._get_observation()

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(self.pole_angle)
        sintheta = np.sin(self.pole_angle)
        temp = (
            force
            + self.mass_pole * self.cart_length * self.pole_angular_vel**2 * sintheta
        ) / (self.mass_cart + self.mass_pole)
        pole_angular_acc = (self.gravity * sintheta - costheta * temp) / (
            self.cart_length
            * (
                (4.0 / 3.0)
                - (self.mass_pole * costheta**2) / (self.mass_cart + self.mass_pole)
            )
        )
        cart_acc = (
            force
            + self.mass_pole
            * self.cart_length
            * (self.pole_angular_vel**2 * sintheta - pole_angular_acc * costheta)
        ) / (self.mass_cart + self.mass_pole)
        self.cart_pos += self.dt * self.cart_vel
        self.cart_vel += self.dt * cart_acc
        if self.cart_vel < self.min_cart_vel:
            self.cart_vel = self.min_cart_vel
        if self.cart_vel > self.max_cart_vel:
            self.cart_vel = self.max_cart_vel
        self.pole_angle += self.dt * self.pole_angular_vel
        self.pole_angular_vel += self.dt * pole_angular_acc
        if self.pole_angular_vel < self.min_pole_angular_vel:
            self.pole_angular_vel = self.min_pole_angular_vel
        if self.pole_angular_vel > self.max_pole_angular_vel:
            self.pole_angular_vel = self.max_pole_angular_vel
        self.time_step += 1
        done = not (
            -self.max_cart_pos <= self.cart_pos <= self.max_cart_pos
            and -self.max_pole_angle <= self.pole_angle <= self.max_pole_angle
            and self.time_step < self.max_time_steps
        )
        reward = 1.0 if not done else 0.0
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        return np.array(
            [
                self.cart_pos,
                self.cart_vel,
                self.pole_angle,
                self.pole_angular_vel,
                self.time_step * self.dt,
            ]
        )
