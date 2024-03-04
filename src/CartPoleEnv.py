import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.envs.classic_control import utils
from gymnasium import spaces, logger


class CartPoleEnvCustom(CartPoleEnv):
    def __init__(self):
        super(CartPoleEnvCustom, self).__init__(render_mode="human")
        self.x_threshold = 3.0
        self.theta_threshold_radians = 5.0 * np.pi / 12.0
        self.x_dot_threshold = 10.0
        self.theta_dot_threshold = np.pi
        self.max_time_steps = 1000

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = 0.0, 0.0
        self.state = self.np_random.uniform(low=low, high=high, size=(5,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot, time = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":  # means explicit euler, that is,
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        x_dot = np.clip(x_dot, -self.x_dot_threshold, self.x_dot_threshold)
        theta_dot = np.clip(
            theta_dot, -self.theta_dot_threshold, self.theta_dot_threshold
        )

        self.state = (x, x_dot, theta, theta_dot, time + 1)

        terminated = bool(
            x <= -self.x_threshold
            or x >= self.x_threshold
            or theta <= -self.theta_threshold_radians
            or theta >= self.theta_threshold_radians
            or time >= self.max_time_steps
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
