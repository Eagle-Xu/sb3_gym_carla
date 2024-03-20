import numpy as np
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from environments.gym_carla.carla_env import CarlaEnv
from environments.gym_carla.models import Encoder


env=CarlaEnv()

# The noise objects for DDPG
n_actions = env.action_space.shape[-1] #2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env,verbose=1,action_noise=action_noise,tensorboard_log="logs")
model.learn(total_timesteps=50000, log_interval=4,tb_log_name="DDPG")
model.save("ddpg_carla")
vec_env = model.get_env()

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_carla")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
