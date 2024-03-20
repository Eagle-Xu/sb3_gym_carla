import gymnasium as gym
import torch

from stable_baselines3 import SAC
from environments.gym_carla.carla_env import CarlaEnv
from environments.gym_carla.models import Encoder

env = CarlaEnv()

model = SAC("MlpPolicy", env, learning_rate=0.0001,verbose=1,tensorboard_log="logs")
model.learn(total_timesteps=700000, log_interval=1,tb_log_name="SAC")
model.save("sac_carla2")

del model   # remove to demonstrate saving and loading

# model = SAC.load("sac_carla")
#
# obs, info = env.reset()
# obs=torch.Tensor(obs)
# obs=obs.unsqueeze()
# # while True:
# #     action, _states = model.predict(obs, deterministic=True)
# #     obs, reward, terminated, truncated, info = env.step(action)
# #     if terminated or truncated:
# #         obs, info = env.reset()