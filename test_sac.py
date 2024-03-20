import torch

from stable_baselines3 import SAC
from environments.gym_carla.carla_env import CarlaEnv
from environments.gym_carla.models import Encoder

# sac_carla1,50w次
env = CarlaEnv()
model = SAC("MlpPolicy", env, learning_rate=0.0001,verbose=1,tensorboard_log="logs")
del model   # remove to demonstrate saving and loading
model = SAC.load("sac_carla2")

for i in range(50):
    obs, info = env.reset()
    obs = torch.Tensor(obs).unsqueeze(dim=0)
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        obs = torch.Tensor(obs).unsqueeze(dim=0)
        if terminated or truncated:
            break

