import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environments.gym_carla.carla_env import CarlaEnv
from environments.gym_carla.models import Encoder

# Parallel environments
vec_env = make_vec_env(CarlaEnv, n_envs=1)

model = PPO("MlpPolicy", vec_env, tensorboard_log="logs",verbose=1)
model.learn(total_timesteps=100000,log_interval=1,tb_log_name="PPO")
model.save("ppo_carla")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_carla")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    # vec_env.render("human")