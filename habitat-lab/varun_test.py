import os
import numpy as np
import quaternion

import matplotlib.pyplot as plt

import habitat

import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor

# Set up the environment for testing
config = habitat.get_config(config_paths='habitat/config/tasks/pointnav_rgbd.yaml')
config.defrost()
config.habitat.dataset.data_path = 'data/datasets/pointnav/habitat-test-scenes/v1/val/val.json.gz'
config.habitat.dataset.scenes_dir = 'data/scene_datasets/'
config.freeze()

# Can also do directly in the config file
config.defrost()
config.habitat.simulator.depth_sensor.normalize_depth = False
config.freeze()

# Intrinsic parameters, assuming width matches height. Requires a simple refactor otherwise
W = config.habitat.simulator.depth_sensor.width
H = config.habitat.simulator.depth_sensor.height

assert(W == H)
hfov = float(config.habitat.simulator.depth_sensor.hfov) * np.pi / 180.


env = habitat.Env(config=config)


obs = env.reset()
initial_state = env._sim.get_agent_state(0)
init_translation = initial_state.position
init_rotation = initial_state.rotation