import math
import os
import random
import sys
import time
import git
import imageio
import magnum as mn
import numpy as np
import quaternion
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

def make_simple_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            rgb = observations['color_sensor']
            rgb_img = Image.fromarray(rgb.astype(np.uint8), mode="RGBA")
            rgb_img = rgb_img.convert('RGB')
            img_arr = np.array(rgb_img)
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            depth_img = observations['depth_sensor']
            collision = observations["collided"]
            depth_img = Image.fromarray((depth_img / 10 * 255).astype(np.uint8), mode="L")
            depth_img = depth_img.convert('RGB')
            depth_arr = np.array(depth_img)
            return img_arr, depth_arr, collision

def manual_control():
    direction = ""
    try:
        while direction != "o":
            direction = input("Enter direction of motion: ")
            if direction == "w":
                action = "move_forward"
            elif direction == "a":
                action = "turn_left"
            elif direction == "d":
                action = "turn_right"
            else:
                continue
            img_arr, depth_arr, collided = navigateAndSee(action)
            gray_depth = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
            cv2.imshow("Main",gray_depth)
            print(f"Agent Position: {agent.get_state().position}")
            print(f"Quaternion: {agent.get_state().rotation}")
            print(f"Agent Orientation: {quaternion.as_rotation_matrix(agent.get_state().rotation)}")
            #range of rotation matrix, rule is that third element of first row always has to be greater than positive 0.5, generate dictionary of all combinations
            print("")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
                
    finally:
        cv2.destroyAllWindows()
def collect_corner_data():
    corners = [(-2.528, -2.739, 1.211), 
        (-2.528, -2.739, 1.816),
        (2.061, -2.739, 1.211),
        (2.061, -2.739, 1.816)]
    for i,corner in enumerate(corners):
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(corner)
        rotation = quaternion.quaternion(-0.642787933349609, 0, -0.766044139862061, 0)
        agent_state.rotation = rotation
        rotation_matrix = quaternion.as_rotation_matrix(rotation)
        agent.set_state(agent_state)
        img_arr, depth_arr, collided = navigateAndSee("move_forward")
        gray_depth = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"cnn_data/corner/{i}.jpg", gray_depth)
        np.save(f"cnn_data/corner/{i}-color.npy", depth_arr)
        np.save(f"cnn_data/corner/{i}-gray.npy",gray_depth) 
        np.save(f"cnn_data/corner/{i}-pos.npy",agent.get_state().position)
        np.save(f"cnn_data/corner/{i}-rot.npy",rotation_matrix)

def get_random_rotation_matrix():
    var1 = round(random.uniform(-0.940, 0.940),3)
    var2 = round(random.uniform(0.342, 1.000), 3)
    var3 = round(random.uniform(-1.000, -0.342), 3)
    var4 = round(random.uniform(-0.940, 0.940), 3)
    return [[var1, 0.0, var2], [-0.0, 1.0, 0.0], [var3, -0.0, var4]]

def data_collection_loop():
    y_value = -2.739
    x_value_bounds = (-2.528,2.061)
    z_value_bounds = (1.211, 1.816)  
    rotation_matrix = [[]]
    for i in range(0,1000):
        random_position_array = [random.uniform(*x_value_bounds), y_value, random.uniform(*z_value_bounds)]
        random_value = np.array(random_position_array)
        agent_state = habitat_sim.AgentState()
        agent_state.position = random_value
        #next step randomize quaternion
        rotation_matrix = get_random_rotation_matrix()
        rotation = quaternion.from_rotation_matrix(rotation_matrix)
        agent_state.rotation = rotation
        agent.set_state(agent_state)
        img_arr, depth_arr, collided = navigateAndSee("move_forward")
        gray_depth = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Main",gray_depth)
        cv2.imwrite(f"cnn_data/{i}.jpg", gray_depth)
        np.save(f"cnn_data/{i}-color.npy", depth_arr)
        np.save(f"cnn_data/{i}-gray.npy",gray_depth)
        np.save(f"cnn_data/{i}-pos.npy",agent.get_state().position)
        np.save(f"cnn_data/{i}-rot.npy",rotation_matrix)

if __name__ == "__main__":
    # @title Configure Sim Settings
    #test_scene = "data/scene_datasets/hm3d-minival-habitat-v0.2/00801-HaxA7YrQdEC/HaxA7YrQdEC.basis.glb"
    test_scene = "data/scene_datasets/hm3d-minival-habitat-v0.2/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
    #test_scene = "data/scene_datasets/hm3d-minival-habitat-v0.2/building_hallway.glb"
    mp3d_scene_dataset = "./data/scene_datasets/hm3d-minival-habitat-v0.2/hm3d_annotated_basis.scene_dataset_config.json"
    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}
    display = True

    sim_settings = {
        "width": 256,  # Spatial resolution of the observations
        "height": 256,
        "scene": test_scene,  # Scene path
        "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
        "default_agent": 0,
        "sensor_height": 0.3,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": True,  # kinematics only
    }
    # Create Simulation
    cfg = make_simple_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([2.4347131,-2.6599462,1.5344026])
    rotation = quaternion.quaternion(-0.642787933349609, 0, -0.766044139862061, 0)
    agent_state.rotation = rotation
    rotation_matrix = quaternion.as_rotation_matrix(rotation)
    agent.set_state(agent_state)
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    print("Discrete action space: ", action_names)
    #collect_corner_data()
    #manual_control()
    data_collection_loop()