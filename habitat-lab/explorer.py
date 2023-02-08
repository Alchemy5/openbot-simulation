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

# function to display the topdown map
from PIL import Image
import cv2
import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
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
            #print("Observation recieved!")
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

def find_and_replace(img, locs):
    for ind, row in enumerate(img):
        for ind_col, col in enumerate(row):
            if (ind,ind_col) in locs:
                img[ind][ind_col] = [0,0,255]
    return img

def highlight_min_pixels(half1, half2, pixel_locs_one, pixel_locs_two):
    half1 = find_and_replace(half1, pixel_locs_one)
    half2 = find_and_replace(half2, pixel_locs_two)
    return half1,half2
    # function returns rgb image with red pixels

def return_count(image, values):
    pixel_count = 0
    pixel_locs = []
    for ind,row in enumerate(image):
        for ind_col,col in enumerate(row):
            if int(col) in values:
                pixel_count += 1
                pixel_locs.append((ind,ind_col))
    return pixel_count, pixel_locs

def find_min(half_image, min_values):
    pixel_count, pixel_locs = return_count(half_image, min_values)
    return pixel_count, pixel_locs

def render_and_return_mins(main, half1, half2, half1_ori=None, half2_ori=None):
    min_values = []
    min_value = np.amax(main)
    threshold = 20
    for i in range(min_value-threshold,min_value+threshold):
        min_values.append(i)

    min_one, pixel_locs_one = find_min(half1, min_values)
    min_two, pixel_locs_two = find_min(half2, min_values)
    #import pdb;pdb.set_trace()
    if half1_ori is not None:
        half1, half2 = highlight_min_pixels(half1_ori, half2_ori, pixel_locs_one, pixel_locs_two)
    cv2.imshow("left_half", half1)
    cv2.imshow("right_half", half2)
    cv2.imshow("Main Output", main)
    #import pdb;pdb.set_trace()
    diff = abs(min_one - min_two)
    return min_one, min_two

def room_strategy():
    try:
        for i in range(0,1):
            action = "move_forward"
            img_arr, depth_arr, collided = navigateAndSee(action)
            depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
            cv2.imshow("main", depth_arr)
            time.sleep(1)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        while True:
            
            #called from collision avoidance loop function
            #spins
            thetas = []
            #spins and theta is incremented by 1, pixel values correspond to theta
            #robot takes a step towards the theta direction with lowest magnitude pixel
            #it moves for 10 rotations counterclockwise and then moves in direction with lowest pixel
            # combine with long term navigation approach where robot has a general direction contained in its memory and it tries to move in this general direction while trying to avoid collisions
            print("Loop to find min value started")
            num_turns_per_scan = 20
            min_value = -10000
            latency = 0.8
            for i in range(0,num_turns_per_scan):
                action = "turn_right"
                img_arr, depth_arr, collided = navigateAndSee(action)
                depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
                cv2.imshow("main", depth_arr)
                time.sleep(latency)
                if np.amax(depth_arr) > min_value:
                    min_value = np.amax(depth_arr)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            for i in range(0,num_turns_per_scan):
                action = "turn_left"
                img_arr, depth_arr, collided = navigateAndSee(action)
                depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
                cv2.imshow("main", depth_arr)
                time.sleep(latency)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            print(f"First loop finished, min-value is {min_value}")
            print("Starting main loop")
            # need to first loop to find min value
            for i in range(0,num_turns_per_scan):
                action = "turn_right"
                img_arr, depth_arr, collided = navigateAndSee(action)
                depth_arr = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
                cv2.imshow("main", depth_arr)
                time.sleep(latency)
                min_values = []
                
                threshold = 100
                #TODO: Rule where if number of max (black pixels) increase after movement, then robot steps back and changes threshold value
                for i in range(min_value-threshold,min_value+threshold):
                    min_values.append(i)
                score,_ = return_count(depth_arr, min_values)
                thetas.append(score)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            # add something to thetas to change up strategy and make it long term
            # sample list of theta values: Theta values: [18711, 22646, 19791, 8915, 2243, 540, 256, 40, 27, 30, 31, 29, 36, 52, 34, 116, 2933, 2342, 2070, 2016]
            # define space as just a circle without any obstacles, non-geometric but logical map
            # robot has to explore and learn to define dead ends
            print(f"Theta values: {thetas}")
            print("Scan Complete")
            min_pixel_value = max(thetas)
            min_pixel_value_index = thetas.index(min_pixel_value)
            for i in range(0,num_turns_per_scan - min_pixel_value_index):
                action = "turn_left"
                img_arr, depth_arr, collided = navigateAndSee(action)
                cv2.imshow("main", depth_arr)
                time.sleep(latency)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
            print("Adjustments made, moving forward...")
            action = "move_forward"
            img_arr, depth_arr, collided = navigateAndSee(action)
            cv2.imshow("main", depth_arr)
            time.sleep(latency)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            print(f"Collision Detected: {collided}")
    finally:
        cv2.destroyAllWindows()

def manual_control(scheme:str, highlight:bool):
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
            print(f"Collision Detected: {collided}")
            one, two = np.split(img_arr, 2, axis=1)
            one_depth, two_depth = np.split(depth_arr, 2, axis=1)    	
            one_depth_gray = cv2.cvtColor(one_depth, cv2.COLOR_BGR2GRAY)
            two_depth_gray = cv2.cvtColor(two_depth, cv2.COLOR_BGR2GRAY)
            one_gray = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
            two_gray = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)
            gray_color = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            gray_depth = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
            if scheme == "depth":
                if highlight:
                    min_one, min_two = render_and_return_mins(
                        gray_depth, one_depth_gray, two_depth_gray, half1_ori=one_depth, half2_ori=two_depth)
                else:
                    min_one, min_two = render_and_return_mins(
                        gray_depth, one_depth_gray, two_depth_gray)
            elif scheme == "color":
                min_one, min_two = render_and_return_mins(
                    gray_color, one_gray, two_gray)
            
            print(f"min-one: {min_one} and min-two: {min_two}")
            if scheme == "depth":
                print(f"Overall min value: {np.amax(gray_depth)}")
            print(f"Agent Position: {agent.get_state().position}")
            print(f"Agent Orientation: {agent.get_state().rotation}")
            if min_one < min_two:
                print("TURN RIGHT")
            elif min_one > min_two:
                print("TURN LEFT")
            else:
                print("GO STRAIGHT")
            print("")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
                
    finally:
        cv2.destroyAllWindows()

def collision_avoidance_loop(scheme:str, highlight:bool):
    action = "move_forward"
    first_time = False
    turn = False
    intensity = 0
    try:
        while True:
            img_arr, depth_arr, collided = navigateAndSee(action)
            if turn:
                img_arr, depth_arr, collided = navigateAndSee("move_forward")
            print(f"Collision Detected: {collided}")
            one, two = np.split(img_arr, 2, axis=1)
            one_depth, two_depth = np.split(depth_arr, 2, axis=1)
            #import pdb;pdb.set_trace()	
            one_depth_gray = cv2.cvtColor(one_depth, cv2.COLOR_BGR2GRAY)
            two_depth_gray = cv2.cvtColor(two_depth, cv2.COLOR_BGR2GRAY)
            one_gray = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
            two_gray = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)
            gray_color = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            gray_depth = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
            if scheme == "depth":
                if highlight:
                    min_one, min_two = render_and_return_mins(
                        gray_depth, one_depth_gray, two_depth_gray, half1_ori=one_depth, half2_ori=two_depth)
                else:
                    min_one, min_two = render_and_return_mins(
                        gray_depth, one_depth_gray, two_depth_gray)
            elif scheme == "color":
                min_one, min_two = render_and_return_mins(
                    gray_color, one_gray, two_gray)
            
            print(f"min-one: {min_one} and min-two: {min_two}")
            if not first_time:
                done_organizing = input("Done organizing? ")
                first_time = True
            
            if min_one < min_two:
                print("TURN RIGHT")
                action = "turn_right"
                turn = True
            elif min_one > min_two:
                print("TURN LEFT")
                action = "turn_left"
                turn = True
            else:
                print("GO STRAIGHT")
                action = "move_forward"
                turn = False
            print("")
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            time.sleep(1)

    finally:
        cv2.destroyAllWindows()

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
    agent_state.rotation = quaternion.quaternion(-0.642787933349609, 0, -0.766044139862061, 0)
    # for hallway tests: np.array([2.4347131,-2.6599462,1.5344026]) and quaternion.quaternion(-0.642787933349609, 0, -0.766044139862061, 0)
    # for room tests: np.array([-4.1224685,-2.7386093,1.5412302]) and quaternion.quaternion(-0.984807908535004, 0, -0.173647731542587, 0)
    #agent_state.position = sim.pathfinder.get_random_navigable_point()
    agent.set_state(agent_state)
    action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
    #import pdb;pdb.set_trace()
    print("Discrete action space: ", action_names)
    #collision_avoidance_loop(scheme="depth", highlight=False)
    # Main Loop
    #collision_avoidance_loop(scheme="depth", highlight=False)
    manual_control(scheme="depth", highlight=False)
    






"""
SCORE FUNCTION ONE

def return_count(image, value):
    pixel_count = 0
    for row in image:
        for j in row:
            if int(j) == value:
                pixel_count += 1
    return pixel_count

def find_min(half_image):
    return return_count(half_image, 0)

"""

"""
SCORE FUNCTION TWO
def return_count(image, value):
    pixel_count = 0
    for row in image:
        for j in row:
            if int(j) == value:
                pixel_count += 1
    return pixel_count

def find_min(half_image, min_value):
    return return_count(half_image, min_value)

"""
"""
SCORE FUNCTION TWO
def return_count(image, value):
    pixel_count = 0
    for row in image:
        for j in row:
            if int(j) >= value:
                pixel_count += 1
    return pixel_count

def find_min(half_image, min_value):
    return return_count(half_image, min_value)

"""

"""
Score function 2, ever-evolving
takes an image, starts by randomly assigning scores to each half
then updates scores based on cost and punnishments
cost = intensity value decreases
punnishment = intensity value increases
perhaps a score function that attempts to find the ideal min_value based on increase in intensity
"""