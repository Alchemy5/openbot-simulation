import habitat_sim
import numpy as np
import quaternion
import cv2

class Simulation:
    """
    Wrapper class for OpenBot simulation trials
    """
    def __init__(self):
        self.hallway_env_path = "data/scene_datasets/hm3d-minival-habitat-v0.2/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
        self.config_file_path = "./data/scene_datasets/hm3d-minival-habitat-v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        self.simulation_settings = {
            "width": 256, # agent camera width
            "height": 256, # agent camera height
            "scene": self.hallway_env_path, # sets hallway scene
            "scene_dataset": self.config_file_path, # sets config file
            "default_agent" : 0,
            "sensor_height": 0.3, # camera height above ground
            "color_sensor": True, # agent has rgb cam
            "depth_sensor": True, # agent has depth cam
            "seed": 1,
            "enable_physics": True, # kinematics enabled
        }
        self.cfg = self._create_sim_config(self.simulation_settings) 
        self.sim = habitat_sim.Simulator(self.cfg) # simulation instance created
        self.agent = sim.initialize_agent(self.simulation_settings["default_agent"]) # agent instance created
        self.agent_state = habitat_sim.AgentState()
        self.agent_state.position = np.array([2.4347131,-2.6599462,1.5344026]) # initial position of agent
        self.agent_state.rotation = quaternion.quaternion(-0.642787933349609, 0, -0.766044139862061, 0) # initial orientation of agent
        self.agent.set_state(agent_state)

    def modify_agent_state(self, position, orientation):
        """
        Change agent's state and return observations.
        Position - numpy array
        Orientation - rotation matrix
        Observations include:
            -rgb img (numpy array 3 channel)
            -depth img (numpy array 1 channel)
            -collision detection (boolean)
        """
        self.agent_state.position = position
        self.agent_state.rotation = quaternion.from_rotation_matrix(orientation)
        self.agent.set_state(agent_state)
        observations = sim.step("move_forward")
        rgb_img = observations['color_sensor']
        rgb_img = np.array(rgb_img)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        depth_img = observations['depth_sensor']
        depth_img = Image.fromarray((depth_img / 10 * 255).astype(np.uint8), mode="L")
        depth_img = depth_img.convert('RGB')
        depth_img = np.array(depth_img)
        depth_img = cv2.cvtColor(depth_arr, cv2.COLOR_BGR2GRAY)
        is_collision = observations["collided"]
        return rgb_img, depth_img, is_collision

    def _create_sim_config(self, settings):
        """
        Returns AI Habitat simulation instance.
        """
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

"""
SIMULATION_CORNERS:
[(-2.528, -2.739, 1.211), 
(-2.528, -2.739, 1.816),
(2.061, -2.739, 1.211),
(2.061, -2.739, 1.816)]
"""

# Sample sanity check

def get_random_rotation_matrix():
    var1 = round(random.uniform(-0.940, 0.940),3)
    var2 = round(random.uniform(0.342, 1.000), 3)
    var3 = round(random.uniform(-1.000, -0.342), 3)
    var4 = round(random.uniform(-0.940, 0.940), 3)
    return [[var1, 0.0, var2], [-0.0, 1.0, 0.0], [var3, -0.0, var4]]

sim = Simulation()
sample_pos = np.array([-2.528, -2.739, 1.211])
sample_rotation_matrix = get_random_rotaton_matrix()
rgb, depth, is_collision = sim.modify_agent_state(sample_pos, sample_rotation_matrix)
print(f"Collided: {is_collision}")
print("Writing RGB and depth images...")
cv2.imwrite("rgb.jpg", rgb)
cv2.imwrite("depth.jpg", depth)