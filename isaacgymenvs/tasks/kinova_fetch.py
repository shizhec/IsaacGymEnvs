import numpy as np
import math
import os
import sys
import time
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
import torch
from typing import Dict, Any, Tuple
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, get_euler_xyz
from isaacgymenvs.tasks.base.vec_task import VecTask

class KinovaFetch(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render) -> None:
        self.cfg = cfg
        self.graphics_device_id = graphics_device_id
        self.debug_vis = cfg["debug_vis"]

        self.test = cfg["test"]
        self.sub_task = cfg["sub_task"]
        self.random_reset = cfg["env"]["random_reset"]
        self.compute_reward_fn = None
        self.compute_obs_fn = None
        if self.sub_task == "pick_and_hover":
            print("================Initializing Pick and Hover Task================")
            self.compute_reward_fn = compute_pick_and_hover_reward
            self.compute_obs_fn = compute_pick_and_hover_obs
            self.cfg["env"]["numObservations"] = 31  # pos(9) + vel(9) + eef_pos(3) + obj(10)
        elif self.sub_task == "pick_and_reach":
            print("================Initializing Pick and Reach Task================")
            self.compute_reward_fn = compute_pick_and_reach_reward
            self.compute_obs_fn = compute_pick_and_reach_obs
            self.cfg["env"]["numObservations"] = 34  # pos(9) + vel(9) + eef_pos(3) + obj(10) + target(3)
        elif self.sub_task == "pick_and_place":
            print("================Initializing Pick and Place Task================")
            self.compute_reward_fn = compute_pick_and_place_reward
            self.compute_obs_fn = compute_pick_and_place_obs
            self.cfg["env"]["numObservations"] = 34  # pos(9) + vel(9) + eef_pos(3) + obj(10) + target(3)
        elif self.sub_task == "push":
            print("================Initializing Push Task================")
            self.compute_reward_fn = compute_push_reward
            self.compute_obs_fn = compute_push_obs
            self.cfg["env"]["numObservations"] = 34  # pos(9) + vel(9) + eef_pos(3) + obj(10) + target(3)

        # actions include: 'joint_pos': {joint_vel (6) + joint_pos (3)}
        self.cfg["env"]["numActions"] = 9

        # args
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.kinova_dof_noise = self.cfg["env"]["kinovaDofNoise"]
        self.reward_settings = {
            "eef_dist_scale": 0.2,       # Reduced to avoid getting stuck in just reaching
            "grasp_scale": 1.0,          # Increased to emphasize grasping
            "obj_dist_scale": 2.0,       # Higher to prioritize moving object to target
            "lift_scale": 5.0,           # Significant bonus for success
            "hit_on_target_scale": 5.0,  # Bonus for hitting target
            "place_threshold": 0.05,     # Distance threshold for being at target
            "grasp_threshold": 0.03,     # Distance threshold for being able to grasp
            "action_scale": 0.1          # Small penalty for excessive movement
        }

        # Values to be filled in at runtime
        self.states = {}                        # states used for reward calculation
        self.handles = {}                       # handles of kinova and cube
        self.actions = None                     # Current actions to be deployed
        self.n_dofs = None                      # number of dofs per env
        self._obj_state = None                  # object state

        # Tensor placeholders
        self._root_tensor = None                # State of root body            (n_envs, 13) [3 position floats, 4 quaternion floats(orientation), 3 linear velocity floats, 3 angular veloctity floats]
        self._dof_tensor = None                 # States of all dofs            (n_dofs, 2) [Position, Velocity]
        self._rb_tensor = None                  # State of all rigid bodies     (n_envs, n_bodies, 13)
        self._pos = None                        # Joint positions               (n_envs, n_dof)
        self._vel = None                        # Joint velocities              (n_envs, n_dof)
        self._eef_state = None                  # end effector state (at grasping point) (n_envs, 13)
        self._arm_control = None                # Tensor buffer for controlling arm
        self._gripper_control = None            # Tensor buffer for controlling gripper
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, 
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        

        # Success tracking parameters
        self.success_threshold_obj = 0.1  # Distance threshold for success (2cm)
        self.success_duration_obj = 30     # Stay at target for 30 steps
        self.success_counter_obj = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_flag_obj = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.success_threshold_target = 0.05  # Distance threshold for success (2cm)
        self.success_duration_target = 30     # Stay at target for 30 steps
        self.success_counter_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_flag_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        

        self._config_camera()

        # Set observation space bounds for model training stability
        self._setup_observation_space()

        # Reset all environments
        self.reset_all()

        init_steps_to_skip = 1 # some small number of steps

	  # do the below for the first time before any compute observation/reward
        for i in range(init_steps_to_skip):
            print(f"Skipping steps ....{i} to allow the environment settled..")
            if self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            self._refresh()

        self.compute_observations()


    def create_sim(self) -> None:
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81

        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()
        
        self._create_ground_plane()
        self._create_envs()


    def _config_camera(self) -> None:
        # point camera at middle env
        cam_pos = gymapi.Vec3(-1, 0, 2)
        cam_target = gymapi.Vec3(1, 0, -0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _setup_observation_space(self) -> None:
        """Setup observation space bounds for model training stability.

        Uses actual Kinova joint limits from URDF and reasonable workspace bounds.
        This prevents gradient explosion in hybrid dynamics models.
        """
        from gym import spaces

        # Joint limits from Kinova URDF (already loaded in _config_kinova_dofs_props)
        joint_pos_low = self.kinova_lower_limits_pos.cpu().numpy().copy()
        joint_pos_high = self.kinova_upper_limits_pos.cpu().numpy().copy()
        joint_vel_low = self.kinova_lower_limits_vel.cpu().numpy().copy()
        joint_vel_high = self.kinova_upper_limits_vel.cpu().numpy().copy()

        # Threshold for detecting very large values (not technically inf but unreasonable)
        LARGE_VALUE_THRESHOLD = 1e10

        # Replace inf/very large values with reasonable bounds
        # Arm joints (first 6): use ±2π for revolute joints where limits are unreasonable
        for i in range(6):
            if abs(joint_pos_low[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_pos_low[i]):
                joint_pos_low[i] = -2 * np.pi
            if abs(joint_pos_high[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_pos_high[i]):
                joint_pos_high[i] = 2 * np.pi

        # Gripper joints (last 3): ensure reasonable bounds
        for i in range(6, 9):
            if abs(joint_pos_low[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_pos_low[i]):
                joint_pos_low[i] = 0.0
            if abs(joint_pos_high[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_pos_high[i]):
                joint_pos_high[i] = 1.5

        # Velocity bounds: replace unreasonable values with ±10.0
        for i in range(9):
            if abs(joint_vel_low[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_vel_low[i]):
                joint_vel_low[i] = -10.0
            if abs(joint_vel_high[i]) > LARGE_VALUE_THRESHOLD or np.isinf(joint_vel_high[i]):
                joint_vel_high[i] = 10.0

        # Workspace bounds (reasonable estimates for Kinova workspace)
        eef_pos_low = np.array([-2.0, -2.0, 0.0])
        eef_pos_high = np.array([2.0, 2.0, 2.0])
        obj_pos_low = np.array([-2.0, -2.0, 0.0])
        obj_pos_high = np.array([2.0, 2.0, 2.0])
        target_pos_low = np.array([-2.0, -2.0, 0.0])
        target_pos_high = np.array([2.0, 2.0, 2.0])

        # Object dynamics bounds
        obj_vel_low = np.array([-5.0, -5.0, -5.0])
        obj_vel_high = np.array([5.0, 5.0, 5.0])
        obj_quat_low = np.array([-1.0, -1.0, -1.0, -1.0])
        obj_quat_high = np.array([1.0, 1.0, 1.0, 1.0])

        # Combine based on observation structure
        if self.num_obs == 34:  # push, pick_and_reach, pick_and_place
            obs_low = np.concatenate([
                joint_pos_low, joint_vel_low, eef_pos_low,
                obj_pos_low, obj_vel_low, obj_quat_low, target_pos_low
            ])
            obs_high = np.concatenate([
                joint_pos_high, joint_vel_high, eef_pos_high,
                obj_pos_high, obj_vel_high, obj_quat_high, target_pos_high
            ])
        elif self.num_obs == 31:  # pick_and_hover
            obs_low = np.concatenate([
                joint_pos_low, joint_vel_low, eef_pos_low,
                obj_pos_low, obj_vel_low, obj_quat_low
            ])
            obs_high = np.concatenate([
                joint_pos_high, joint_vel_high, eef_pos_high,
                obj_pos_high, obj_vel_high, obj_quat_high
            ])
        else:
            # Fallback: use inf bounds
            print(f"[KinovaFetch] Warning: Unknown observation size {self.num_obs}, using inf bounds")
            obs_low = np.ones(self.num_obs) * -np.Inf
            obs_high = np.ones(self.num_obs) * np.Inf

        # Override the default observation space
        self.obs_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        print(f"[KinovaFetch] Set observation space bounds using Kinova joint limits: shape={obs_low.shape}")

    def _create_ground_plane(self) -> None:
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _config_kinova_dofs_props(self):
        # get joint limits and ranges for Kinova
        kinova_dof_props = self.gym.get_asset_dof_properties(self.kinova_asset)
        self.n_dofs = len(kinova_dof_props)
        self.n_dofs_arm = 6
        self.n_dofs_gripper = self.n_dofs - self.n_dofs_arm
        self.kinova_lower_limits_pos = to_torch(kinova_dof_props['lower'], device=self.device, dtype=torch.float32)
        self.kinova_upper_limits_pos = to_torch(kinova_dof_props['upper'], device=self.device, dtype=torch.float32)
        self.kinova_lower_limits_vel = to_torch(-kinova_dof_props['velocity'], device=self.device, dtype=torch.float32)
        self.kinova_upper_limits_vel = to_torch(kinova_dof_props['velocity'], device=self.device, dtype=torch.float32)

        # set default joint_pos to mid_pos
        self.default_dof_pos = to_torch([0.0, 3.14, 4.20, 0.50, -1.70, 0.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32)
        
        # set dof_props for joint_vel control
        kinova_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS                              # control with joint pos, but action is vel
        self.action_scale_arm = self.kinova_upper_limits_vel[:self.n_dofs_arm]              # since actions are cliped between [-1, 1], action scales should be the upper limits
        self.action_scale_gripper = self.kinova_upper_limits_vel[self.n_dofs_arm:]
        
        kinova_dof_props['stiffness'].fill(4000)
        kinova_dof_props['damping'].fill(400)

        return kinova_dof_props


    def _create_envs(self) -> None:
        # load table asset
        table_size = [0.6, 0.8, 0.4]
        table_height = table_size[-1]
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *table_size, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, table_height/2)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               self.cfg["env"]["asset"]["assetRoot"])
        # load object asset
        self.cube_size = 0.07
        cube_asset_options = gymapi.AssetOptions()
        # cube_asset_options.linear_damping = 0.5
        # cube_asset_options.angular_damping = 0.5
        # cube_asset_options.density = 100.0
        # cube_asset_options.max_linear_velocity = 1.0
        # cube_asset_options.max_angular_velocity = 4.0
        cube_asset = self.gym.create_box(self.sim, *[self.cube_size, self.cube_size, self.cube_size], cube_asset_options)
        # if not self.sub_task == "push":
        cube_rs_properties = self.gym.get_asset_rigid_shape_properties(cube_asset)
        if self.sub_task == "push":
            cube_rs_properties[0].friction = 0.5
        else:
            cube_rs_properties[0].friction = 5.0
        # cube_rs_properties[0].rest_offset = 0.0
        # cube_rs_properties[0].contact_offset = 0.000001
        # cube_rs_properties[0].restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(cube_asset, cube_rs_properties)
        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.default_cube_height = table_height + self.cube_size / 2
        self.default_cube_pos = to_torch([0.5, 0.0, self.default_cube_height], device=self.device, dtype=torch.float32)
        self.default_cube_quat = to_torch([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)

        # load kinova assets
        kinova_asset_options = gymapi.AssetOptions()
        kinova_asset_options.fix_base_link = True
        kinova_asset_options.collapse_fixed_joints = False
        kinova_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # kinova_asset_options.armature = 0.01
        kinova_asset_options.thickness = 0.001
        kinova_asset_options.use_mesh_materials = True
        kinova_asset_options.disable_gravity = True

        kinova_asset_file = self.cfg["env"]["asset"]["assetFileNameKinova"]
        print("Loading asset '%s' from '%s'" % (kinova_asset_file, asset_root))
        self.kinova_asset = self.gym.load_asset(self.sim, asset_root, kinova_asset_file, kinova_asset_options)

        kinova_dof_props = self._config_kinova_dofs_props()

        kinova_pose = gymapi.Transform()
        kinova_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        kinova_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.1)
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom_target = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(1, 0, 0))
        self.sphere_geom_debug = gymutil.WireframeSphereGeometry(0.03, 12, 12, sphere_pose, color=(0, 1, 0))

        # set up the env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # # compute aggregate size
        num_kinova_bodies = self.gym.get_asset_rigid_body_count(self.kinova_asset)
        num_kinova_shapes = self.gym.get_asset_rigid_shape_count(self.kinova_asset)
        max_agg_bodies = num_kinova_bodies + 2     # 2 for table + objects
        max_agg_shapes = num_kinova_shapes + 2     # 2 for table + objects

        # cache useful handles
        self.envs = []

        print(f"Creating {self.num_envs} environments")
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            # aggregate begins
            self.gym.begin_aggregate(env, max_agg_bodies, max_agg_shapes, True)

            # add table
            self.gym.create_actor(env, table_asset, table_pose, "table", i, 0, 0)

            # add object
            obj_handle = self.gym.create_actor(env, cube_asset, cube_pose, "object", i, 1, 0)
            color = gymapi.Vec3(0.027, 0.592, 0.902)
            self.gym.set_rigid_body_color(env, obj_handle, 0, gymapi.MESH_VISUAL, color)

            # add kinova
            kinova_handle = self.gym.create_actor(env, self.kinova_asset, kinova_pose, "kinova", i, 2, 0)

            # set dof properties
            self.gym.set_actor_dof_properties(env, kinova_handle, kinova_dof_props)

            # aggregate ends
            self.gym.end_aggregate(env)

            self.envs.append(env)
        
        self.middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]

        self.init_data()
        

    def init_data(self):
        env_ptr = self.envs[0]
        self.table_handle = 0
        self.obj_handle = 1
        self.kinova_handle = 2

        self.handles = {
            # Kinova
            "end_effector": self.gym.find_actor_rigid_body_handle(env_ptr, self.kinova_handle, "j2n6s300_end_effector"),
            "finger_one": self.gym.find_actor_rigid_body_handle(env_ptr, self.kinova_handle, "j2n6s300_link_finger_1"),
            "finger_two": self.gym.find_actor_rigid_body_handle(env_ptr, self.kinova_handle, "j2n6s300_link_finger_2"),
            "finger_three": self.gym.find_actor_rigid_body_handle(env_ptr, self.kinova_handle, "j2n6s300_link_finger_3")
        }

        # init tensor buffer
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self._root_tensor = gymtorch.wrap_tensor(_root_state).view(self.num_envs, -1, 13)   # (n_envs, n_actors, 13)
        self._dof_tensor = gymtorch.wrap_tensor(_dof_states).view(self.num_envs, -1, 2)     # (n_envs, n_dofs, 2(pos, vel))
        self._rb_tensor = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)      # (n_envs, n_rbs, 13)
        self._pos = self._dof_tensor[..., 0]                                                # (n_envs, n_dofs)
        self._vel = self._dof_tensor[..., 1]                                                # (n_envs, n_dofs)
        self._eef_state = self._rb_tensor[:, self.handles["end_effector"], :]      
        self._eef_f1_state = self._rb_tensor[:, self.handles["finger_one"], :]              # (n_envs, 13)
        self._eef_f2_state = self._rb_tensor[:, self.handles["finger_two"], :]              # (n_envs, 13)
        self._eef_f3_state = self._rb_tensor[:, self.handles["finger_three"], :]            # (n_envs, 13)
        self._obj_state = self._root_tensor[:, self.obj_handle, :]

        # Initialize target position (num_envs, pos)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        # Initialize arm action controls
        self._pos_control = torch.zeros((self.num_envs, self.n_dofs), dtype=torch.float, device=self.device)
        self._arm_control = self._pos_control[:, :self.n_dofs_arm]
        self._gripper_control = self._pos_control[:, self.n_dofs_arm:]

        # Initialize indices (globle, each envs contains 3 actors (table cube arm))
        self._global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)


    def _reset_kinova_dofs(self, env_ids):
        reset_noise = torch.rand((len(env_ids), self.n_dofs), device=self.device, dtype=torch.float32)
        pos = torch.clamp(self.default_dof_pos.unsqueeze(0) +
                          self.kinova_dof_noise * 2.0 * (reset_noise - 0.5),
                          self.kinova_lower_limits_pos.unsqueeze(0), self.kinova_upper_limits_pos.unsqueeze(0))
        
        # open up the gripper at beginning
        pos[:, self.n_dofs_arm:].fill_(0.0)

        # refresh pos and vel
        self._pos[env_ids, :] = pos
        self._vel[env_ids, :] = torch.zeros_like(self._vel[env_ids])

        # reset vel control
        self._pos_control[env_ids, :] = pos

        # reset kinova
        multi_env_ids_int32 = self._global_indices[env_ids, self.kinova_handle].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_tensor),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))


    def sample_uniform_tensor(self, n, low, high) -> torch.Tensor:
        """
        Sample n values uniformly from [low, high) as a PyTorch tensor

        Args:
            n (int): Number of samples to generate
            low (float): Lower bound (inclusive)
            high (float): Upper bound (exclusive)

        Returns:
            torch.Tensor: Tensor of shape (n,) with uniformly sampled values
        """
        # Generate uniform samples between 0 and 1
        samples = torch.rand(n, device=self.device, dtype=torch.float32)

        # Scale and shift to desired range
        samples = samples * (high - low) + low

        return samples


    def sample_polar_positions(self, n, center_x, center_y, radius_min, radius_max) -> torch.Tensor:
        """
        Sample n positions using polar coordinates around a center point

        Args:
            n (int): Number of samples to generate
            center_x (float): X coordinate of the center point
            center_y (float): Y coordinate of the center point
            radius_min (float): Minimum radius from center
            radius_max (float): Maximum radius from center

        Returns:
            torch.Tensor: Tensor of shape (n, 2) with (x, y) coordinates
        """
        # Sample angles uniformly from [0, 2*pi)
        angles = torch.rand(n, device=self.device, dtype=torch.float32) * 2 * math.pi

        # Sample radii uniformly from [radius_min, radius_max)
        radii = self.sample_uniform_tensor(n, radius_min, radius_max)

        # Convert polar to Cartesian coordinates
        x = center_x + radii * torch.cos(angles)
        y = center_y + radii * torch.sin(angles)

        # Stack into (n, 2) tensor
        positions = torch.stack([x, y], dim=-1)

        return positions


    def _reset_object_position(self, env_ids, random=True):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)

        if random:
            # random sample x, y, z
            x = self.sample_uniform_tensor(num_resets, 0.475, 0.525).unsqueeze(1)
            y = self.sample_uniform_tensor(num_resets, -0.05, 0.05).unsqueeze(1)
            z = to_torch([self.default_cube_height], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            
            sampled_obj_pos = torch.cat([x, y, z], dim=-1)

            # random quaternions
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            sampled_obj_quat = torch.zeros(num_resets, 4, device=self.device, dtype=torch.float32)
            sampled_obj_quat[:, -1] = 1.0
            aa_rot[:, 2] = 0.5 * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_obj_quat[:, :] = quat_mul(axisangle2quat(aa_rot), sampled_obj_quat[:, :])
        else:
            sampled_obj_pos = self.default_cube_pos.repeat(num_resets, 1)
            sampled_obj_quat = self.default_cube_quat.repeat(num_resets, 1)

        self._obj_state[env_ids, :3] = sampled_obj_pos
        self._obj_state[env_ids, 3:7] = sampled_obj_quat
        multi_env_ids_objs_int32 = self._global_indices[env_ids, self.obj_handle].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_tensor),
            gymtorch.unwrap_tensor(multi_env_ids_objs_int32), len(multi_env_ids_objs_int32))


    def _reset_target_position(self, env_ids, random=True):
        num_resets = len(env_ids)
        min_distance = 0.2 if self.sub_task == "pick_and_reach" else 0.1  # Minimum required distance between target and object

        # Initial sampling for all positions
        if self.sub_task == "pick_and_hover":
            x = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            y = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            z = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
        elif self.sub_task == "pick_and_reach":
            if random:
                x = self.sample_uniform_tensor(num_resets, 0.45, 0.55).unsqueeze(1)
                y = self.sample_uniform_tensor(num_resets, -0.10, 0.10).unsqueeze(1)
                z = self.sample_uniform_tensor(num_resets, self.default_cube_height + 0.30, self.default_cube_height + 0.35).unsqueeze(1)
            else:
                x = to_torch([0.5], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
                y = to_torch([-0.05], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
                z = to_torch([self.default_cube_height + 0.3], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
        elif self.sub_task in ["pick_and_place", "push"]:
            if random:
                x_y = self.sample_polar_positions(num_resets, center_x=0.5, center_y=0.0, radius_min=0.10, radius_max=0.15)
                x = x_y[:, 0].unsqueeze(1)
                y = x_y[:, 1].unsqueeze(1)
                z = to_torch([self.default_cube_height], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            else:
                x = to_torch([0.55], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
                y = to_torch([-0.05], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
                z = to_torch([self.default_cube_height], device=self.device, dtype=torch.float32).repeat(num_resets, 1)


        sampled_target_pos = torch.cat([x, y, z], dim=-1)        

        # Get object positions
        obj_positions = self._obj_state[env_ids, :3]
        
        # Calculate distances and find invalid samples
        distances = torch.norm(sampled_target_pos - obj_positions, dim=1)
        invalid_mask = distances < min_distance
        
        # Resample only invalid positions
        while torch.any(invalid_mask):
            num_invalid = invalid_mask.sum().item()
            
            # Resample positions for invalid ones
                    # Initial sampling for all positions
            if self.sub_task == "pick_and_hover":
                x_new = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_invalid, 1)
                y_new = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_invalid, 1)
                z_new = to_torch([0.0], device=self.device, dtype=torch.float32).repeat(num_invalid, 1)
            elif self.sub_task == "pick_and_reach":
                x_new = self.sample_uniform_tensor(num_invalid, 0.4, 0.6).unsqueeze(1)
                y_new = self.sample_uniform_tensor(num_invalid, -0.15, 0.15).unsqueeze(1)
                z_new = self.sample_uniform_tensor(num_invalid, self.default_cube_height + 0.25, self.default_cube_height + 0.35).unsqueeze(1)
            elif self.sub_task in ["pick_and_place", "push"]:
                x_new = self.sample_uniform_tensor(num_invalid, 0.4, 0.6).unsqueeze(1)
                y_new = self.sample_uniform_tensor(num_invalid, -0.15, 0.15).unsqueeze(1)
                z_new = to_torch([self.default_cube_height], device=self.device, dtype=torch.float32).repeat(num_invalid, 1)
            new_samples = torch.cat([x_new, y_new, z_new], dim=-1)
            
            # Update only invalid positions
            sampled_target_pos[invalid_mask] = new_samples
            
            # Recalculate distances and invalid mask
            distances = torch.norm(sampled_target_pos - obj_positions, dim=1)
            invalid_mask = distances < min_distance

        # Update target positions
        self.target_pos[env_ids, :] = sampled_target_pos

        # draw axis and sphere at target position
        if self.viewer is not None:
            self.gym.clear_lines(self.viewer)
            for i in range(num_resets):
                env_id = env_ids[i]
                target_pos = sampled_target_pos[i]

                # Create random position for placing objects
                target_pose = gymapi.Transform(gymapi.Vec3(*target_pos), gymapi.Quat(0.0, 0.0, 0.0, 1.0))

                gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[env_id], target_pose)
                gymutil.draw_lines(self.sphere_geom_target, self.gym, self.viewer, self.envs[env_id], target_pose)


    def reset_all(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        
        # reset kinova dofs pos and vel
        self._reset_kinova_dofs(env_ids)
        
        # reset object position
        self._reset_object_position(env_ids, random=self.random_reset)

        # reset target position
        self._reset_target_position(env_ids, random=self.random_reset)

        if self.test:
            # Calculate success rate as the ratio of successful environments to total environments
            success_rate_obj = torch.sum(self.success_flag_obj).item() / self.num_envs
            success_rate_target = torch.sum(self.success_flag_target).item() / self.num_envs
            print(f"Success rate (reaching obj): {success_rate_obj:.2f}  |  Success rate (reaching target): {success_rate_target:.2f}")

            # Reset success tracking for these environments
            self.success_counter_obj[env_ids] = 0
            self.success_counter_target[env_ids] = 0
            self.success_flag_obj[env_ids] = False
            self.success_flag_target[env_ids] = False

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _update_states(self):

        self.states.update({
            # Kinova
            "pos": self._pos[:, :],                    # 6 arm joints + 3 gripper joints
            "vel": self._vel[:, :],
            "arm_pos": self._pos[:, :self.n_dofs_arm],
            "arm_vel": self._vel[:, :self.n_dofs_arm],
            "gripper_pos": self._pos[:, self.n_dofs_arm:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_lin_vel": self._eef_state[:, 7:10], # Linear velocity (x, y, z)
            "eef_ang_vel": self._eef_state[:, 10:13], # Angular velocity (x, y, z)
            "eef_f1_pos": self._eef_f1_state[:, :3],
            "eef_f2_pos": self._eef_f2_state[:, :3],
            "eef_f3_pos": self._eef_f3_state[:, :3],
            # objects
            "obj_pos": self._obj_state[:, :3],
            "obj_quat": self._obj_state[:, 3:7],
            "obj_lin_vel": self._obj_state[:, 7:10],  # Linear velocity (x, y, z)
            "obj_ang_vel": self._obj_state[:, 10:13],  # Angular velocity (x, y, z)
            "eef_to_obj_pos": self._eef_state[:, :3] - self._obj_state[:, :3],
            "obj_to_target_pos": self._obj_state[:, :3] - self.target_pos,
            "eef_to_obj_quat": quat_mul(quat_conjugate(self._obj_state[:, 3:7]), self._eef_state[:, 3:7]),
            # targets
            "target_pos": self.target_pos,
            "eef_pos_to_target": self._eef_state[:, :3] - self.target_pos,
        })

    
    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    
    def compute_action_cost(self, actions: torch.Tensor):
        u_arm, u_gripper = actions[:, :self.n_dofs_arm], actions[:, self.n_dofs_arm:]

        arm_cost = torch.norm(u_arm, dim=-1)

        if self.actions == None:
            gripper_cost = torch.norm(u_gripper, dim=-1)
        else:
            last_u_gripper = self.actions[:, self.n_dofs_arm:]
            gripper_cost = torch.norm(last_u_gripper - u_gripper, dim=-1)
        
        action_cost = arm_cost + gripper_cost
        self.states.update({"action_cost": action_cost})


    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        self.compute_action_cost(actions)
        self.actions = actions.clone().to(self.device)

        # u_arm: joint vel
        # u_gripper: bool -> joint pos
        u_arm, u_gripper = self.actions[:, :self.n_dofs_arm], self.actions[:, self.n_dofs_arm:]

        # control arm
        self._pos_control[:, :self.n_dofs_arm] = self._pos_control[:, :self.n_dofs_arm] + self.dt * u_arm * self.action_scale_arm

        # control gripper
        self._pos_control[:, self.n_dofs_arm:] = self._pos_control[:, self.n_dofs_arm:] + self.dt * u_gripper * self.action_scale_gripper

        # clip action
        self._pos_control = torch.clamp(self._pos_control, self.kinova_lower_limits_pos, self.kinova_upper_limits_pos)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = self.compute_reward_fn(
            self.reset_buf, self.progress_buf, self.states, self.reward_settings, self.max_episode_length)


    def compute_observations(self):
        self._refresh()

        obs = self.compute_obs_fn()
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        if self.viewer and self.debug_vis:
            self.gym.clear_lines(self.viewer)
            eef_pos = self.states['eef_pos'][0]
            self.draw_debug_geom(eef_pos)

        return self.obs_buf
    

    def compute_extra(self):
        self.extras.update({"states": torch.cat((self.states['pos'], self.states['vel']), dim=-1)})


    def get_state(self):
        return torch.cat((self.states['pos'], self.states['vel']), dim=-1)


    def get_state_dim(self):
        return self.n_dofs * 2
    

    def check_success(self):
        """
        Check if the end-effector has reached the target position and remained there for sufficient time.
        """
        # Calculate distance from end-effector to target
        dist_to_target = torch.norm(self.states['obj_to_target_pos'], dim=-1)
        dist_to_obj = torch.norm(self.states['eef_to_obj_pos'], dim=-1)
        
        # Check which environments have the end-effector close enough to the target
        at_obj = dist_to_obj < self.success_threshold_obj
        at_target = dist_to_target < self.success_threshold_target
        
        # Increment counters for environments where arm is at target
        self.success_counter_obj = torch.where(at_obj, self.success_counter_obj + 1, torch.zeros_like(self.success_counter_obj))
        self.success_counter_target = torch.where(at_target, self.success_counter_target + 1, torch.zeros_like(self.success_counter_target))

        # Set success flag for environments where counter exceeds success duration
        self.success_flag_obj = self.success_flag_obj | self.success_counter_obj >= self.success_duration_obj
        self.success_flag_target = self.success_flag_target | self.success_counter_target >= self.success_duration_target


    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        self.progress_buf += 1

        # Check for success
        if self.test:
            self.check_success()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()
        self.compute_extra()


    def draw_debug_geom(self, pos):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[0], pose)
        gymutil.draw_lines(self.sphere_geom_debug, self.gym, self.viewer, self.envs[0], pose)


    def simulate(self) -> None:
        while not self.gym.query_viewer_has_closed(self.viewer):
            
            action = torch.randn((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
            self.step(action)
            # print(self.states["eef_quat"])


        print("Done")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    
    def ik_pick_hover(self):
        """
        Simulate pick and hover behavior using inverse kinematics control for batched environments
        """
        # Initialize tensors for IK control
        damping = 0.05
        
        # Function to calculate damped least squares solution
        def control_ik(dpose, j_eef, damping):
            j_eef_T = torch.transpose(j_eef, 1, 2)
            lmbda = torch.eye(6, device=dpose.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(dpose.shape[0], -1)
            return u

        def orientation_error(desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

        # Acquire Jacobian tensor
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "kinova")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        
        # Get end-effector Jacobian
        j_eef = jacobian[:, self.handles["end_effector"] - 1, :, :6]  # Adjust index if needed
        
        # Main control loop
        while True:
            self._refresh()

            # Get current states
            current_eef_pos = self.states["eef_pos"]
            current_eef_quat = self.states["eef_quat"]
            obj_pos = self.states["obj_pos"]
            
            # Phase 1: Move to grasp position
            grasp_pos = obj_pos.clone()
            grasp_pos[:, 2] += 0.01  # 1cm above object for grasping
            
            # Phase 3: Lift position
            hover_pos = obj_pos.clone()
            hover_pos[:, 2] = self.default_cube_height + 0.05  # 5cm lift height
            
            # Set target position based on phase
            eef_to_obj_xy = self.states["eef_to_obj_pos"][:, :2]
            horizontal_dist = torch.norm(eef_to_obj_xy, dim=-1)
            eef_obj_dist = torch.norm(self.states["eef_to_obj_pos"], dim=-1)
            print(eef_obj_dist, horizontal_dist)
            ready_to_grasp = torch.logical_and(horizontal_dist < 0.01, eef_obj_dist < 0.03)

            gripper_closing = torch.norm(self.states["gripper_pos"], dim=-1) > 1.0
            ready_to_lift = ready_to_grasp & gripper_closing
            desired_pos = torch.where(ready_to_lift, hover_pos, grasp_pos)
            print(ready_to_lift)
            
            # Keep orientation vertical (adjust if needed)
            desired_quat = torch.tensor([0.0, -1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            
            # Calculate position and orientation error
            pos_err = desired_pos - current_eef_pos
            orn_err = orientation_error(desired_quat, current_eef_quat)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            
            # Compute joint velocities using IK
            dof_pos = control_ik(dpose, j_eef, damping)
            
            # Create action tensor (joint velocities + gripper control)
            actions = torch.zeros((self.num_envs, 9), device=self.device)
            actions[:, :6] = dof_pos  # Joint velocities
            
            # Gripper control: close when close to object
            gripper_action = torch.where(ready_to_grasp.unsqueeze(-1).repeat(1, 3),
                                    torch.ones((self.num_envs, 3), device=self.device),   # Close gripper
                                    torch.zeros((self.num_envs, 3), device=self.device))  # Open gripper
            actions[:, 6:] = gripper_action
            
            # Apply actions and step simulation
            self.step(actions)


def compute_pick_and_hover_obs():
    """
    Observation for HybridDynamicsModel with gripper URDF (j2n6s300_gripper_clean.urdf):
    - Robot state: [pos, vel] (18) - 9 DOFs (6 arm + 3 gripper)
    - eef_pos: (3) - recomputed by kinematics, kept for consistency
    - Scene dynamic: object state (10)
    Total: 9 + 9 + 3 + 10 = 31 dims
    Note: gripper_pos removed (already in pos[6:9])
    """
    return [
        "pos",                # (9) All joint positions: 6 arm + 3 gripper - for analytical model
        "vel",                # (9) All joint velocities: 6 arm + 3 gripper
        "eef_pos",            # (3) End effector position - recomputed by kinematics
        # Scene dynamic (learned residuals):
        "obj_pos",            # (3) Object position
        "obj_lin_vel",        # (3) Object linear velocity
        "obj_quat",           # (4) Object orientation
    ]


def compute_pick_and_reach_obs():
    """
    Observation for HybridDynamicsModel with gripper URDF (j2n6s300_gripper_clean.urdf):
    - Robot state: [pos, vel] (18) - 9 DOFs (6 arm + 3 gripper)
    - eef_pos: (3) - recomputed by kinematics, kept for consistency
    - Scene dynamic: object state + target (13)
    Total: 9 + 9 + 3 + 13 = 34 dims
    Note: gripper_pos removed (already in pos[6:9])
    """
    return [
        "pos",                # (9) All joint positions: 6 arm + 3 gripper - for analytical model
        "vel",                # (9) All joint velocities: 6 arm + 3 gripper
        "eef_pos",            # (3) End effector position - recomputed by kinematics
        # Scene dynamic (learned residuals):
        "obj_pos",            # (3) Object position
        "obj_lin_vel",        # (3) Object linear velocity
        "obj_quat",           # (4) Object orientation
        "target_pos",         # (3) Target position - static, learns ~0 residual
    ]


def compute_pick_and_place_obs():
    """
    Observation for HybridDynamicsModel with gripper URDF (j2n6s300_gripper_clean.urdf):
    - Robot state: [pos, vel] (18) - 9 DOFs (6 arm + 3 gripper)
    - eef_pos: (3) - recomputed by kinematics, kept for consistency
    - Scene dynamic: object state + target (13)
    Total: 9 + 9 + 3 + 13 = 34 dims
    Note: gripper_pos removed (already in pos[6:9])
    """
    return [
        "pos",                # (9) All joint positions: 6 arm + 3 gripper - for analytical model
        "vel",                # (9) All joint velocities: 6 arm + 3 gripper
        "eef_pos",            # (3) End effector position - recomputed by kinematics
        # Scene dynamic (learned residuals):
        "obj_pos",            # (3) Object position
        "obj_lin_vel",        # (3) Object linear velocity
        "obj_quat",           # (4) Object orientation
        "target_pos",         # (3) Target position - static, learns ~0 residual
    ]


def compute_push_obs():
    """
    Observation for HybridDynamicsModel with gripper URDF (j2n6s300_gripper_clean.urdf):
    - Robot state: [pos, vel] (18) - 9 DOFs (6 arm + 3 gripper)
    - eef_pos: (3) - recomputed by kinematics, kept for consistency
    - Scene dynamic: object state + target (13)
    Total: 9 + 9 + 3 + 13 = 34 dims
    Note: gripper_pos removed (already in pos[6:9])
    """
    return [
        "pos",                # (9) All joint positions: 6 arm + 3 gripper - for analytical model
        "vel",                # (9) All joint velocities: 6 arm + 3 gripper
        "eef_pos",            # (3) End effector position - recomputed by kinematics
        # Scene dynamic (learned residuals):
        "obj_pos",            # (3) Object position
        "obj_lin_vel",        # (3) Object linear velocity
        "obj_quat",           # (4) Object orientation
        "target_pos",         # (3) Target position - static, learns ~0 residual
    ]

"""
PPO Reward Functions

1. Pick and Hover
@torch.jit.script
def compute_pick_and_hover_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    obj_height = states["obj_pos"][:, 2]
    table_height = 0.4
    
    # Get gripper position (from fully closed 1.51 to fully open 0.0)
    gripper_pos = torch.mean(states["gripper_pos"], dim=-1)  # Average finger positions
    
    # 1. Reaching reward - small reward to encourage reaching
    reaching_reward = 0.2 * (1.0 - torch.tanh(5.0 * dist_hand_obj))
    
    # 2. Grasping reward - encourage closing gripper when close to object
    close_to_obj = dist_hand_obj < 0.04
    grasping_reward = torch.where(close_to_obj, 
                                 0.5 * (gripper_pos / 1.51),  # Normalized gripper closure
                                 0.0)
    
    # 3. Lifting reward - significant reward for lifting object
    lift_height = obj_height - table_height
    lifting_reward = torch.where(lift_height > 0.1,
                                10.0 * torch.tanh(10.0 * lift_height),
                                0.0)
    
    # Combine rewards
    reward = reaching_reward + grasping_reward + lifting_reward

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf

2. Pick and Reach
@torch.jit.script
def compute_pick_and_reach_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    dist_obj_target = torch.norm(states["obj_to_target_pos"], dim=-1)
    obj_height = states["obj_pos"][:, 2]
    table_height = 0.4
    
    # Get gripper position (from fully closed 1.51 to fully open 0.0)
    gripper_pos = torch.mean(states["gripper_pos"], dim=-1)  # Average finger positions
    
    # 1. Reaching reward - small reward to encourage reaching
    reaching_reward = 0.2 * (1.0 - torch.tanh(5.0 * dist_hand_obj))
    
    # 2. Grasping reward - encourage closing gripper when close to object
    close_to_obj = dist_hand_obj < 0.04
    grasping_reward = torch.where(close_to_obj, 
                                 0.5 * (gripper_pos / 1.51),  # Normalized gripper closure
                                 0.0)
    
    # 3. Lifting reward - significant reward for lifting object
    lift_height = obj_height - table_height
    lifting_reward = torch.where(lift_height > 0.1,
                                5 * torch.tanh(10.0 * lift_height),
                                0.0)
    
    # 4. Target reward - only when lifted
    is_lifted = lift_height > 0.1
    target_reward = torch.where(is_lifted,
                               10 * (1.0 - torch.tanh(5.0 * dist_obj_target)),
                               0.0)
    
    # Combine rewards
    reward = reaching_reward + grasping_reward + lifting_reward + target_reward
    
    # Success bonus
    success = is_lifted & (dist_obj_target < 0.01)
    reward = torch.where(success, reward + 15.0, reward)

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf

3. Pick and Place
4. Push

"""


@torch.jit.script
def compute_pick_and_hover_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    obj_height = states["obj_pos"][:, 2]
    obj_default_height = 0.435
    
    # Get gripper position (from fully closed 1.51 to fully open 0.0)
    gripper_pos = torch.mean(states["gripper_pos"], dim=-1)  # Average finger positions
    
    # 1. Reaching reward - small reward to encourage reaching
    reaching_reward = 0.1 * (1.0 - torch.tanh(15.0 * dist_hand_obj))

    # 2. Grasping reward - encourage closing gripper when close to object
    close_to_obj = dist_hand_obj < 0.04
    grasping_reward = torch.where(close_to_obj, 
                                 0.2 * (gripper_pos / 1.51),  # Normalized gripper closure
                                 0.0)
    
    # 3. Lifting reward - significant reward for lifting object
    lift_height = obj_height - obj_default_height
    lifting_reward = torch.where(close_to_obj,
                                10.0 * torch.tanh(1.5 * lift_height),
                                0.0)
    
    # Combine rewards
    reward = reaching_reward + grasping_reward + lifting_reward
    # print("reaching_reawrd", reaching_reward[0])
    # print("grasping_reward", grasping_reward[0])
    # print("lifting_reward", lifting_reward[0])

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf


@torch.jit.script
def compute_pick_and_reach_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    dist_obj_target = torch.norm(states["obj_to_target_pos"], dim=-1)
    obj_height = states["obj_pos"][:, 2]
    obj_default_height = 0.435
    
    # Get gripper position (from fully closed 1.51 to fully open 0.0)
    gripper_pos = torch.mean(states["gripper_pos"], dim=-1)  # Average finger positions
    
    # 1. Reaching reward - small reward to encourage reaching
    reaching_reward = 0.1 * (1.0 - torch.tanh(15.0 * dist_hand_obj))

    # 2. Grasping reward - encourage closing gripper when close to object
    close_to_obj = dist_hand_obj < 0.04
    grasping_reward = torch.where(close_to_obj, 
                                 0.2 * (gripper_pos / 1.51),  # Normalized gripper closure
                                 0.0)
    
    # 3. Lifting reward - significant reward for lifting object
    lift_height = obj_height - obj_default_height
    lifting_reward = torch.where(close_to_obj,
                                5 * torch.tanh(1.5 * lift_height),
                                0.0)
    
    # 4. Target reward - only when lifted
    is_lifted = lift_height > 0.15
    target_reward = torch.where(is_lifted,
                               10 * (1.0 - torch.tanh(8.0 * dist_obj_target)),
                               0.0)

    # Combine rewards
    reward = reaching_reward + grasping_reward + lifting_reward + target_reward
    
    # Success bonus
    success = close_to_obj & (dist_obj_target < 0.02)
    reward = torch.where(success, reward + 15.0, reward)

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf


@torch.jit.script
def compute_pick_and_place_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    dist_obj_target = torch.norm(states["obj_to_target_pos"], dim=-1)
    obj_height = states["obj_pos"][:, 2]
    obj_default_height = 0.435
    
    # Get gripper position (from fully closed 1.51 to fully open 0.0)
    gripper_pos = torch.mean(states["gripper_pos"], dim=-1)  # Average finger positions
    
    # 1. Reaching reward - small reward to encourage reaching
    reaching_reward = 0.1 * (1.0 - torch.tanh(15.0 * dist_hand_obj))

    # 2. Grasping reward - encourage closing gripper when close to object
    close_to_obj = dist_hand_obj < 0.04
    grasping_reward = torch.where(close_to_obj, 
                                 0.2 * (gripper_pos / 1.51),  # Normalized gripper closure
                                 0.0)
    
    # 3. Lifting reward - significant reward for lifting object
    lift_height = obj_height - obj_default_height
    lifting_reward = torch.where(close_to_obj,
                                5 * torch.tanh(15.0 * lift_height),
                                0.0)
    
    # 4. Target reward
    target_reward = torch.where(close_to_obj,
                                10 * (1.0 - torch.tanh(10.0 * dist_obj_target)),
                                0.0)
    
    # Combine rewards
    reward = reaching_reward + grasping_reward + lifting_reward + target_reward
    
    # Success bonus
    far_from_obj = dist_hand_obj > 0.05
    success = far_from_obj & (dist_obj_target < 0.02)
    reward = torch.where(success, reward + 15.0, reward)

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf


@torch.jit.script
def compute_push_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    # Get key distances and states
    dist_hand_obj = torch.norm(states["eef_to_obj_pos"], dim=-1)
    dist_obj_target = torch.norm(states["obj_to_target_pos"], dim=-1)
    
    # Give reaching reward first
    reaching_reward = 0.5 * (1.0 - torch.tanh(10.0 * dist_hand_obj))
    
    close_to_obj = dist_hand_obj < 0.1
    # Give stronger reward for pushing object closer to target

    pushing_reward = torch.where(
        close_to_obj,
        0.5 * (1.0 - torch.tanh(10.0 * dist_obj_target)),  # Double the scale for stronger signal
        torch.zeros_like(reward)
    )
    
    # Success bonus when very close to target
    success_bonus = torch.where(
        dist_obj_target < 0.02,
        1.0 * torch.ones_like(reward),  # Increased success bonus
        torch.zeros_like(reward)
    )
    
    # Combine rewards
    reward = reaching_reward + pushing_reward + success_bonus

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf


@torch.jit.script
def compute_push_reward_codex(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]
    """Push reward that uses planar distances and velocity progress for smoother shaping."""

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

    eef_to_obj = states["eef_to_obj_pos"]
    obj_to_target = states["obj_to_target_pos"]
    obj_lin_vel = states["obj_lin_vel"]

    # Use planar distances for reaching/pushing to avoid penalizing safe hover height.
    dist_hand_obj_xy = torch.norm(eef_to_obj[:, :2], dim=-1)
    dist_obj_target_xy = torch.norm(obj_to_target[:, :2], dim=-1)
    vertical_offset = torch.abs(eef_to_obj[:, 2])

    # Smooth reaching reward following existing scaling convention.
    reaching_reward = reward_settings["eef_dist_scale"] * (1.0 - torch.tanh(5.0 * dist_hand_obj_xy))

    close_xy = dist_hand_obj_xy < 0.06
    near_surface = vertical_offset < 0.10
    contact_window = close_xy & near_surface

    # Stronger reward once we are in a good pushing pose.
    pushing_reward = torch.where(
        contact_window,
        reward_settings["obj_dist_scale"] * (1.0 - torch.tanh(3.0 * dist_obj_target_xy)),
        torch.zeros_like(reward)
    )

    # Reward moving the object toward the target when in contact.
    obj_velocity_xy = obj_lin_vel[:, :2]
    progress_velocity = -torch.sum(obj_to_target[:, :2] * obj_velocity_xy, dim=-1) / (torch.norm(obj_to_target[:, :2], dim=-1) + 1e-6)
    velocity_reward = torch.where(
        contact_window,
        0.1 * torch.clamp(progress_velocity, min=0.0),
        torch.zeros_like(reward)
    )

    # Keep the end-effector close to table height when near the cube.
    height_penalty = torch.where(
        contact_window,
        0.05 * torch.clamp(vertical_offset - 0.03, min=0.0),
        torch.zeros_like(reward)
    )

    success_bonus = torch.where(
        dist_obj_target_xy < reward_settings["place_threshold"],
        reward_settings["hit_on_target_scale"] * torch.ones_like(reward),
        torch.zeros_like(reward)
    )

    reward = reaching_reward + pushing_reward + velocity_reward + success_bonus - height_penalty

    # Reset only when max episode length is reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1,
                           torch.ones_like(reset_buf),
                           reset_buf)

    return reward, reset_buf


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    Converts scaled axis-angle to quat.
    Args:
        vec (tensor): (..., 3) tensor where final dim is (ax,ay,az) axis-angle exponential coordinates
        eps (float): Stability value below which small values will be mapped to 0

    Returns:
        tensor: (..., 4) tensor where final dim is (x,y,z,w) vec4 float quaternion
    """
    # type: (Tensor, float) -> Tensor
    # store input shape and reshape
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # Grab angle
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # Create return array
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # Grab indexes where angle is not zero an convert the input to its quaternion form
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # Reshape and return output
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat
