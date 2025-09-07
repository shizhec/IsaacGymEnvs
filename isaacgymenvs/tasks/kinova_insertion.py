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

class KinovaInsertion(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render) -> None:
        self.cfg = cfg
        self.graphics_device_id = graphics_device_id
        self.debug_vis = cfg["debug_vis"]

        # actions include: 'joint_pos': {joint_vel (6) + joint_pos (3)}
        self.cfg["env"]["numActions"] = 6

        # args
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.kinova_dof_noise = self.cfg["env"]["kinovaDofNoise"]
        self.reward_settings = {}

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
        
        self._config_camera()

        # Reset all environments
        self.reset_all()

        # Refresh State
        self._refresh()


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

        factory_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               self.cfg["env"]["asset"]["factoryAssetRoot"])
        peg_file = self.cfg["env"]["asset"]["pegAssetFile"]
        hole_file = self.cfg["env"]["asset"]["holeAssetFile"]
        # load peg asset
        peg_options = gymapi.AssetOptions()
        peg_options.flip_visual_attachments = False
        peg_options.fix_base_link = True
        peg_options.thickness = 0.0  # default = 0.02
        peg_options.armature = 0.0  # default = 0.0
        peg_options.use_physx_armature = True
        peg_options.linear_damping = 0.0  # default = 0.0
        peg_options.max_linear_velocity = 1000.0  # default = 1000.0
        peg_options.angular_damping = 0.0  # default = 0.5
        peg_options.max_angular_velocity = 64.0  # default = 64.0
        peg_options.disable_gravity = False
        peg_options.enable_gyroscopic_forces = True
        peg_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        peg_options.use_mesh_materials = False
        print("Loading asset '%s' from '%s'" % (peg_file, factory_asset_root))
        peg_asset = self.gym.load_asset(self.sim, factory_asset_root, peg_file, peg_options)
        self.peg_pose = gymapi.Transform()
        self.peg_pose.p.x = 0.45
        self.peg_pose.p.y = 0.0
        self.peg_pose.p.z = table_height
        self.peg_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        hole_options = gymapi.AssetOptions()
        hole_options.flip_visual_attachments = False
        hole_options.fix_base_link = False
        hole_options.thickness = 0.0  # default = 0.02
        hole_options.armature = 0.0  # default = 0.0
        hole_options.use_physx_armature = True
        hole_options.linear_damping = 0.0  # default = 0.0
        hole_options.max_linear_velocity = 1000.0  # default = 1000.0
        hole_options.angular_damping = 0.0  # default = 0.5
        hole_options.max_angular_velocity = 64.0  # default = 64.0
        hole_options.disable_gravity = False
        hole_options.enable_gyroscopic_forces = True
        hole_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        hole_options.use_mesh_materials = False
        print("Loading asset '%s' from '%s'" % (hole_file, factory_asset_root))
        hole_asset = self.gym.load_asset(self.sim, factory_asset_root, hole_file, hole_options)
        self.hole_pose = gymapi.Transform()
        self.hole_pose.p.x = 0.5
        self.hole_pose.p.y = 0.0
        self.hole_pose.p.z = table_height
        self.hole_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # load kinova assets
        kinova_asset_options = gymapi.AssetOptions()
        kinova_asset_options.fix_base_link = True
        kinova_asset_options.collapse_fixed_joints = False
        kinova_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # kinova_asset_options.armature = 0.01
        kinova_asset_options.thickness = 0.001
        kinova_asset_options.use_mesh_materials = True
        kinova_asset_options.disable_gravity = True

        kinova_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               self.cfg["env"]["asset"]["assetRoot"])
        kinova_asset_file = self.cfg["env"]["asset"]["assetFileNameKinova"]
        print("Loading asset '%s' from '%s'" % (kinova_asset_file, kinova_asset_root))
        self.kinova_asset = self.gym.load_asset(self.sim, kinova_asset_root, kinova_asset_file, kinova_asset_options)

        kinova_dof_props = self._config_kinova_dofs_props()

        kinova_pose = gymapi.Transform()
        kinova_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        kinova_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # set up the env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # # compute aggregate size
        num_kinova_bodies = self.gym.get_asset_rigid_body_count(self.kinova_asset)
        num_kinova_shapes = self.gym.get_asset_rigid_shape_count(self.kinova_asset)
        max_agg_bodies = num_kinova_bodies + 3     # 2 for table + peg + hole
        max_agg_shapes = num_kinova_shapes + 3     # 2 for table + peg + hole

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

            # add peg and hole
            self.gym.create_actor(env, peg_asset, self.peg_pose, "peg", i, 1, 0)
            self.gym.create_actor(env, hole_asset, self.hole_pose, "hole", i, 2, 0)

            # add kinova
            kinova_handle = self.gym.create_actor(env, self.kinova_asset, kinova_pose, "kinova", i, 4, 0)

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
        self.peg_handle = 1
        self.hole_handle = 2
        self.kinova_handle = 3

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
        self._peg_state = self._root_tensor[:, self.peg_handle, :]
        self._hole_state = self._root_tensor[:, self.hole_handle, :]

        # Initialize arm action controls
        self._pos_control = torch.zeros((self.num_envs, self.n_dofs), dtype=torch.float, device=self.device)
        self._arm_control = self._pos_control[:, :self.n_dofs_arm]
        self._gripper_control = self._pos_control[:, self.n_dofs_arm:]

        # Initialize indices (globle, each envs contains 4 actors (table peg hole arm))
        self._global_indices = torch.arange(self.num_envs * 4, dtype=torch.int32,
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


    def _reset_hole_position(self, env_ids):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)

        hole_noise_xy = 2 * (torch.rand((num_resets, 2), dtype=torch.float32, device=self.device) - 0.5)  # Random noise in [-1, 1]
        hole_noise_xy = hole_noise_xy * torch.tensor([0.02, 0.02], device=self.device)  # Scale noise (e.g., ±2cm)

        hole_pos = torch.zeros((num_resets, 3), dtype=torch.float32, device=self.device)
        hole_pos[:, 0] = self.hole_pose.p.x + hole_noise_xy[:, 0]
        hole_pos[:, 1] = self.hole_pose.p.y + hole_noise_xy[:, 1]
        hole_pos[:, 2] = self.hole_pose.p.z

        # Add rotation noise
        hole_rot_noise = (torch.rand((num_resets, 3), dtype=torch.float32, device=self.device) - 0.5) * 0.2  # Random rotation ±0.1 radians
        hole_rot_euler = (
            torch.zeros((num_resets, 3), dtype=torch.float32, device=self.device)
            + hole_rot_noise              
        )
        hole_rot_quat = quat_from_euler_xyz(hole_rot_euler[:, 0], hole_rot_euler[:, 1], hole_rot_euler[:, 2])

        self._hole_state[env_ids, :3] = hole_pos
        self._hole_state[env_ids, 3:7] = hole_rot_quat

        # Stable the hole
        self._hole_state[env_ids, 7:] = 0.0

        multi_env_ids_objs_int32 = self._global_indices[env_ids, self.hole_handle].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_tensor),
            gymtorch.unwrap_tensor(multi_env_ids_objs_int32), len(multi_env_ids_objs_int32))
        
        self._simulate_and_refresh()


    def _reset_peg_position(self, env_ids):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)

        hole_pos = self._hole_state[env_ids, :3]
        hole_height = 0.010  # Height of the hole
        
        # Set peg position to be above the hole
        self._peg_state[env_ids, :3] = hole_pos.clone()
        self._peg_state[env_ids, 2] += hole_height

        # Generate random noise for peg position
        plug_noise_xy = 2 * (torch.rand((num_resets, 2), dtype=torch.float32, device=self.device) - 0.5)  # Random noise in [-1, 1]
        plug_noise_xy = plug_noise_xy * torch.tensor([0.005, 0.005], device=self.device)  # Scale noise (e.g., ±2cm)

        # Apply XY noise to plugs
        self._peg_state[env_ids, :2] += plug_noise_xy


        peg_quat = torch.zeros((num_resets, 4), dtype=torch.float32, device=self.device)
        peg_quat[:, 0] = self.peg_pose.r.x
        peg_quat[:, 1] = self.peg_pose.r.y
        peg_quat[:, 2] = self.peg_pose.r.z
        peg_quat[:, 3] = self.peg_pose.r.w
        self._peg_state[env_ids, 3:7] = peg_quat

        # Stable the peg
        self._peg_state[env_ids, 7:] = 0.0

        multi_env_ids_objs_int32 = self._global_indices[env_ids, self.peg_handle].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_tensor),
            gymtorch.unwrap_tensor(multi_env_ids_objs_int32), len(multi_env_ids_objs_int32))
        
        self._simulate_and_refresh()


    def reset_all(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def reset_idx(self, env_ids):
        print("resetting envs", env_ids)
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        
        # reset kinova dofs pos and vel
        self._reset_kinova_dofs(env_ids)

        # Close gripper onto plug
        self.disable_gravity()
        self._reset_hole_position(env_ids)
        self._reset_peg_position(env_ids)
        # self._move_gripper_to_grasp_pose(env_ids)
        # self._close_gripper(env_ids)
        self.enable_gravity()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def enable_gravity(self):
        """Enable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity = gymapi.Vec3(*self.cfg["sim"]["gravity"])
        self.gym.set_sim_params(self.sim, sim_params)


    def disable_gravity(self):
        """Disable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = 0.0
        self.gym.set_sim_params(self.sim, sim_params)


    def _update_states(self):

        self.states.update({
            # Kinova
            "pos": self._pos[:, :],                    # 6 arm joints + 3 gripper joints
            "vel": self._vel[:, :],
            "gripper_pos": self._pos[:, self.n_dofs_arm:],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_lin_vel": self._eef_state[:, 7:10], # Linear velocity (x, y, z)
            "eef_ang_vel": self._eef_state[:, 10:13], # Angular velocity (x, y, z)
            "eef_f1_pos": self._eef_f1_state[:, :3],
            "eef_f2_pos": self._eef_f2_state[:, :3],
            "eef_f3_pos": self._eef_f3_state[:, :3],
            # peg
            "peg_pos": self._peg_state[:, :3],
            "peg_quat": self._peg_state[:, 3:7],
            "peg_lin_vel": self._peg_state[:, 7:10],  # Linear velocity (x, y, z)
            "peg_ang_vel": self._peg_state[:, 10:13],  # Angular velocity (x, y, z)
            "eef_to_peg_pos": self._eef_state[:, :3] - self._peg_state[:, :3],
            "eef_to_peg_quat": quat_mul(quat_conjugate(self._peg_state[:, 3:7]), self._eef_state[:, 3:7]),
        })

    
    def _simulate_and_refresh(self):
        self.gym.simulate(self.sim)
        self._refresh()
        self.render()

    
    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()


    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """
        self.actions = actions.clone().to(self.device)

        # u_arm: joint vel
        # u_gripper: bool -> joint pos
        u_arm = self.actions[:, :self.n_dofs_arm]

        # control arm
        self._pos_control[:, :self.n_dofs_arm] = self._pos_control[:, :self.n_dofs_arm] + self.dt * u_arm * self.action_scale_arm

        # control gripper
        # self._pos_control[:, self.n_dofs_arm:] = self._pos_control[:, self.n_dofs_arm:] + self.dt * u_gripper * self.action_scale_gripper

        # clip action
        self._pos_control = torch.clamp(self._pos_control, self.kinova_lower_limits_pos, self.kinova_upper_limits_pos)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_reward(
            self.reset_buf, self.progress_buf, self.states, self.reward_settings, self.max_episode_length)


    def compute_observations(self):
        self._refresh()

        obs = [
            "pos",                # (9) Joint positions
            "eef_pos",            # (3) End effector position
            "eef_lin_vel",        # (3) End effector linear velocity
            "eef_quat",           # (4) End effector orientation
            "peg_pos",            # (3) Object position
            "peg_lin_vel",        # (3) Object linear velocity
            "peg_quat",           # (4) Object orientation
            "eef_to_peg_pos",     # (3) Relative position to object
            "eef_to_peg_quat",    # (4) Relative orientation to object
        ]
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


    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""
        self.progress_buf += 1

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
            
            action = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
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


@torch.jit.script
def compute_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    reward = torch.zeros_like(progress_buf, dtype=torch.float32)

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