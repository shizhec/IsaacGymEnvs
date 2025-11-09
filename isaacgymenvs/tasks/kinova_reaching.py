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

class KinovaReaching(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render) -> None:
        self.cfg = cfg
        self.graphics_device_id = graphics_device_id
        self.debug_vis = cfg["debug_vis"]

        # args
        self.test = cfg["test"]
        self.random_reset = getattr(cfg, 'random_reset', True)
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.kinova_dof_noise = self.cfg["env"]["kinovaDofNoise"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.reward_settings = {
            "dist_scale": cfg["env"]["distRewardScale"],
        }

        # Values to be filled in at runtime
        self.states = {}                        # states used for reward calculation
        self.handles = {}                       # handles of kinova and cube
        self.actions = None                     # Current actions to be deployed
        self.n_dofs = None                      # number of dofs per env (will be set after loading URDF)

        # dimensions - will be set dynamically after loading URDF in create_sim
        # obs include: 'joint_pos': {dof_pos (n_dofs) + dof_vel(n_dofs) + eef_pos(3) + target_pos(3)}
        # actions include: 'joint_pos': {joint_vel (n_dofs)}
        # These are temporarily set and will be updated in _create_envs
        self.cfg["env"]["numObservations"] = 18  # Default for 6-DOF, updated dynamically
        self.cfg["env"]["numActions"] = 6        # Default for 6-DOF, updated dynamically

        # Tensor placeholders
        self._root_tensor = None                # State of root body            (n_envs, 13) [3 position floats, 4 quaternion floats(orientation), 3 linear velocity floats, 3 angular veloctity floats]
        self._dof_tensor = None                 # States of all dofs            (n_dofs, 2) [Position, Velocity]
        self._rb_tensor = None                  # State of all rigid bodies     (n_envs, n_bodies, 13)
        self._pos = None                        # Joint positions               (n_envs, n_dof)
        self._vel = None                        # Joint velocities              (n_envs, n_dof)
        self._eef_state = None                  # end effector state (at grasping point) (n_envs, 13)
        self._arm_control = None                # Tensor buffer for controlling arm
        self._global_indices = None             # Unique indices corresponding to all envs in flattened array

        super().__init__(config=cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, 
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        

        # Success tracking parameters
        self.success_threshold = 0.10  # Distance threshold for success (2cm)
        self.success_duration = 10     # Stay at target for 30 steps
        self.success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self._config_camera()

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

        # Compute Initial Observations
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
        cam_pos = gymapi.Vec3(12.0, 10.5, 3.0)
        cam_target = gymapi.Vec3(4.5, 2.0, -5.0)
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
        self.kinova_lower_limits_pos = to_torch(kinova_dof_props['lower'], device=self.device, dtype=torch.float32)
        self.kinova_upper_limits_pos = to_torch(kinova_dof_props['upper'], device=self.device, dtype=torch.float32)
        self.kinova_lower_limits_vel = to_torch(-kinova_dof_props['velocity'], device=self.device, dtype=torch.float32)
        self.kinova_upper_limits_vel = to_torch(kinova_dof_props['velocity'], device=self.device, dtype=torch.float32)

        # set default joint_pos to home pose
        self.default_dof_pos = to_torch([0.000, 3.8082, 4.2944, 2.4488, 1.7400, 0.9989], device=self.device, dtype=torch.float32)
        
        # set dof_props for joint_vel control
        kinova_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS          # control with joint pos, but action is vel
        self.action_scale = self.kinova_upper_limits_vel                # since actions are cliped between [-1, 1], action scales should be the upper limits
        # kinova_dof_props['stiffness'].fill(4000.0)
        # kinova_dof_props['damping'].fill(40)

        return kinova_dof_props


    def _create_envs(self) -> None:
        # load table asset
        table_size = [1.1, 1.8, 0.02]
        table_thickness = table_size[-1]
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *table_size, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.5, 0.0, 0.01)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # load kinova assets
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               self.cfg["env"]["asset"]["assetRoot"])
        asset_file = self.cfg["env"]["asset"]["assetFileNameKinova"]

        kinova_asset_options = gymapi.AssetOptions()
        kinova_asset_options.fix_base_link = True
        kinova_asset_options.collapse_fixed_joints = False
        kinova_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # kinova_asset_options.armature = 0.01
        kinova_asset_options.thickness = 0.001
        kinova_asset_options.use_mesh_materials = True
        kinova_asset_options.disable_gravity = True

        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        self.kinova_asset = self.gym.load_asset(self.sim, asset_root, asset_file, kinova_asset_options)

        kinova_dof_props = self._config_kinova_dofs_props()

        # Update observation and action dimensions based on actual DOF count
        # obs = [dof_pos(n_dofs), dof_vel(n_dofs), eef_pos(3), target_pos(3)]
        self.cfg["env"]["numObservations"] = 2 * self.n_dofs + 6
        self.cfg["env"]["numActions"] = self.n_dofs
        print(f"Updated numObservations to {self.cfg['env']['numObservations']} (for {self.n_dofs} DOFs)")
        print(f"Updated numActions to {self.cfg['env']['numActions']}")

        kinova_pose = gymapi.Transform()
        kinova_pose.p = gymapi.Vec3(0.0, 0.0, table_thickness)
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
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # # compute aggregate size
        num_kinova_bodies = self.gym.get_asset_rigid_body_count(self.kinova_asset)
        num_kinova_shapes = self.gym.get_asset_rigid_shape_count(self.kinova_asset)
        max_agg_bodies = num_kinova_bodies + 1     # 1 for table
        max_agg_shapes = num_kinova_shapes + 1     # 1 for table

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

            # add kinova
            kinova_handle = self.gym.create_actor(env, self.kinova_asset, kinova_pose, "kinova", i, 1, 0)

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
        self.kinova_handle = 1

        self.handles = {
            # Kinova
            "end_effector": self.gym.find_actor_rigid_body_handle(env_ptr, self.kinova_handle, "j2n6s300_end_effector"),
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
        self._eef_state = self._rb_tensor[:, self.handles["end_effector"], :]               # (n_envs, 13)

        # Initialize target position (num_envs, pos)
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)

        # Initialize arm action controls
        self._arm_control = torch.zeros((self.num_envs, self.n_dofs), dtype=torch.float32, device=self.device)

        # Initialize indices (globle, each envs contains 3 actors (table cube arm))
        self._global_indices = torch.arange(self.num_envs * 2, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)


    def _reset_kinova_dofs(self, env_ids, random=True):
        if random:
            reset_noise = torch.rand((len(env_ids), self.n_dofs), device=self.device, dtype=torch.float32)
            pos = torch.clamp(self.default_dof_pos.unsqueeze(0) +
                            self.kinova_dof_noise * 2.0 * (reset_noise - 0.5),
                            self.kinova_lower_limits_pos.unsqueeze(0), self.kinova_upper_limits_pos.unsqueeze(0))
        else:
            # When random=False, use default positions without noise
            pos = self.default_dof_pos.unsqueeze(0).repeat(len(env_ids), 1)

        # refresh pos and vel
        self._pos[env_ids, :] = pos
        self._vel[env_ids, :] = torch.zeros_like(self._vel[env_ids])

        # reset vel control
        self._arm_control[env_ids, :] = pos

        # reset kinova
        multi_env_ids_int32 = self._global_indices[env_ids, self.kinova_handle].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_tensor),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))


    def _reset_target_position(self, env_ids, random=True):
        self.gym.clear_lines(self.viewer)
        num_resets = len(env_ids)

        if random:
            # random sample angle
            theta = (torch.rand(num_resets, 1, device=self.device, dtype=torch.float32) - 0.5) * torch.pi / 3

            # random sample radius within a range(min_rad, max_rad)
            radius = torch.rand(num_resets, 1, device=self.device, dtype=torch.float32) * (0.8 - 0.5) + 0.5

            # get x and y
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)

            # sample random z
            z = torch.rand(num_resets, 1, device=self.device, dtype=torch.float32) * 0.4 + 0.4

            sampled_target_pos = torch.cat([x, y, z], dim=-1)
        else:
            theta = torch.tensor([0.25], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            radius = torch.tensor([0.6], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)
            z = torch.tensor([0.6], device=self.device, dtype=torch.float32).repeat(num_resets, 1)
            sampled_target_pos = torch.cat([x, y, z], dim=-1)
        
        # update sample target pos
        self.target_pos[env_ids, :] = sampled_target_pos

        # draw axis and sphere at target position
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
        # If env_ids is None, we reset all the envs
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # reset kinova dofs pos and vel
        self._reset_kinova_dofs(env_ids, random=self.random_reset)
        
        # reset target position
        self._reset_target_position(env_ids, random=self.random_reset)

        if self.test:
            # Calculate success rate as the ratio of successful environments to total environments
            success_rate = torch.sum(self.success_flag).item() / self.num_envs
            print(f"Success rate: {success_rate:.2f}")

            # Reset success tracking for these environments
            self.success_counter[env_ids] = 0
            self.success_flag[env_ids] = False

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _update_states(self):
        self.states.update({
            # Kinova
            "pos": self._pos[:, :],                    # 6 arm joint
            "vel": self._vel[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            # Targets
            "target_pos": self.target_pos,
            "eef_pos_to_target": self._eef_state[:, :3] - self.target_pos
        })

    
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

        # control arm
        self._arm_control = self._arm_control + self.dt * self.actions * self.action_scale
        self._arm_control = torch.clamp(self._arm_control, self.kinova_lower_limits_pos, self.kinova_upper_limits_pos)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._arm_control))


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_kinova_reward(
            self.reset_buf, self.progress_buf, self.states, self.reward_settings, self.max_episode_length)


    def compute_observations(self):
        self._refresh()

        obs = ["pos", "vel", "eef_pos", "target_pos"]
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
        dist_to_target = torch.norm(self.states['eef_pos_to_target'], dim=-1)
        
        # Check which environments have the end-effector close enough to the target
        at_target = dist_to_target < self.success_threshold
        
        # Increment counters for environments where arm is at target
        self.success_counter = torch.where(at_target, self.success_counter + 1, torch.zeros_like(self.success_counter))
        
        # Set success flag for environments where counter exceeds success duration
        self.success_flag = self.success_flag | self.success_counter >= self.success_duration                           


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
            t = self.gym.get_sim_time(self.sim)

            self.step(torch.tensor([[-1, 0.1, 0.1, 0.1, 0.0, 0.0]], device=self.device, dtype=torch.float))

        print("Done")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


@torch.jit.script
def compute_kinova_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    d = torch.norm(states['target_pos'] - states['eef_pos'], dim=-1)
    dist_reward = 1 - torch.tanh(5.0 * d)


    # compute final reward
    reward = dist_reward * reward_settings["dist_scale"]

    # Compute resets (reset if maximum episode length)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

    return reward, reset_buf
