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

class KinovaGraspAndLift(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render) -> None:
        self.cfg = cfg
        self.graphics_device_id = graphics_device_id
        self.debug_vis = cfg["debug_vis"]

        # args
        self.max_episode_length = cfg["env"]["episodeLength"]
        self.kinova_dof_noise = self.cfg["env"]["kinovaDofNoise"]
        self.obj_position_noise = cfg["env"]["objPositionNoise"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.reward_settings = {
            "eef_dist_scale": cfg["env"]["eefDistRewardScale"],
            "grasp_threshold": cfg["env"]["graspThreshold"],
            "grasp_scale": cfg["env"]["graspRewardScale"],
            "lift_threshold": cfg["env"]["liftThreshold"],
            "lift_scale": cfg["env"]["liftRewardScale"],
            "action_scale": cfg["env"]["actionCostScale"]
        }

        # obs include: 'joint_pos': {"pos", "vel", "eef_pos", "eef_quat", "target_pos", "target_quat"}
        self.cfg["env"]["numObservations"] = 32
        # actions include: 'joint_pos': {joint_vel (6) + joint_pos (3)}
        self.cfg["env"]["numActions"] = 9
        
        # Values to be filled in at runtime
        self.states = {}                        # states used for reward calculation
        self.handles = {}                       # handles of kinova and cube
        self.actions = None                     # Current actions to be deployed
        self.n_dofs = None                      # number of dofs per env
        self._cylinder_state = None             # object state
        self._capsule_state = None              # object state
        self._cube_state = None                 # object state

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
        self.kinova_mids_pos = 0.5 * (self.kinova_upper_limits_pos + self.kinova_lower_limits_pos)
        self.default_dof_pos = self.kinova_mids_pos
        
        # set dof_props for joint_vel control
        kinova_dof_props["driveMode"][:] = gymapi.DOF_MODE_POS                              # control with joint pos, but action is vel
        self.action_scale_arm = self.kinova_upper_limits_vel[:self.n_dofs_arm]              # since actions are cliped between [-1, 1], action scales should be the upper limits
        # kinova_dof_props['stiffness'].fill(4000.0)
        # kinova_dof_props['damping'].fill(40)

        return kinova_dof_props


    def _create_envs(self) -> None:
        # load table asset
        table_size = [1.0, 1.8, 0.02]
        table_thickness = table_size[-1]
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *table_size, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.45, 0.0, 0.01)
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                               self.cfg["env"]["asset"]["assetRoot"])
        # create cylinder
        cylinder_asset_file = self.cfg["env"]["asset"]["assetFileNameCylinder"]
        print("Loading asset '%s' from '%s'" % (cylinder_asset_file, asset_root))
        cylinder_asset_options = gymapi.AssetOptions()
        cylinder_asset_options.linear_damping = 0.5
        cylinder_asset_options.angular_damping = 0.5
        cylinder_asset_options.density = 64.0
        cylinder_asset_options.max_linear_velocity = 1.0
        cylinder_asset_options.max_angular_velocity = 4.0
        cylinder_asset = self.gym.load_asset(self.sim, asset_root, cylinder_asset_file, cylinder_asset_options)
        cylinder_pose = gymapi.Transform()
        cylinder_pose.p = gymapi.Vec3(0.5, 0.0, table_thickness + 0.065)
        cylinder_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.default_cylinder_pos = to_torch([0.5, 0.0, table_thickness + 0.065], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)

        # create capsule
        capsule_size = [0.035, 0.06]
        capsule_asset_options = gymapi.AssetOptions()
        capsule_asset_options.linear_damping = 1.0
        capsule_asset_options.angular_damping = 1.0
        capsule_asset_options.max_linear_velocity = 1.0
        capsule_asset_options.max_angular_velocity = 0.5
        capsule_asset = self.gym.create_capsule(self.sim, *capsule_size, capsule_asset_options)
        capsule_pose = gymapi.Transform()
        capsule_pose.p = gymapi.Vec3(0.3, 0.5, table_thickness + 0.035)
        capsule_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.default_capsule_pos = to_torch([0.3, 0.5, table_thickness + 0.035], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)

        # create cube
        cube_size = [0.06, 0.06, 0.06]
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.linear_damping = 0.5
        cube_asset_options.angular_damping = 0.5
        cube_asset_options.density = 64.0
        cube_asset_options.max_linear_velocity = 1.0
        cube_asset_options.max_angular_velocity = 4.0
        cube_asset = self.gym.create_box(self.sim, *cube_size, cube_asset_options)
        cube_pose = gymapi.Transform()
        cube_pose.p = gymapi.Vec3(0.3, -0.5, table_thickness + 0.03)
        cube_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.default_cube_pos = to_torch([0.3, -0.5, table_thickness + 0.03], device=self.device, dtype=torch.float32).repeat(self.num_envs, 1)

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
        kinova_pose.p = gymapi.Vec3(0.0, 0.0, table_thickness)
        kinova_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # Create helper geometry used for visualization
        # Create an wireframe axis
        self.axes_geom = gymutil.AxesGeometry(0.2)
        # Create an wireframe sphere
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.sphere_geom_target = gymutil.WireframeSphereGeometry(0.07, 12, 12, sphere_pose, color=(1, 0, 0))
        self.sphere_geom_debug = gymutil.WireframeSphereGeometry(0.07, 12, 12, sphere_pose, color=(0, 1, 0))

        # set up the env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # # compute aggregate size
        num_kinova_bodies = self.gym.get_asset_rigid_body_count(self.kinova_asset)
        num_kinova_shapes = self.gym.get_asset_rigid_shape_count(self.kinova_asset)
        max_agg_bodies = num_kinova_bodies + 4     # 4 for 1 table + 3 objects
        max_agg_shapes = num_kinova_shapes + 4     # 4 for 1 table + 3 objects

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

            # add cylinder
            cylinder_handle = self.gym.create_actor(env, cylinder_asset, cylinder_pose, "cylinder", i, 2, 0)
            cylinder_color = gymapi.Vec3(0.027, 0.592, 0.902)
            self.gym.set_rigid_body_color(env, cylinder_handle, 0, gymapi.MESH_VISUAL, cylinder_color)

            # add capsule
            capsule_handle = self.gym.create_actor(env, capsule_asset, capsule_pose, "capsule", i, 4, 0)
            capsule_color = gymapi.Vec3(0.91, 0.325, 0.027)
            self.gym.set_rigid_body_color(env, capsule_handle, 0, gymapi.MESH_VISUAL, capsule_color)

            # add cube
            cube_handle = self.gym.create_actor(env, cube_asset, cube_pose, "cube", i, 8, 0)
            cube_color = gymapi.Vec3(0.561, 0.027, 0.91)
            self.gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL, cube_color)

            # aggregate ends
            self.gym.end_aggregate(env)

            self.envs.append(env)
        
        self.middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]

        self.init_data()
        

    def init_data(self):
        env_ptr = self.envs[0]
        self.table_handle = 0
        self.kinova_handle = 1
        self.cylinder_handle = 2
        self.capsule_handle = 3
        self.cube_handle = 4
        

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
        self._cylinder_state = self._root_tensor[:, self.cylinder_handle, :]
        self._capsule_state = self._root_tensor[:, self.capsule_handle, :]
        self._cube_state = self._root_tensor[:, self.cube_handle, :]
        self._obj_state_list = self._root_tensor[:, self.cylinder_handle:self.cube_handle+1, :].view(self.num_envs, 3, 13)

        # Initialize target states (num_envs, pos)
        self.target_state = torch.zeros(self.num_envs, 13, device=self.device, dtype=torch.float32)
        self.target_idx_map = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.env_ids = torch.arange(self.num_envs, device=self.device)

        # Initialize arm action controls
        self._pos_control = torch.zeros((self.num_envs, self.n_dofs), dtype=torch.float, device=self.device)
        self._arm_control = self._pos_control[:, :self.n_dofs_arm]
        self._gripper_control = self._pos_control[:, self.n_dofs_arm:]

        # Initialize indices (globle, each envs contains 3 actors (table arm cylinder capsule cube))
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                            device=self.device).view(self.num_envs, -1)


    def _reset_kinova_dofs(self, env_ids):
        reset_noise = torch.rand((len(env_ids), self.n_dofs), device=self.device, dtype=torch.float32)
        pos = torch.clamp(self.default_dof_pos.unsqueeze(0) +
                          self.kinova_dof_noise * 2.0 * (reset_noise - 0.5),
                          self.kinova_lower_limits_pos.unsqueeze(0), self.kinova_upper_limits_pos.unsqueeze(0))

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


    def _reset_object_position(self, env_ids):
        # Initialize buffer to hold sampled values
        num_resets = len(env_ids)

        dx = (torch.rand((num_resets, 3), device=self.device, dtype=torch.float32) - 0.5) * 0.2
        dy = (torch.rand((num_resets, 3), device=self.device, dtype=torch.float32) - 0.5) * 0.2

        # random quaternions (now fixed)
        # aa_rot = torch.zeros(num_resets, 3, device=self.device)
        sampled_obj_quat = torch.zeros(num_resets, 4, device=self.device, dtype=torch.float32)
        sampled_obj_quat[:, -1] = 1.0
        # aa_rot[:, 2] = 0.5 * (torch.rand(num_resets, device=self.device) - 0.5)
        # sampled_obj_quat[:, :] = quat_mul(axisangle2quat(aa_rot), sampled_obj_quat[:, :])

        # reset cylinder
        cylinder_pos = self.default_cylinder_pos[env_ids, :].clone()
        cylinder_pos[:, 0] += dx[:, 0]
        cylinder_pos[:, 1] += dy[:, 0]
        self._cylinder_state[env_ids, :3] = cylinder_pos
        self._cylinder_state[env_ids, 3:7] = sampled_obj_quat

        # reset capsule
        capsule_pos = self.default_capsule_pos[env_ids, :].clone()
        capsule_pos[:, 0] += dx[:, 1]
        capsule_pos[:, 1] += dy[:, 1]
        self._capsule_state[env_ids, :3] = capsule_pos
        self._capsule_state[env_ids, 3:7] = sampled_obj_quat

        # reset cube
        cube_pos = self.default_cube_pos[env_ids, :].clone()
        cube_pos[:, 0] += dx[:, 2]
        cube_pos[:, 1] += dy[:, 2]
        self._cube_state[env_ids, :3] = cube_pos
        self._cube_state[env_ids, 3:7] = sampled_obj_quat

        multi_env_ids_objs_int32 = self._global_indices[env_ids, self.cylinder_handle:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_tensor),
            gymtorch.unwrap_tensor(multi_env_ids_objs_int32), len(multi_env_ids_objs_int32))


    def _choose_target(self, env_ids):
        num_resets = len(env_ids)
        # sampled_object_idx = torch.randint(0, 3, (num_resets,), device=self.device)
        sampled_object_idx = torch.ones((num_resets,), device=self.device, dtype=torch.long) * 0
        self.target_idx_map[env_ids] = sampled_object_idx


    def _draw_target_marker(self):
        # draw axis and sphere at target position
        for i in range(self.num_envs):
            target_pos = self.target_state[i, :3]
            target_quat = self.target_state[i, 3:7]

            # Create random position for placing objects
            target_pose = gymapi.Transform(gymapi.Vec3(*target_pos), gymapi.Quat(*target_quat))

            gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, self.envs[i], target_pose)
            gymutil.draw_lines(self.sphere_geom_target, self.gym, self.viewer, self.envs[i], target_pose)


    def reset_all(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)
        
        # reset kinova dofs pos and vel
        self._reset_kinova_dofs(env_ids)
        
        # reset object position
        self._reset_object_position(env_ids)

        # choose target to grasp
        self._choose_target(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def _update_states(self):
        self.states.update({
            # Kinova
            "pos": self._pos[:, :],                    # 6 arm joints + 3 gripper joints
            "vel": self._vel[:, :],
            "eef_pos": self._eef_state[:, :3],
            "eef_quat": self._eef_state[:, 3:7],
            "eef_vel": self._eef_state[:, 7:],
            "eef_f1_pos": self._eef_f1_state[:, :3],
            "eef_f2_pos": self._eef_f2_state[:, :3],
            "eef_f3_pos": self._eef_f3_state[:, :3],
            # targets
            "target_pos": self.target_state[:, :3],
            "target_quat": self.target_state[:, 3:7],
            "eef_pos_to_target": self._eef_state[:, :3] - self.target_state[:, :3],
        })

    
    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # refresh target state based on selected target
        self.target_state[self.env_ids, :] = self._obj_state_list[self.env_ids, self.target_idx_map[self.env_ids], :]

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
        # u_gripper: joint pos
        u_arm, u_gripper = self.actions[:, :self.n_dofs_arm], self.actions[:, self.n_dofs_arm:]

        # control arm
        self._pos_control[:, :self.n_dofs_arm] = self._pos_control[:, :self.n_dofs_arm] + self.dt * u_arm * self.action_scale_arm

        # control gripper
        self._pos_control[:, self.n_dofs_arm:] = u_gripper * self.kinova_upper_limits_pos[self.n_dofs_arm:]

        # clip action
        self._pos_control = torch.clamp(self._pos_control, self.kinova_lower_limits_pos, self.kinova_upper_limits_pos)

        # Deploy actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_kinova_reward(
            self.reset_buf, self.progress_buf, self.states, self.reward_settings, self.max_episode_length)


    def compute_observations(self):
        self._refresh()

        obs = ["pos", "vel", "eef_pos", "eef_quat", "target_pos", "target_quat"]
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        if self.viewer and self.debug_vis:
            self.gym.clear_lines(self.viewer)
            self._draw_target_marker()
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
            
            # self.reset_all()
            # for i in range(300):
            self.step(torch.randn((self.num_envs, self.num_actions), device=self.device, dtype=torch.float))


        print("Done")

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


@torch.jit.script
def compute_kinova_reward(reset_buf, progress_buf, states, reward_settings, max_episode_length):
    # type: (Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # reward for gripper close to the object
    d = torch.norm(states['eef_pos_to_target'], dim=-1)
    # print(f"target pos: {states['target_pos']}, df1 pos: {states['eef_f1_pos']}")
    # print(f"d: {d} \n df1: {d_f1} \n df2: {d_f2} \n df3: {d_f3}")
    eef_target_dist_reward = 1 - torch.tanh(10.0 * d)
    reward = eef_target_dist_reward * reward_settings["eef_dist_scale"]

    # reward for grasping the object
    # print(d)
    reached = d < reward_settings["grasp_threshold"]
    # print(reached)
    grasping_reward = reached * torch.mean(states["pos"][:, 6:], dim=-1)
    reward +=  grasping_reward * reward_settings["grasp_scale"]

    # reward for lifting the object
    lift_distance = torch.abs(states['target_pos'][:, -1] - reward_settings["lift_threshold"])
    lift_reward = 1 - torch.tanh(10.0 * lift_distance)
    reward += reached * lift_reward * reward_settings["lift_scale"]

    # # penalty for action
    # reward -= states["action_cost"] * reward_settings["action_scale"]

    # Compute resets (reset if maximum episode length)
    reset_buf = torch.where((progress_buf >= max_episode_length - 1), torch.ones_like(reset_buf), reset_buf)

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