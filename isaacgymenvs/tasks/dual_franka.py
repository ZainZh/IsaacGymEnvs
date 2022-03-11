import math
import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask

class DualFranka(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.num_agents = self.cfg["env"]["numAgents"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "y"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        # prop dimensions
        self.prop_width = 0.08
        self.prop_height = 0.08
        self.prop_length = 0.08
        self.prop_spacing = 0.09

        num_obs = 46
        num_acts = 18

        self.cfg["env"]["numObservations"] = 46
        self.cfg["env"]["numActions"] = 18
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                                               device=self.device)
        self.franka_default_dof_pos_1 = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                                               device=self.device)


        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_state_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.franka_dof_pos_1 = self.franka_dof_state_1[..., 0]
        self.franka_dof_vel_1 = self.franka_dof_state_1[..., 1]
        self.table_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.table_dof_pos = self.table_dof_state[..., 0]
        self.table_dof_vel = self.table_dof_state[..., 1]
        self.cup_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_cup_dofs:]
        self.cup_dof_pos = self.cup_dof_state[..., 0]
        self.cup_dof_vel = self.cup_dof_state[..., 1]
        self.spoon_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_spoon_dofs:]
        self.spoon_dof_pos = self.spoon_dof_state[..., 0]
        self.spoon_dof_vel = self.spoon_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 2:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.franka_dof_targets_1 = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.franka_dof_targets_all= torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Y
        self.sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
        self.sim=super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)
    def _create_envs(self,num_envs,spacing,num_per_row):

        env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        cup_asset_file = 'urdf/cup/urdf/cup.urdf'
        spoon_asset_file = 'urdf/spoon/urdf/spoon.urdf'
        shelf_asset_file = 'urdf/shelf/urdf/shelf.urdf'

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.armature = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
        franka_asset_1 = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)
       # load table asset
        cup_asset_options = gymapi.AssetOptions()
        table_dims = gymapi.Vec3(2.4, 2.0, 3.0)
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        cup_asset = self.gym.load_asset(self.sim, asset_root, cup_asset_file, cup_asset_options)
        cup_asset_options.fix_base_link = True
        shelf_asset = self.gym.load_asset(self.sim, asset_root, shelf_asset_file, cup_asset_options)
        #load cup and spoon

        asset_options.fix_base_link=False
        asset_options.disable_gravity=False

       # cup_asset_options.fix_base_link = False

        spoon_asset = self.gym.load_asset(self.sim, asset_root, spoon_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

        self.num_franka_bodies=self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs=self.gym.get_asset_dof_count(franka_asset)

        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_dofs = self.gym.get_asset_dof_count(table_asset)

        self.num_franka_bodies_1 = self.gym.get_asset_rigid_body_count(franka_asset_1)
        self.num_franka_dofs_1 = self.gym.get_asset_dof_count(franka_asset_1)

        self.num_cup_bodies=self.gym.get_asset_rigid_body_count(cup_asset)
        self.num_cup_dofs=self.gym.get_asset_dof_count(cup_asset)

        self.num_spoon_bodies=self.gym.get_asset_rigid_body_count(spoon_asset)
        self.num_spoon_dofs=self.gym.get_asset_dof_count(spoon_asset)

        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)
        print("num franka bodies: ", self.num_franka_bodies_1)
        print("num franka dofs_1: ", self.num_franka_dofs_1)
        print("num cup bodies: ", self.num_cup_bodies)
        print("num cup dofs: ", self.num_cup_dofs)
        print("num spoon bodies: ", self.num_cup_bodies)
        print("num spoon dofs: ", self.num_cup_dofs)

        # set franka dof properties
        franka_dof_props=self.gym.get_asset_dof_properties(franka_asset)
        franka_dof_props_1= self.gym.get_asset_dof_properties(franka_asset_1)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i]=gymapi.DOF_MODE_POS
            franka_dof_props_1['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
                franka_dof_props_1['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props_1['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0
                franka_dof_props_1['stiffness'][i] = 7000.0
                franka_dof_props_1['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits,device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits,device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200
        franka_dof_props_1['effort'][7] = 200
        franka_dof_props_1['effort'][8] = 200


        #create pose
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0, 0, 0)

        pose = gymapi.Transform()

        pose.p.x = table_pose.p.x - 0.3
        pose.p.y = 1
        pose.p.z = 0.29
        pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        pose_1 = gymapi.Transform()
        pose_1.p.x = table_pose.p.x - 0.3
        pose_1.p.y = 1
        pose_1.p.z = -0.29

        pose_1.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        cup_pose = gymapi.Transform()
        cup_pose.p.x = table_pose.p.x + 0.3
        cup_pose.p.y = 1.0
        cup_pose.p.z = 0.29
        cup_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        spoon_pose = gymapi.Transform()
        spoon_pose.p.x = table_pose.p.x + 0.25
        spoon_pose.p.y = 1.107
        spoon_pose.p.z = -0.29
        spoon_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        shelf_pose = gymapi.Transform()
        shelf_pose.p.x = table_pose.p.x + 0.3
        shelf_pose.p.y = 1
        shelf_pose.p.z = -0.29
        shelf_pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

        #compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_cup_bodies = self.gym.get_asset_rigid_body_count(cup_asset)
        num_cup_shapes = self.gym.get_asset_rigid_shape_count(cup_asset)
        num_spoon_bodies = self.gym.get_asset_rigid_body_count(spoon_asset)
        num_spoon_shapes = self.gym.get_asset_rigid_shape_count(spoon_asset)

        max_agg_bodies=2*num_franka_bodies+num_spoon_bodies+num_table_bodies+num_cup_bodies
        max_agg_shapes=2+num_franka_shapes+num_spoon_shapes+num_cup_shapes+num_table_shapes

        # Point camera at environments
        # cam_pos = gymapi.Vec3(-4.0, 4.0, -1.0)
        # cam_target = gymapi.Vec3(0.0, 2.0, 1.0)
        # viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        # self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        self.frankas = []
        self.frankas_1= []
        self.table= []
        self.spoon= []
        self.cup =[]
        #prop means spoon add cup
        self.default_spoon_states = []
        self.default_cup_states=[]
        self.prop_start = []
        self.envs = []

        for i in range(self.num_envs):

            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            franka_actor_1 = self.gym.create_actor(env_ptr, franka_asset_1, pose_1, "franka1", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor_1, franka_dof_props_1)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, pose, "franka", i, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)



            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            cup_actor= self.gym.create_actor(env_ptr, cup_asset, cup_pose, "cup", i, 0)
            spoon_actor = self.gym.create_actor(env_ptr, spoon_asset, spoon_pose, "spoon", i, 0)
            shelf = self.gym.create_actor(env_ptr, shelf_asset, shelf_pose, "shelf", i, 0)
            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))
                cup_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cup_actor, "cup_handle")

            # deflaut??????????
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            self.frankas_1.append(franka_actor_1)
            self.table.append(table_actor)
            self.cup.append(cup_actor)
            self.spoon.append(spoon_actor)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        self.hand_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_link7")
        self.table_handle = self.gym.find_actor_rigid_body_handle(env_ptr, table_actor, "box")
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.lfinger_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_leftfinger")
        self.rfinger_handle_1 = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor_1, "panda_rightfinger")
        self.cup_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cup_actor, "base_link")
        self.spoon_handle = self.gym.find_actor_rigid_body_handle(env_ptr, spoon_actor, "base_link")

        self.init_data()
    def init_data(self):
        #franka
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (lfinger_pose.p + rfinger_pose.p) * 0.5
        finger_pose.r = lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))

        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        #franka_1

        hand_1= self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_link7")
        lfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_leftfinger")
        rfinger_1 = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas_1[0], "panda_rightfinger")

        hand_pose_1 = self.gym.get_rigid_transform(self.envs[0], hand_1)
        lfinger_pose_1 = self.gym.get_rigid_transform(self.envs[0], lfinger_1)
        rfinger_pose_1 = self.gym.get_rigid_transform(self.envs[0], rfinger_1)

        finger_pose_1 = gymapi.Transform()
        finger_pose_1.p = (lfinger_pose_1.p + rfinger_pose_1.p) * 0.5
        finger_pose_1.r = lfinger_pose_1.r

        hand_pose_inv_1 = hand_pose_1.inverse()
        grasp_pose_axis_1 = 1
        franka_local_grasp_pose_1 = hand_pose_inv_1 * finger_pose_1
        franka_local_grasp_pose_1.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis_1))

        self.franka_local_grasp_pos_1 = to_torch([franka_local_grasp_pose_1.p.x, franka_local_grasp_pose_1.p.y,
                                                franka_local_grasp_pose_1.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot_1 = to_torch([franka_local_grasp_pose_1.r.x, franka_local_grasp_pose_1.r.y,
                                                franka_local_grasp_pose_1.r.z, franka_local_grasp_pose_1.r.w], device=self.device).repeat((self.num_envs, 1))


       #get the cup grasp pose (should add 150mm in y axis from origin)
        cup_local_grasp_pose = gymapi.Transform()
        cup_local_grasp_pose.p.x = 0.3
        cup_local_grasp_pose.p.y = 1.075
        cup_local_grasp_pose.p.z = 0.29
        cup_local_grasp_pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

        self.cup_local_grasp_pos = to_torch([cup_local_grasp_pose.p.x, cup_local_grasp_pose.p.y,
                                                cup_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.cup_local_grasp_rot = to_torch([cup_local_grasp_pose.r.x, cup_local_grasp_pose.r.y,
                                                cup_local_grasp_pose.r.z, cup_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        spoon_local_grasp_pose = gymapi.Transform()
        spoon_local_grasp_pose.p.x =  0.4
        spoon_local_grasp_pose.p.y = 1.107
        spoon_local_grasp_pose.p.z = -0.5
        spoon_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)

        self.spoon_local_grasp_pos = to_torch([spoon_local_grasp_pose.p.x, spoon_local_grasp_pose.p.y,
                                                spoon_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.spoon_local_grasp_rot = to_torch([spoon_local_grasp_pose.r.x, spoon_local_grasp_pose.r.y,
                                                spoon_local_grasp_pose.r.z, spoon_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.cup_inward_axis=to_torch([1,0,0],device=self.device).repeat((self.num_envs,1))
        self.gripper_up_axis = to_torch([1,0,0], device=self.device).repeat((self.num_envs, 1))
        self.cup_up_axis=to_torch([0,1,0], device=self.device).repeat((self.num_envs, 1))

        self.gripper_forward_axis_1 = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.spoon_inward_axis = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis_1 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.spoon_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw

        self.franka_grasp_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_grasp_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        self.franka_grasp_rot_1[..., -1] = 1  # xyzw

        self.cup_grasp_pos = torch.zeros_like(self.cup_local_grasp_pos)
        self.cup_grasp_rot = torch.zeros_like(self.cup_local_grasp_rot)
        self.cup_grasp_rot[..., -1] = 1

        self.spoon_grasp_pos = torch.zeros_like(self.spoon_local_grasp_pos)
        self.spoon_grasp_rot = torch.zeros_like(self.spoon_local_grasp_rot)
        self.spoon_grasp_rot[..., -1] = 1

        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        #
        self.franka_lfinger_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_rfinger_pos_1 = torch.zeros_like(self.franka_local_grasp_pos_1)
        self.franka_lfinger_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        self.franka_rfinger_rot_1 = torch.zeros_like(self.franka_local_grasp_rot_1)
        ##########################################################################3
        #important

    def compute_reward(self):
        self.rew_buf[:],self.reset_buf[:]=compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.cup_dof_pos,self.franka_grasp_pos, self.cup_grasp_pos,self.franka_grasp_rot,
            self.spoon_dof_pos,self.franka_grasp_pos_1, self.spoon_grasp_pos, self.franka_grasp_rot_1,
            self.cup_grasp_rot,self.spoon_grasp_rot,
            self.cup_inward_axis,self.cup_up_axis,self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.spoon_inward_axis,self.spoon_up_axis,self.franka_lfinger_pos_1, self.franka_rfinger_pos_1,
            self.gripper_forward_axis,self.gripper_up_axis,
            self.gripper_forward_axis_1,self.gripper_up_axis_1,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length)

        #important
###########################################################
    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        hand_pos_1 = self.rigid_body_states[:, self.hand_handle_1][:, 0:3]
        hand_rot_1 = self.rigid_body_states[:, self.hand_handle_1][:, 3:7]

        cup_pos = self.rigid_body_states[:, self.cup_handle][:, 0:3]
        cup_rot = self.rigid_body_states[:, self.cup_handle][:, 3:7]

        spoon_pos = self.rigid_body_states[:, self.spoon_handle][:, 0:3]
        spoon_rot = self.rigid_body_states[:, self.spoon_handle][:, 3:7]
        #franka with cup and franka1 with spoon
        # exist some question
        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.cup_grasp_rot[:], self.cup_grasp_pos[:],\
        self.franka_grasp_rot_1[:], self.franka_grasp_pos_1[:], self.spoon_grasp_rot[:], self.spoon_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     cup_rot, cup_pos, self.cup_local_grasp_rot, self.cup_local_grasp_pos,
                                     hand_rot_1, hand_pos_1, self.franka_local_grasp_rot_1, self.franka_local_grasp_pos_1,
                                    spoon_rot, spoon_pos, self.spoon_local_grasp_rot, self.spoon_local_grasp_pos
                                     )


        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        self.franka_lfinger_pos_1 = self.rigid_body_states[:, self.lfinger_handle_1][:, 0:3]
        self.franka_rfinger_pos_1 = self.rigid_body_states[:, self.rfinger_handle_1][:, 0:3]
        self.franka_lfinger_rot_1 = self.rigid_body_states[:, self.lfinger_handle_1][:, 3:7]
        self.franka_rfinger_rot_1 = self.rigid_body_states[:, self.rfinger_handle_1][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        dof_pos_scaled_1 = (2.0 * (self.franka_dof_pos_1 - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)
        to_target = self.cup_grasp_pos - self.franka_grasp_pos
        to_target_1 = self.spoon_grasp_pos- self.franka_grasp_pos_1
        self.obs_buf = torch.cat((dof_pos_scaled,dof_pos_scaled_1,
                                  self.franka_dof_vel * self.dof_vel_scale, to_target,
                                  self.cup_dof_pos[:, 3].unsqueeze(-1), self.cup_dof_vel[:, 3].unsqueeze(-1),self.franka_dof_vel_1 * self.dof_vel_scale, to_target_1,
                                  self.spoon_dof_pos[:, 3].unsqueeze(-1), self.spoon_dof_vel[:, 3].unsqueeze(-1)), dim=-1)


        return self.obs_buf
    def reset_idx(self,env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos

        # reset franka1
        pos_1 = tensor_clamp(
            self.franka_default_dof_pos_1.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs_1), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos_1[env_ids, :] = pos_1
        self.franka_dof_vel_1[env_ids, :] = torch.zeros_like(self.franka_dof_vel_1[env_ids])
        self.franka_dof_targets_1[env_ids, :self.num_franka_dofs] = pos_1

        # reset cup
        self.cup_dof_state[env_ids, :] = torch.zeros_like(self.cup_dof_state[env_ids])

        # reset spoon
        self.spoon_dof_state[env_ids, :] = torch.zeros_like(self.spoon_dof_state[env_ids])

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets_1),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self,actions):
        self.actions = actions.clone().to(self.device)
        # print(self.actions,"\n")


        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions[:,0:9] * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)

        targets_1 = self.franka_dof_targets_1[:,:self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions[:,9:18] * self.action_scale
        self.franka_dof_targets_1[:, :self.num_franka_dofs] = tensor_clamp(
            targets_1, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_targets_all=torch.cat((self.franka_dof_targets[:,0:9],self.franka_dof_targets_1[:,0:9]),1)
        # print("f_d_t=",self.franka_dof_targets_all, "\n")

        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets_all))


    def post_physics_step(self):
        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()
        self.compute_reward()
        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)


            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([1, 0, 0],device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([0, 1, 0],device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos_1[i] + quat_apply(self.franka_grasp_rot_1[i], to_torch([0, 0, 1],device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]],[0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]],[0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]],[0.1, 0.1, 0.85])

                px = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i], to_torch([1, 0, 0],device=self.device) * 0.2)).cpu().numpy()
                py = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i], to_torch([0, 1, 0],device=self.device) * 0.2)).cpu().numpy()
                pz = (self.cup_grasp_pos[i] + quat_apply(self.cup_grasp_rot[i], to_torch([0, 0, 1],device=self.device) * 0.2)).cpu().numpy()

                p0 = self.cup_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.spoon_grasp_pos[i] + quat_apply(self.spoon_grasp_rot[i],to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.spoon_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos_1[i] + quat_apply(self.franka_lfinger_rot_1[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos_1[i] + quat_apply(self.franka_rfinger_rot_1[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos_1[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                #####################################################################
                ###=========================jit functions=========================###
                #####################################################################
@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions,cup_dof_pos,franka_grasp_pos, cup_grasp_pos, franka_grasp_rot, spoon_dof_pos,franka_grasp_pos_1, spoon_grasp_pos, franka_grasp_rot_1,cup_grasp_rot,spoon_grasp_rot,
    cup_inward_axis,cup_up_axis,franka_lfinger_pos, franka_rfinger_pos,
    spoon_inward_axis,spoon_up_axis,franka_lfinger_pos_1, franka_rfinger_pos_1,
    gripper_forward_axis,gripper_up_axis,
    gripper_forward_axis_1,gripper_up_axis_1,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor,Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor, Tensor, Tensor, Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,Tensor,int, float, float, float, float, float, float, float) -> Tuple[Tensor,Tensor]
    d = torch.norm(franka_grasp_pos - cup_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    d_1 = torch.norm(franka_grasp_pos_1 - spoon_grasp_pos, p=2, dim=-1)
    dist_reward_1 = 1.0 / (1.0 + d_1 ** 2)
    dist_reward_1 *= dist_reward_1
    dist_reward_1 = torch.where(d_1 <= 0.02, dist_reward_1 * 2, dist_reward_1)

    axis1 = tf_vector(franka_grasp_rot, gripper_forward_axis)
    axis2 = tf_vector(cup_grasp_rot, cup_inward_axis)
    axis3 = tf_vector(franka_grasp_rot, gripper_up_axis)
    axis4 = tf_vector(cup_grasp_rot, cup_up_axis)

    axis1_1 = tf_vector(franka_grasp_rot_1, gripper_forward_axis_1)
    axis2_1 = tf_vector(spoon_grasp_rot, spoon_inward_axis)
    axis3_1 = tf_vector(franka_grasp_rot_1, gripper_up_axis_1)
    axis4_1 = tf_vector(spoon_grasp_rot, spoon_up_axis)

    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2 = torch.bmm(axis3.view(num_envs, 1, 3), axis4.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)

    dot1_1 = torch.bmm(axis1_1.view(num_envs, 1, 3), axis2_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    dot2_1 = torch.bmm(axis3_1.view(num_envs, 1, 3), axis4_1.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)
    # reward for matching the orientation of the hand to the cup(fingers wrapped)
    rot_reward = 0.5 * (torch.sign(dot1) * dot1 ** 2 + torch.sign(dot2) * dot2 ** 2)
    rot_reward_1 = 0.5 * (torch.sign(dot1_1) * dot1_1 ** 2 + torch.sign(dot2_1) * dot2_1 ** 2)
    # bonus if left finger is above the drawer handle and right below
    around_handle_reward = torch.zeros_like(rot_reward)
    around_handle_reward = torch.where(franka_lfinger_pos[:, 2] > cup_grasp_pos[:, 2],
                                       torch.where(franka_rfinger_pos[:, 2] < cup_grasp_pos[:, 2],
                                                   around_handle_reward + 0.5, around_handle_reward), around_handle_reward)
    around_handle_reward_1 = torch.zeros_like(rot_reward_1)
    around_handle_reward_1 = torch.where(franka_lfinger_pos_1[:, 2] > spoon_grasp_pos[:, 2],
                                       torch.where(franka_rfinger_pos_1[:, 2] < spoon_grasp_pos[:, 2],
                                                   around_handle_reward_1 + 0.5, around_handle_reward_1),
                                       around_handle_reward_1)

    # reward for distance of each finger from the drawer
    finger_dist_reward = torch.zeros_like(rot_reward)
    lfinger_dist = torch.abs(franka_lfinger_pos[:, 2] - cup_grasp_pos[:, 2]+0.04)
    rfinger_dist = torch.abs(franka_rfinger_pos[:, 2] - cup_grasp_pos[:, 2]+0.04)
    finger_dist_reward = torch.where(franka_lfinger_pos[:, 2] > cup_grasp_pos[:, 2],
                                     torch.where(franka_rfinger_pos[:, 2] < cup_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist) + (0.04 - rfinger_dist), finger_dist_reward), finger_dist_reward)

    # reward for distance of each finger from the drawer
    finger_dist_reward_1 = torch.zeros_like(rot_reward_1)
    lfinger_dist_1 = torch.abs(franka_lfinger_pos_1[:, 2] - spoon_grasp_pos[:, 2]+0.04)
    rfinger_dist_1 = torch.abs(franka_rfinger_pos_1[:, 2] - spoon_grasp_pos[:, 2]+0.04)
    finger_dist_reward_1 = torch.where(franka_lfinger_pos_1[:, 2] > spoon_grasp_pos[:, 2],
                                     torch.where(franka_rfinger_pos_1[:, 2] < spoon_grasp_pos[:, 2],
                                                 (0.04 - lfinger_dist_1) + (0.04 - rfinger_dist_1), finger_dist_reward_1), finger_dist_reward)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards=dist_reward_scale*dist_reward+rot_reward_scale*rot_reward\
            +around_handle_reward_scale*around_handle_reward\
            +finger_dist_reward_scale*finger_dist_reward \
            +dist_reward_scale * dist_reward_1 + rot_reward_scale * rot_reward_1 \
            + around_handle_reward_scale * around_handle_reward_1 \
            + finger_dist_reward_scale * finger_dist_reward_1 \
            -action_penalty*action_penalty_scale

    #bonus for take up the cup properly
    rewards = torch.where(cup_dof_pos[:,1] > 1.01,rewards+0.5,rewards)
    rewards = torch.where(cup_dof_pos[:,1] > 1.2, rewards + around_handle_reward, rewards)
    rewards = torch.where(cup_dof_pos[:,1] > 1.3, rewards + (2*around_handle_reward), rewards)

    #bonus for take up the cup properly
    rewards= torch.where(spoon_dof_pos[:,1] > 1.12,rewards+0.5,rewards)
    rewards = torch.where(spoon_dof_pos[:,1] > 1.2, rewards + around_handle_reward_1, rewards)
    rewards = torch.where(spoon_dof_pos[:,1] > 1.3, rewards + (2*around_handle_reward_1), rewards)

    #reset if cup and spoon is taken up (max) or max length reached
    reset_buf = torch.where(cup_dof_pos[:,1]>0.3,(torch.where(spoon_dof_pos[:, 1] > 0.3, torch.ones_like(reset_buf), reset_buf)),reset_buf)
    reset_buf =torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)


    return rewards,reset_buf

##### franka1
@torch.jit.script
def compute_grasp_transforms(hand_rot,hand_pos,franka_local_grasp_rot,franka_local_grasp_pos,
                             cup_rot,cup_pos,cup_local_grasp_rot,cup_local_grasp_pos,hand_rot_1,hand_pos_1,franka_local_grasp_rot_1,franka_local_grasp_pos_1,
                             spoon_rot,spoon_pos,spoon_local_grasp_rot,spoon_local_grasp_pos
                             ):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor]

        global_franka_rot,global_franka_pos=tf_combine(
            hand_rot,hand_pos,franka_local_grasp_rot,franka_local_grasp_pos)
        global_cup_rot,global_cup_pos=tf_combine(cup_rot,cup_pos,cup_local_grasp_rot,cup_local_grasp_pos)

        global_franka_rot_1, global_franka_pos_1 = tf_combine(
            hand_rot_1, hand_pos_1, franka_local_grasp_rot_1, franka_local_grasp_pos_1)
        global_spoon_rot, global_spoon_pos = tf_combine(spoon_rot, spoon_pos, spoon_local_grasp_rot, spoon_local_grasp_pos)

        return global_franka_rot,global_franka_pos,global_cup_rot,global_cup_pos,global_franka_rot_1,global_franka_pos_1,global_spoon_rot,global_spoon_pos