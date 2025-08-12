
# isaac gym库存在问题，一定要先import pinocchio再import isaacgym
import pinocchio
import os
import pink
from pink.tasks import FrameTask
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import panda_py
from panda_py import libfranka
from pathlib import Path
import time
import yaml
import math
import numpy as np
import torch
import os 
import sys
from pathlib import Path
import time
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 添加父目录到系统路径，用于模块导入
from TeleVision import OpenTeleVision
from teleop.open_television.franka.tv_wrapper import TeleVisionWrapper

# from teleop.open_television.television import TeleVision as OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations
from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from pytransform3d import rotations
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
from teleop.robot_control.robot_arm_ik import Franka_Panda_ARMIK
from teleop.open_television.franka.constants import *
from teleop.utils.mat_tool import mat_update, fast_mat_inv
from teleop.open_television.teleop_utils import *
import numpy as np


import numpy as np

def rotate_about_y(R_orig, angle_deg):
    """
    将一个3x3旋转矩阵绕Y轴再旋转指定角度（单位为度）

    参数：
        R_orig: 原始3x3旋转矩阵 (numpy.ndarray)
        angle_deg: 绕Y轴旋转的角度（单位为度，正为逆时针，负为顺时针）

    返回：
        R_new: 旋转后的3x3旋转矩阵
    """
    angle_rad = np.radians(angle_deg)
    Ry = np.array([
        [ np.cos(angle_rad), 0, np.sin(angle_rad)],
        [ 0,                1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    R_new = Ry @ R_orig
    return R_new
import numpy as np

def rotate_about_z(R_orig, angle_deg):
    """
    将一个3x3旋转矩阵绕Z轴再旋转指定角度（单位为度）

    参数：
        R_orig: 原始3x3旋转矩阵 (numpy.ndarray)
        angle_deg: 绕Z轴旋转的角度（单位为度，正为逆时针，负为顺时针）

    返回：
        R_new: 旋转后的3x3旋转矩阵
    """
    angle_rad = np.radians(angle_deg)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                 1]
    ])
    R_new = Rz @ R_orig
    return R_new

class VuerTeleop:
    def __init__(self, config_file_path=None):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)
        self.left_retargeting = None
        self.right_retargeting = None
        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        binocular = True
        # self.tv = OpenTeleVision(binocular, self.img_shape, self.shm.name, cert_file="cert.pem", key_file="key.pem", ngrok=False):
        # self.tv = OpenTeleVision(self.img_shape,self.shm.name, image_queue, toggle_streaming,stream_mode = "image",ngrok=False)
        self.tv = TeleVisionWrapper(binocular, self.img_shape, self.shm.name)
        # self.processor = VuerPreprocessor()
        if config_file_path is not None:
            RetargetingConfig.set_default_urdf_dir('assets')
            with Path(config_file_path).open('r') as f:
                cfg = yaml.safe_load(f)
            left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()     
    def step2(self):
        # 使用mat_update函数更新头部、左右手腕的变换矩阵，tv中的矩阵为当前帧的新数据
        head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())  # 头部姿态矩阵
        left_wrist_vuer_mat, left_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.left_hand.copy())  # 左手腕原始矩阵
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())  # 右手腕原始矩阵
        print("right_wrist_vuer_mat", right_wrist_vuer_mat)
        # 坐标系变换：将原始数据从Y轴向上(Y-up)的坐标系转换到Z轴向上(Z-up)的坐标系
        # T_robot_openxr 是Y-up到Z-up的基础变换矩阵
        # 通过相似变换公式：新坐标系下的矩阵 = 基变换矩阵 @ 原矩阵 @ 基变换逆矩阵
        head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat  = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)


        # 计算左手腕相对于头部的变换矩阵
        left_wrist_mat = left_wrist_mat @ (T_to_franka_left_wrist if left_wrist_flag else np.eye(4))
        # 计算右手腕相对于头部的变换矩阵
        right_wrist_mat = right_wrist_mat @ (T_to_franka_right_wrist if right_wrist_flag else np.eye(4))
        # 使手腕位置相对于头部原点（机器人控制需要相对坐标系）
        left_wrist_mat[0:3, 3]  -= head_mat[0:3, 3]
        right_wrist_mat[0:3, 3] -= head_mat[0:3, 3]
        head_rmat = head_mat[:3, :3]
        left_wrist_mat[0, 3] +=0.3
        right_wrist_mat[0,3] +=0.3

        left_wrist_mat[1, 3] -=0.05
        right_wrist_mat[1,3] +=0.05
        # # Z轴正向偏移0.45米（垂直方向补偿）
        left_wrist_mat[2, 3] +=0.45
        right_wrist_mat[2,3] +=0.50
        # right_wrist_mat[:3,:3] = np.array([
        #     [ 1, 0 ,0],
        #     [ 0,-1,0],
        #     [ 0 , 0,-1.]
        # ])
        right_wrist_mat[:3,:3] = rotate_about_y(right_wrist_vuer_mat[:3,:3],180) #rotate_about_y(right_wrist_vuer_mat[:3,:3],+45)
        if right_wrist_mat[2,3] < 0.25:
            print("right_wrist_mat[2,3]",right_wrist_mat[2,3])
            right_wrist_mat[2,3] =0.25
        if right_wrist_mat[0,3] < 0.1:
            right_wrist_mat[0,3] =0.1
        print("right_wrist_mat",right_wrist_mat)
        # 根据需求，若末端低于 base_link，可发出警告或进行保护处理
        return head_rmat, left_wrist_mat, right_wrist_mat, None, None,self.tv.right_pinching,self.tv.left_pinching  
    def franka_step(self):
            """获取处理后的姿态数据
            返回：.gripper
                head_rmat: 头部旋转矩阵（3x3）
                franka_left_wrist: 左手腕姿态矩阵（4x4，franka规范）
                franka_right_wrist: 右手腕姿态矩阵（4x4，franka规范）
                franka_left_hand: 左手关键点坐标（25.gripperx3，franka规范）
                franka_right_hand: 右手关键点坐标（25x3，franka规范）
            """
        # ---------------------- 头部数据处理 ----------------------
            # 获取头部姿态矩阵（基于OpenXR坐标系）
            head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())

            # ---------------------- 腕部数据处理 ----------------------
            # 获取左右手腕原始矩阵（基于OpenXR坐标系）
            left_wrist_vuer_mat, left_wrist_flag  = mat_update(vuer_left_wrist_mat, self.tv.left_hand.copy())
            right_wrist_vuer_mat, right_wrist_flag = mat_update(vuer_right_wrist_mat, self.tv.right_hand.copy())
            print("right_wrist_vuer_mat",right_wrist_vuer_mat)
            # right_wrist_vuer_mat = right_wrist_vuer_mat @ (T_to_franka_right_wrist if right_wrist_flag else np.eye(4))
            # 坐标系基础变换：OpenXR -> 机器人基础坐标系
            # 通过相似变换保持旋转效果在目标坐标系的一致性
            head_mat = T_franka_openxr @ head_vuer_mat @ fast_mat_inv(T_franka_openxr)
            franka_right_wrist = T_franka_openxr @ right_wrist_vuer_mat @ T_franka_openxr.T
            franka_left_wrist = T_franka_openxr @ left_wrist_vuer_mat @ T_franka_openxr.T
            # 坐标平移变换：世界坐标系 -> 头部相对坐标系
            # 使手腕位置相对于头部原点（机器人控制需要相对坐标系）
            franka_left_wrist[0:3, 3]  -= head_mat[0:3, 3]
            franka_right_wrist[0:3, 3] -= head_mat[0:3, 3]

            # ------------------------ 机械臂基座校准 ------------------------
            # 调整手腕位置到机器人躯干坐标系（WAIST关节电机原点）
            # 增加X轴和Z轴偏移量，补偿头部坐标系到躯干坐标系的物理距离
            head_rmat = head_mat[:3, :3]  # 提取头部旋转矩阵
            # X轴正向偏移0.25米（水平方向补偿）
            franka_left_wrist[0, 3] +=0.25
            franka_right_wrist[0,3] +=0.25

            # franka_left_wrist[1, 3] -=0.05
            franka_right_wrist[1,3] +=0.2
            # Z轴正向偏移0.45米（垂直方向补偿）
            franka_left_wrist[2, 3] +=0.45
            franka_right_wrist[2,3] +=0.5
            # franka_right_wrist[:3,:3] = right_wrist_vuer_mat[:3,:3]
            franka_right_wrist[:3,:3] = rotate_about_x(right_wrist_vuer_mat[:3,:3],180)
            if franka_right_wrist[2,3] < 0.05:
                print("franka_right_wrist[2,3]",franka_right_wrist[2,3])
                franka_right_wrist[2,3] =0.05
            if franka_right_wrist[0,3] < 0.1:
                franka_right_wrist[0,3] =0.1
            # franka_right_wrist[:3, :3] = const_right_wrist_vuer_mat[:3, :3]
            return head_rmat, franka_left_wrist, franka_right_wrist, None, None,self.tv.right_pinching,self.tv.left_pinching 
    def step(self):
            """获取处理后的姿态数据
            返回：.gripper
                head_rmat: 头部旋转矩阵（3x3）
                franka_left_wrist: 左手腕姿态矩阵（4x4，franka规范）
                franka_right_wrist: 右手腕姿态矩阵（4x4，franka规范）
                franka_left_hand: 左手关键点坐标（25.gripperx3，franka规范）
                franka_right_hand: 右手关键点坐标（25x3，franka规范）
            """
        # ---------------------- 头部数据处理 ----------------------
            # 获取头部姿态矩阵（基于OpenXR坐标系）
            head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())

            # ---------------------- 腕部数据处理 ----------------------
            # 获取左右手腕原始矩阵（基于OpenXR坐标系）
            left_wrist_vuer_mat, left_wrist_flag  = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())
            right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())
            # 坐标系基础变换：OpenXR -> 机器人基础坐标系
            # 通过相似变换保持旋转效果在目标坐标系的一致性
            head_mat = T_franka_openxr @ head_vuer_mat @ fast_mat_inv(T_franka_openxr)
            left_wrist_mat  = T_franka_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_franka_openxr)
            right_wrist_mat = T_franka_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_franka_openxr)

            # 对齐URDF手腕坐标系定义：
            # 左腕绕x轴旋转-90度，右腕绕x轴旋转+90度（修正轴向差异）
            franka_left_wrist = left_wrist_mat @ (T_to_franka_left_wrist if left_wrist_flag else np.eye(4))
            franka_right_wrist = right_wrist_mat @ (T_to_franka_right_wrist if right_wrist_flag else np.eye(4))

            # 坐标平移变换：世界坐标系 -> 头部相对坐标系
            # 使手腕位置相对于头部原点（机器人控制需要相对坐标系）
            franka_left_wrist[0:3, 3]  -= head_mat[0:3, 3]
            franka_right_wrist[0:3, 3] -= head_mat[0:3, 3]

            # ------------------------ 机械臂基座校准 ------------------------
            # 调整手腕位置到机器人躯干坐标系（WAIST关节电机原点）
            # 增加X轴和Z轴偏移量，补偿头部坐标系到躯干坐标系的物理距离
            head_rmat = head_mat[:3, :3]  # 提取头部旋转矩阵
            # X轴正向偏移0.25米（水平方向补偿）
            # franka_left_wrist[0, 3] +=0.25
            # franka_right_wrist[0,3] +=0.25

            franka_left_wrist[1, 3] -=0.05
            franka_right_wrist[1,3] +=0.3
            # Z轴正向偏移0.45米（垂直方向补偿）
            franka_left_wrist[2, 3] +=0.45
            franka_right_wrist[2,3] +=0.5
            # franka_right_wrist[:3,:3] =  Rx(-np.pi/2)*franka_right_wrist[:3,:3] 
            print("right_wrist_vuer_mat",right_wrist_vuer_mat)
            print("franka_right_wrist",franka_right_wrist)
            # franka_right_wrist[:3, :3] = const_right_wrist_vuer_mat[:3, :3]
            return head_rmat, franka_left_wrist, franka_right_wrist, None, None,self.tv.right_pinching,self.tv.left_pinching 
class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq
        self.previous_left_pose = None
        self.previous_right_pose = None
        self.last_pos =  np.array([
            [ 1, 0 ,0,0.3],
            [ 0,-1,0,0],
            [ 0 , 0,-1.,0.5],
            [ 0. , 0. ,  0.,1.     ]
        ])
        # # 系统初始化
        # desk = panda_py.Desk(hostname, username, password)
        # desk.unlock()
        # desk.activate_fci()
        # # 创建机械臂控制类
        # self.robot = panda_py.Panda(hostname)
        # gripper = libfranka.Gripper(hostname)
        # # 等待直到成功连接机械臂
        # while self.robot is None:
        #     time.sleep(0.01)
        #     self.robot = panda_py.Panda(hostname)
        #     self.gripper = libfranka.Gripper(hostname)
        #     print("[Franka_ArmController] Waiting to connect ...")
        
        self.ik = Franka_Panda_ARMIK()
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        asset_root = "assets"
        right_asset_path = "franka_description/panda_with_gripper.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.dof = self.gym.get_asset_dof_count(right_asset)
        print(f"dof: {self.dof}")
        print(f"所有关节名称:{self.gym.get_asset_joint_names(right_asset)}")
        # 遍历每个 DOF 获取名称
        dof_names = []
        for i in range(self.dof):
            dof_name = self.gym.get_asset_dof_name(right_asset, i)
            dof_names.append(dof_name)

        print(f"所有 DOF 名称: {dof_names}")
        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # left_hand
        # pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        # pose.r = gymapi.Quat(0, 0, 0, 1)
        # self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        # self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
        #                               gymapi.STATE_ALL)
        # left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        # pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.p = gymapi.Vec3(-0.6, 0, 1.1)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        # self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-1.5, 0, 1.6])

        # create left 1st preson viewer
        left_camera_props = gymapi.CameraProperties()
        left_camera_props.width = 1280
        left_camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, left_camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        right_camera_props = gymapi.CameraProperties()
        right_camera_props.width = 1280
        right_camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, right_camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))
    
    def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos,right_pinching,left_pinching):

        if self.print_freq:
            start = time.time()

        # self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        # self.right_root_states[0:6] = torch.tensor(right_pose, dtype=float)
        # self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        right_states = np.zeros(9, dtype=gymapi.DofState.dtype)
        # right_qpos = panda_py.ik(right_pose)
        # if np.isnan(right_qpos[0]):
        #     right_qpos = panda_py.ik(self.last_pos)
        # else:
        #     self.last_pos = right_pose
        # right_states[:7] = right_qpos
        right_states[:7], _ = self.ik.solve_ik(right_pose)
        right_states[7:]  = [0,0] if right_pinching else [0.2,0.2]
        self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)
        # left_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        # left_states['pos'] = left_qpos
        # self.gym.set_actor_dof_states(self.env, self.left_handle, left_states, gymapi.STATE_POS)

        # right_states = np.zeros(self.dof, dtype=gymapi.DofState.dtype)
        # right_states['pos'] = right_qpos
        # self.gym.set_actor_dof_states(self.env, self.right_handle, right_states, gymapi.STATE_POS)

        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))
        # print("left_image",left_image)
        return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == '__main__':
    teleoperator = VuerTeleop()
    simulator = Sim()

    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos,right_pinching,left_pinching  = teleoperator.tv.step() #
            left_img, right_img = simulator.step(head_rmat, left_pose, right_pose, left_qpos, right_qpos,right_pinching,left_pinching)
            head_image = np.hstack((left_img, right_img))
            np.copyto(teleoperator.img_array,head_image )
    except KeyboardInterrupt:
        simulator.end()
        exit(0)