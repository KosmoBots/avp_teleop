import numpy as np
from teleop.open_television.franka.television import TeleVision
from teleop.open_television.franka.constants import *
from teleop.utils.mat_tool import mat_update, fast_mat_inv

"""
坐标系约定说明（基于不同设备和规范）：
[1,2,7](@ref)
(basis) OpenXR 坐标系：y轴向上，z轴向后，x轴向右（虚拟现实设备通用标准）
(basis) Robot 坐标系：z轴向上，y轴向左，x轴向前（机器人标准坐标系）
注：Vuer系统的原始数据均遵循OpenXR坐标系

手腕初始姿态约定：
# XR/AppleVisionPro 左手腕：
    ▪ x轴从手腕指向中指
    ▪ y轴从食指指向小指
    ▪ z轴从掌心指向手背
# URDF 左手腕（Unitree规范）：
    ▪ x轴从手腕指向中指
    ▪ y轴从掌心指向手背
    ▪ z轴从小指指向食指

手部关键点约定：
# XR/AppleVisionPro 左手：
    ▪ x轴从手腕指向中指
    ▪ y轴从食指指向小指
    ▪ z轴从掌心指向手背
# URDF 左手（Unitree规范）：
    ▪ x轴从掌心指向手背
    ▪ y轴从中指指向手腕
    ▪ z轴从小指指向食指
"""
import numpy as np

def rotate_about_x(R_orig, angle_deg):
    """
    将一个3x3旋转矩阵绕X轴再旋转指定角度（单位为度）

    参数：
        R_orig: 原始3x3旋转矩阵 (numpy.ndarray)
        angle_deg: 绕X轴旋转的角度（单位为度，正为逆时针，负为顺时针）

    返回：
        R_new: 旋转后的3x3旋转矩阵
    """
    angle_rad = np.radians(angle_deg)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    R_new = Rx @ R_orig
    return R_new
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
        [               0.0, 1,              0.0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    R_new = Ry @ R_orig
    return R_new
def rotate_about_z(R_orig, angle_deg):
    """
    将一个3x3旋转矩阵绕Z轴再旋转指定角度（单位为度）

    参数：
        R_orig: 原始3x3旋转矩阵 (numpy.ndarray)
        angle_deg: 绕Z轴旋转的角度（单位为度，正为逆时针，负为顺时针）

    返回：
        R_new: 旋转后的3x3旋转矩阵
    """
    angle_rad = np.radians(angle_deg)  # 角度转弧度
    # 绕Z轴的旋转矩阵（标准右手坐标系，正角度为逆时针）[2,3,6](@ref)
    Rz = np.array([
        [ np.cos(angle_rad), -np.sin(angle_rad), 0 ],  # X' = X*cosθ - Y*sinθ
        [ np.sin(angle_rad),  np.cos(angle_rad), 0 ],  # Y' = X*sinθ + Y*cosθ
        [               0.0,               0.0, 1 ]   # Z轴不变
    ])
    R_new = Rz @ R_orig  # 矩阵乘法：先应用原始旋转，再叠加Z轴旋转
    return R_new
class TeleVisionWrapper:
    def __init__(self, binocular, img_shape, img_shm_name):
        """初始化视觉处理模块
        参数：
            binocular: 是否使用双目摄像头
            img_shape: 图像尺寸 (height, width, channels)
            img_shm_name: 共享内存名称
        """
        self.tv = TeleVision(binocular, img_shape, img_shm_name)
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
        print("right_wrist_vuer_mat",right_wrist_vuer_mat)
        # right_wrist_vuer_mat = right_wrist_vuer_mat @ (T_to_franka_right_wrist if right_wrist_flag else np.eye(4))
        # 坐标系基础变换：OpenXR -> 机器人基础坐标系
        # 通过相似变换保持旋转效果在目标坐标系的一致性
        head_mat = T_franka_openxr @ head_vuer_mat @ fast_mat_inv(T_franka_openxr)
        franka_right_wrist = T_franka_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_franka_openxr)
        franka_left_wrist = T_franka_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_franka_openxr)
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
        if franka_right_wrist[2,3] < 0.05:
            print("franka_right_wrist[2,3]",franka_right_wrist[2,3])
            franka_right_wrist[2,3] =0.05
        if franka_right_wrist[0,3] < 0.1:
            franka_right_wrist[0,3] =0.1
        franka_right_wrist[:3, :3] = franka_right_wrist[:3, :3] @ T_to_franka_wrist
        return head_rmat, franka_left_wrist, franka_right_wrist, None, None,self.tv.right_pinching.value,self.tv.left_pinching.value 