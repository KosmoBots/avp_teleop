import numpy as np
from teleop.open_television.gripper.television import TeleVision
from teleop.open_television.gripper.constants import *
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

class TeleVisionWrapper:
    def __init__(self, binocular, img_shape, img_shm_name):
        """初始化视觉处理模块
        参数：
            binocular: 是否使用双目摄像头
            img_shape: 图像尺寸 (height, width, channels)
            img_shm_name: 共享内存名称
        """
        self.tv = TeleVision(binocular, img_shape, img_shm_name)

    def get_data(self):
        """获取处理后的姿态数据
        返回：.gripper
            head_rmat: 头部旋转矩阵（3x3）
            unitree_left_wrist: 左手腕姿态矩阵（4x4，Unitree规范）
            unitree_right_wrist: 右手腕姿态矩阵（4x4，Unitree规范）
            unitree_left_hand: 左手关键点坐标（25.gripperx3，Unitree规范）
            unitree_right_hand: 右手关键点坐标（25x3，Unitree规范）
        """
        # ================================ 手腕数据处理 ================================
        # 获取原始数据（OpenXR坐标系）
        head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())  # 头部姿态矩阵
        left_wrist_vuer_mat, left_wrist_flag = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())  # 左手腕原始矩阵
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())  # 右手腕原始矩阵
        is_pinching = self.tv.is_pinching.value
        """
        坐标系转换原理[1,7](@ref)：
        执行相似变换：B = P * A * P⁻¹ 
        其中：
        - P = T_robot_openxr（从OpenXR到Robot坐标系的变换矩阵）
        - A = 原始姿态矩阵
        该变换确保在Robot坐标系下的旋转效果与OpenXR坐标系下一致
        """
        head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

        # 转换到Unitree手腕规范（绕x轴旋转90度）
        unitree_left_wrist = left_wrist_mat @ (T_to_unitree_left_wrist if left_wrist_flag else np.eye(4))  # 左手腕逆时针旋转
        unitree_right_wrist = right_wrist_mat @ (T_to_unitree_right_wrist if right_wrist_flag else np.eye(4))  # 右手腕顺时针旋转

        # 坐标系平移：从世界坐标系转换到头部坐标系
        # ------------------------ 手腕位置校准（相对于头部坐标系） ------------------------
        # 将左右手腕的位置从头部坐标系原点平移到世界坐标系（减去头部原点偏移）
        # [0:3, 3] 表示取4x4变换矩阵中的平移分量(x,y,z)
        unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_mat[0:3, 3]  # 左腕位置校准
        unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_mat[0:3, 3]  # 右腕位置校准

        # ------------------------ 手部关键点坐标系转换 ------------------------
        # 将手部关键点转换为齐次坐标系（增加第四维的1）
        # 左/右手部关键点矩阵形状：(4,25)，其中25个关键点，每个点包含[x,y,z,1]
        left_hand_vuer_mat  = np.concatenate([self.tv.left_landmarks.copy().T, np.ones((1, self.tv.left_landmarks.shape[0]))])
        right_hand_vuer_mat = np.concatenate([self.tv.right_landmarks.copy().T, np.ones((1, self.tv.right_landmarks.shape[0]))])

        # 坐标系转换：从OpenXR约定到机器人本体坐标系（应用预定义的变换矩阵）
        # 此处实现跨设备坐标系统一，参考遥操作系统中的运动重定向技术[5](@ref)
        left_hand_mat  = T_robot_openxr @ left_hand_vuer_mat   # 左手坐标系转换
        right_hand_mat = T_robot_openxr @ right_hand_vuer_mat  # 右手坐标系转换

        # ------------------------ 手腕相对坐标系转换 ------------------------
        # 将手部位置从世界坐标系转换到手腕局部坐标系
        # 通过手腕变换矩阵的逆矩阵实现坐标系重定向
        left_hand_mat_wb  = fast_mat_inv(left_wrist_mat) @ left_hand_mat    # 左手腕局部坐标系
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat # 右手腕局部坐标系

        # ------------------------ 手部姿态映射 ------------------------
        # 将手部姿态转换为Unitree机器人专用URDF格式
        # 应用预定义的手部姿态转换矩阵，实现人手到机械手的运动重定向[1](@ref)
        unitree_left_hand  = (T_to_unitree_hand @ left_hand_mat_wb)[0:3, :].T  # 左手关键点(25x3)
        unitree_right_hand = (T_to_unitree_hand @ right_hand_mat_wb)[0:3, :].T # 右手关键点(25x3)

        # ------------------------ 机械臂基座校准 ------------------------
        # 调整手腕位置到机器人躯干坐标系（WAIST关节电机原点）
        # 增加X轴和Z轴偏移量，补偿头部坐标系到躯干坐标系的物理距离
        head_rmat = head_mat[:3, :3]  # 提取头部旋转矩阵
        # X轴正向偏移0.25米（水平方向补偿）
        unitree_left_wrist[0, 3] +=0.15
        unitree_right_wrist[0,3] -=0.15
        unitree_left_wrist[1, 3] +=0.10
        unitree_right_wrist[1,3] -=0.10
        # Z轴正向偏移0.45米（垂直方向补偿）
        unitree_left_wrist[2, 3] +=0.45
        unitree_right_wrist[2,3] +=0.40
        print("right_wrist_vuer_mat",right_wrist_vuer_mat)
        print("unitree_right_wrist",unitree_right_wrist)
        # unitree_right_wrist[:3, :3] = const_right_wrist_vuer_mat[:3, :3]
        return head_rmat, unitree_left_wrist, unitree_right_wrist, unitree_left_hand, unitree_right_hand,is_pinching
    def get_data_2(self):

        # --------------------------------wrist-------------------------------------

        # TeleVision obtains a basis coordinate that is OpenXR Convention
        head_vuer_mat, head_flag = mat_update(const_head_vuer_mat, self.tv.head_matrix.copy())
        left_wrist_vuer_mat, left_wrist_flag  = mat_update(const_left_wrist_vuer_mat, self.tv.left_hand.copy())
        right_wrist_vuer_mat, right_wrist_flag = mat_update(const_right_wrist_vuer_mat, self.tv.right_hand.copy())

        # Change basis convention: VuerMat ((basis) OpenXR Convention) to WristMat ((basis) Robot Convention)
        # p.s. WristMat = T_{robot}_{openxr} * VuerMat * T_{robot}_{openxr}^T
        # Reason for right multiply fast_mat_inv(T_robot_openxr):
        #   This is similarity transformation: B = PAP^{-1}, that is B ~ A
        #   For example:
        #   - For a pose data T_r under the Robot Convention, left-multiplying WristMat means:
        #   - WristMat * T_r  ==>  T_{robot}_{openxr} * VuerMat * T_{openxr}_{robot} * T_r
        #   - First, transform to the OpenXR Convention (The function of T_{openxr}_{robot})
        #   - then, apply the rotation VuerMat in the OpenXR Convention (The function of VuerMat)
        #   - finally, transform back to the Robot Convention (The function of T_{robot}_{openxr})
        #   This results in the same rotation effect under the Robot Convention as in the OpenXR Convention.
        head_mat = T_robot_openxr @ head_vuer_mat @ fast_mat_inv(T_robot_openxr)
        left_wrist_mat  = T_robot_openxr @ left_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)
        right_wrist_mat = T_robot_openxr @ right_wrist_vuer_mat @ fast_mat_inv(T_robot_openxr)

        # Change wrist convention: WristMat ((Left Wrist) XR/AppleVisionPro Convention) to UnitreeWristMat((Left Wrist URDF) Unitree Convention)
        # Reason for right multiply (T_to_unitree_left_wrist) : Rotate 90 degrees counterclockwise about its own x-axis.
        # Reason for right multiply (T_to_unitree_right_wrist): Rotate 90 degrees clockwise about its own x-axis.
        unitree_left_wrist = left_wrist_mat @ (T_to_unitree_left_wrist if left_wrist_flag else np.eye(4))
        unitree_right_wrist = right_wrist_mat @ (T_to_unitree_right_wrist if right_wrist_flag else np.eye(4))

        # Transfer from WORLD to HEAD coordinate (translation only).
        unitree_left_wrist[0:3, 3]  = unitree_left_wrist[0:3, 3] - head_mat[0:3, 3]
        unitree_right_wrist[0:3, 3] = unitree_right_wrist[0:3, 3] - head_mat[0:3, 3]

        # --------------------------------hand-------------------------------------

        # Homogeneous, [xyz] to [xyz1]
        # p.s. np.concatenate([25,3]^T,(1,25)) ==> hand_vuer_mat.shape is (4,25)
        # Now under (basis) OpenXR Convention, mat shape like this:
        #    x0 x1 x2 ··· x23 x24
        #    y0 y1 y1 ··· y23 y24
        #    z0 z1 z2 ··· z23 z24
        #     1  1  1 ···   1   1
        left_hand_vuer_mat  = np.concatenate([self.tv.left_landmarks.copy().T, np.ones((1, self.tv.left_landmarks.shape[0]))])
        right_hand_vuer_mat = np.concatenate([self.tv.right_landmarks.copy().T, np.ones((1, self.tv.right_landmarks.shape[0]))])

        # Change basis convention: from (basis) OpenXR Convention to (basis) Robot Convention
        # Just a change of basis for 3D points. No rotation, only translation. No need to right-multiply fast_mat_inv(T_robot_openxr).
        left_hand_mat  = T_robot_openxr @ left_hand_vuer_mat
        right_hand_mat = T_robot_openxr @ right_hand_vuer_mat

        # Transfer from WORLD to WRIST coordinate. (this process under (basis) Robot Convention)
        # p.s.  HandMat_WristBased = WristMat_{wrold}_{wrist}^T * HandMat_{wrold}
        #       HandMat_WristBased = WristMat_{wrist}_{wrold}   * HandMat_{wrold}, that is HandMat_{wrist}
        left_hand_mat_wb  = fast_mat_inv(left_wrist_mat) @ left_hand_mat
        right_hand_mat_wb = fast_mat_inv(right_wrist_mat) @ right_hand_mat
        # Change hand convention: HandMat ((Left Hand) XR/AppleVisionPro Convention) to UnitreeHandMat((Left Hand URDF) Unitree Convention)
        # Reason for left multiply : T_to_unitree_hand @ left_hand_mat_wb ==> (4,4) @ (4,25) ==> (4,25), (4,25)[0:3, :] ==> (3,25), (3,25).T ==> (25,3)           
        # Now under (Left Hand URDF) Unitree Convention, mat shape like this:
        #    [x0, y0, z0]
        #    [x1, y1, z1]
        #    ···
        #    [x23,y23,z23] 
        #    [x24,y24,z24]               
        unitree_left_hand  = (T_to_unitree_hand @ left_hand_mat_wb)[0:3, :].T
        unitree_right_hand = (T_to_unitree_hand @ right_hand_mat_wb)[0:3, :].T

        # --------------------------------offset-------------------------------------

        head_rmat = head_mat[:3, :3]
        # The origin of the coordinate for IK Solve is the WAIST joint motor. You can use teleop/robot_control/robot_arm_ik.py Unit_Test to check it.
        # The origin of the coordinate of unitree_left_wrist is HEAD. So it is necessary to translate the origin of unitree_left_wrist from HEAD to WAIST.
        unitree_left_wrist[0, 3] +=0.15
        unitree_right_wrist[0,3] +=0.15
        unitree_left_wrist[2, 3] +=0.45
        unitree_right_wrist[2,3] +=0.45

        return head_rmat, unitree_left_wrist, unitree_right_wrist, unitree_left_hand, unitree_right_hand